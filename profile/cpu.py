import argparse
import os
import time
import numpy as np
import yaml
import json
from tqdm import tqdm
from glob import glob
import cv2

import torch
# ----------------------------
# 프로파일러 임포트
# ----------------------------
import torch.profiler
# ----------------------------

import albumentations as A
from albumentations.core.composition import Compose
import nibabel as nib
from PIL import Image

import archs
from metrics import iou_score # 임포트는 하지만 사용 X
from utils import AverageMeter # 임포트는 하지만 사용 X

# -----------------------------------------------------------------
#  Dataset 클래스 (마스크 로딩 생략)
# -----------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = nib.load(img_path).get_fdata()[:, :, 0]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # (H, W, 3)
        
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        img_tensor = torch.tensor(img, dtype=torch.float32)
        return img_tensor

# -----------------------------------------------------------------
#  유틸리티 함수 (parse_args, get_model 등... 이전과 동일)
# -----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CPU (F32) Layer-wise Profiler")
    parser.add_argument('--name', default=None, required=True, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리 (config.yml 로드용)')
    parser.add_argument('--test_image_dir', type=str, required=True, help='Test 이미지 폴더 경로')
    parser.add_argument('--num_threads', type=int, default=None, 
                        help='사용할 CPU 스레드 수 (기본값: 모두 사용)')
    args = parser.parse_args()
    return args

def load_model_weights_cpu(path, model, device='cpu'):
    map_location = torch.device(device)
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    
    if any(k.startswith('module.') for k in state.keys()):
        print("DDP로 학습된 모델을 감지했습니다. 'module.' 접두사를 제거합니다.")
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print("strict=False로 다시 시도합니다.")
        model.load_state_dict(state, strict=False)
    print(f"모델 가중치를 {path}에서 ({device}로) 성공적으로 불러왔습니다.")


def make_test_loader(cfg, test_image_dir, img_ext):
    print(f"{test_image_dir}에서 '{img_ext}' 확장자를 가진 이미지 파일을 스캔합니다...")
    img_paths = sorted(glob(os.path.join(test_image_dir, '*' + img_ext)))
    
    if not img_paths:
        raise ValueError(f"{test_image_dir}에서 '{img_ext}' 확장자를 가진 파일을 찾을 수 없습니다.")

    test_ids = [os.path.basename(p)[:-len(img_ext)] for p in img_paths]
    print(f"{len(test_ids)}개의 Test ID를 찾았습니다 (예: {test_ids[0]}).")

    test_tf = A.Compose([
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])

    test_ds = Dataset(
        img_ids=test_ids,
        img_dir=test_image_dir,
        img_ext=img_ext,
        transform=test_tf
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, 
        num_workers=cfg.get('num_workers', 0), pin_memory=False
    )
    return test_loader

def get_model(config):
    return archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'], 
        config['deep_supervision'], 
        embed_dims=config.get('input_list', [128,160,256])
    )

def get_model_path(output_dir, name):
    model_path = os.path.join(output_dir, name, 'best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dir, name, 'last.pth')
    return model_path

# -----------------------------------------------------------------
#  메인 실행 함수
# -----------------------------------------------------------------
def main():
    args = parse_args()
    
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
        print(f"CPU 스레드 수를 {args.num_threads}개로 제한합니다.")

    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    print(f"실험 {args.name}의 CPU (F32) 레이어 프로파일링 시작")
    print(f"  - Test Image Dir: {args.test_image_dir}")
    print('-'*20)

    # --- 모델 로드 (CPU) ---
    model = get_model(config).cpu().eval()
    model_path = get_model_path(args.output_dir, args.name)
    load_model_weights_cpu(model_path, model, device='cpu')

    # --- 데이터 로더 준비 ---
    img_ext = '.nii.gz'
    if config['dataset'] == 'busi':
        img_ext = '_0000.nii.gz'
    test_loader = make_test_loader(config, args.test_image_dir, img_ext)

    # --- 프로파일링 ---
    # 1. Warmup
    print("Warming up CPU...")
    loader_iter = iter(test_loader)
    for _ in range(5): 
        try:
            input_tensor = next(loader_iter).cpu()
        except StopIteration:
            loader_iter = iter(test_loader)
            input_tensor = next(loader_iter).cpu()
            
        with torch.no_grad():
            _ = model(input_tensor)
    print("Warmup complete. Starting profiler...")

    # 2. Profiler 실행 (CUDA Activity 제외)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2), # 10회 측정
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{args.name}_cpu'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        
        loader_iter = iter(test_loader) # 로더 다시 시작
        for i in tqdm(range(10), desc="Profiling"):
            try:
                input_tensor = next(loader_iter).cpu()
            except StopIteration:
                print("프로파일링 중 데이터셋 끝. 중단합니다.")
                break
                
            with torch.no_grad():
                _ = model(input_tensor)
            
            prof.step() 

    print("\nProfiling complete.")
    
    # 3. 콘솔에 요약 테이블 출력
    print("---[ CPU 연산 시간 Top 20 (Self CPU Time) ]---")
    # self_cpu_time_total: CPU에서 순수하게 실행된 시간 기준 정렬
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total", row_limit=20
    ))
    
    print("\n---[ 연산자별 요약 (Operator Summary) ]---")
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=20
    ))

    # 4. JSON 트레이스 파일 저장
    trace_path = f"{args.name}_cpu_profile.json"
    prof.export_chrome_trace(trace_path)
    print(f"\n상세 타임라인이 {trace_path} 에 저장되었습니다.")
    print(f"Chrome 브라우저에서 'chrome://tracing'을 열어 위 파일을 로드하세요.")


if __name__ == '__main__':
    main()