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
import torch.backends.cudnn as cudnn
from thop import profile
import albumentations as A
from albumentations.core.composition import Compose
import nibabel as nib
from PIL import Image

# test.py에서 metrics.py와 utils.py를 사용하므로 임포트
import archs
from metrics import iou_score 
from utils import AverageMeter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# -----------------------------------------------------------------
#  Dataset 클래스 (test.py의 Dataset 로직)
# -----------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # 이미지 읽기 (nii.gz, 0번째 슬라이스, 3채널 변환)
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = nib.load(img_path).get_fdata()[:, :, 0]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # (H, W, 3)

        # 마스크 읽기 (nii.gz)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        mask = nib.load(mask_path).get_fdata()[:, :]
        mask = mask.astype('uint8') # (H, W)

        # Albumentations 변환 적용
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']  # A.Normalize()에 의해 (3, H, W) numpy가 됨
            mask = augmented['mask'] # (H, W) numpy
        
        # 归一化图像
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)  # 将 HWC 转为 CHW

        # 归一化掩码
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)  # 将 HWC 转为 CHW

        # 마스크 값 0과 1로 정규화
        if mask.max() < 1:
            mask[mask > 0] = 1.0

        # 텐서로 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)
        target_tensor = torch.tensor(mask, dtype=torch.float32)

        return img_tensor, target_tensor, {'img_id': img_id}

# -----------------------------------------------------------------
#  유틸리티 함수
# -----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GPU (BF16) Performance Script")
    parser.add_argument('--name', default=None, required=True, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리 (config.yml 로드용)')
    parser.add_argument('--test_image_dir', type=str, required=True, help='Test 이미지 폴더 경로')
    parser.add_argument('--test_mask_dir', type=str, required=True, help='Test 마스크 폴더 경로')
    args = parser.parse_args()
    return args

def load_model_weights(path, model, device='cuda'):
    map_location = torch.device(device)
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    
    # --- [수정된 부분] ---
    # (k for k in state.keys())가 필요합니다.
    if any(k.startswith('module.') for k in state.keys()):
    # --------------------
        print("DDP로 학습된 모델을 감지했습니다. 'module.' 접두사를 제거합니다.")
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    
    try:
        model.load_state_dict(state, strict=True)
        print(f"모델 가중치를 {path}에서 ({device}로) 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생 (strict=True): {e}")
        print("strict=False로 다시 시도합니다.")
        model.load_state_dict(state, strict=False)
        print(f"모델 가중치를 strict=False로 {path}에서 ({device}로) 불러왔습니다.")

def make_test_loader(cfg, test_image_dir, test_mask_dir, img_ext, mask_ext):
    print(f"{test_image_dir}에서 '{img_ext}' 확장자를 가진 이미지 파일을 스캔합니다...")
    img_paths = sorted(glob(os.path.join(test_image_dir, '*' + img_ext)))
    
    if not img_paths:
        raise ValueError(f"{test_image_dir}에서 '{img_ext}' 확장자를 가진 파일을 찾을 수 없습니다.")

    test_ids = []
    for p in img_paths:
        filename = os.path.basename(p)
        if filename.endswith(img_ext):
            base_id = filename[:-len(img_ext)]
            test_ids.append(base_id)
        else:
            base_id = os.path.splitext(filename)[0]
            test_ids.append(base_id)
    
    print(f"{len(test_ids)}개의 Test ID를 찾았습니다 (예: {test_ids[0]}).")

    test_tf = A.Compose([
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])

    test_ds = Dataset(
        img_ids=test_ids,
        img_dir=test_image_dir,
        mask_dir=test_mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=cfg['num_classes'],
        transform=test_tf
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1, # 성능 측정을 위해 배치 1로 고정
        shuffle=False, 
        num_workers=cfg.get('num_workers', 2), # config에 num_workers가 없을 경우 대비
        pin_memory=True,
        drop_last=False
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} 또는 'best.pth'를 찾을 수 없습니다.")
    return model_path

def print_metrics(name, iou_meter, dice_meter, hd95_meter):
    """성능 측정 결과를 출력합니다."""
    iou = iou_meter.avg
    dice = dice_meter.avg
    hd95 = hd95_meter.avg
    
    print(f"\n--- 결과: {name} ---")
    print(f"  - IoU: {iou:.4f}")
    print(f"  - Dice: {dice:.4f}")
    if hd95 > 0:
        print(f"  - HD95: {hd95:.4f}")

# -----------------------------------------------------------------
#  시나리오별 테스트 함수
# -----------------------------------------------------------------

def test_gpu_bf16(config, test_loader, model_path):
    print("\n[시작] 2. GPU (BF16) 성능 측정")
    
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("BF16이 현재 GPU에서 지원되지 않습니다. 건너뜁니다.")
        return

    model = get_model(config)
    load_model_weights(model_path, model, device='cuda')
    model.cuda().eval()
    
    # 성능 측정기
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    
    # [수정] BF16 autocast 활성화
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        print(f"Testing (BF16) on {len(test_loader)} images...")
        for input_tensor, target_tensor, _ in tqdm(test_loader, desc="GPU BF16"):
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
            
            output = model(input_tensor)
            
            # 성능 계산
            try:
                # iou_score가 bf16 출력을 처리할 수 있어야 함 (보통은 문제 없음)
                iou, dice, hd95_ = iou_score(output, target_tensor)
                iou_avg_meter.update(iou, input_tensor.size(0))
                dice_avg_meter.update(dice, input_tensor.size(0))
                hd95_avg_meter.update(hd95_, input_tensor.size(0))
            except ValueError:
                iou, dice, _ = iou_score(output, target_tensor)
                iou_avg_meter.update(iou, input_tensor.size(0))
                dice_avg_meter.update(dice, input_tensor.size(0))
                hd95_avg_meter.update(0, input_tensor.size(0)) # HD95 계산 실패 시
            
    # 최종 결과 출력
    print_metrics("GPU (BF16)", iou_avg_meter, dice_avg_meter, hd95_avg_meter)

# -----------------------------------------------------------------
#  메인 실행 함수
# -----------------------------------------------------------------

def main():
    args = parse_args()
    cudnn.benchmark = True # GPU 벤치마크 모드 활성화

    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    print(f"실험 {args.name}의 GPU (BF16) 성능 측정 시작")
    print(f"  - Test Image Dir: {args.test_image_dir}")
    print(f"  - 입력 크기: {config['input_channels']}x{config['input_h']}x{config['input_w']}")
    print('-'*20)

    # --- GFLOPs 및 파라미터 측정 (참고용) ---
    model_f32_cpu = get_model(config).eval()
    total_params = sum(p.numel() for p in model_f32_cpu.parameters())
    print(f"모델 파라미터 (Total Params): {total_params / 1e6:.2f} M")
    
    try:
        dummy_input_cpu = torch.randn(1, config['input_channels'], config['input_h'], config['input_w'])
        flops, _ = profile(model_f32_cpu, inputs=(dummy_input_cpu, ), verbose=False)
        print(f"기준 GFLOPs (f32): {flops / 1e9:.2f} G")
    except Exception as e:
        print(f"GFLOPs 측정 실패: {e}")
    del model_f32_cpu

    # --- 데이터 로더 및 모델 경로 준비 ---
    img_ext = '_0000.nii.gz'
    mask_ext = '.nii.gz'
    if config['dataset'] == 'busi':
        img_ext = '_0000.nii.gz'
        mask_ext = '.png' 
        print("Busi 데이터셋 감지: img_ext='_0000.nii.gz', mask_ext='.png' 사용")

    test_loader = make_test_loader(
        config, 
        args.test_image_dir, 
        args.test_mask_dir, 
        img_ext, 
        mask_ext
    )
    model_path = get_model_path(args.output_dir, args.name)

    # --- 테스트 실행 ---
    test_gpu_bf16(config, test_loader, model_path)


if __name__ == '__main__':
    main()
