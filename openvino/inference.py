import argparse
import os
import time
import numpy as np
import yaml
import json
from tqdm import tqdm
from glob import glob
import cv2
import sys

import torch
import torch.backends.cudnn as cudnn
import albumentations as A
from albumentations.core.composition import Compose
import nibabel as nib
from PIL import Image

# OpenVINO Core 임포트
import openvino as ov

# --- [필수] KM_UNet 프로젝트 경로 추가 ---
# (metrics.py, utils.py 임포트를 위해)
sys.path.append(os.path.abspath('/workspace/my/KM_UNet'))
try:
    from metrics import iou_score
    from utils import AverageMeter
except ImportError as e:
    print(f"오류: 'metrics.py' 또는 'utils.py'를 임포트할 수 없습니다.")
    print(f"경로: /workspace/my/KM_UNet에 해당 파일이 있는지 확인하세요. (오류: {e})")
    sys.exit(1)
# ------------------------------------


# -----------------------------------------------------------------
#  Dataset 클래스 (IPEX 스크립트에서 가져옴 + 버그 수정)
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
            img = augmented['image']  # A.Normalize() -> (H, W, 3) numpy
            mask = augmented['mask'] # (H, W) numpy
        
        # 이미지: HWC -> CHW
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)  # (3, H, W)

        # 마스크: HWC -> CHW (1, H, W)
        mask = mask.astype('float32')
        # [수정] 2D 마스크(H, W)를 (1, H, W) 텐서 형태로 변환
        img = img.transpose(2, 0, 1)

        # 마스크 값 0과 1로 정규화
        if mask.max() < 1:
            mask[mask > 0] = 1.0

        # 텐서로 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)
        target_tensor = torch.tensor(mask, dtype=torch.float32)

        return img_tensor, target_tensor, {'img_id': img_id}

# -----------------------------------------------------------------
#  유틸리티 함수 (IPEX 스크립트에서 가져옴)
# -----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="OpenVINO (INT8) Evaluation Script")
    
    # [수정] --model_xml : 평가할 OpenVINO 모델 경로
    parser.add_argument('--model_xml', type=str, required=True, 
                        help='평가할 OpenVINO INT8 모델(.xml) 파일 경로')
    
    parser.add_argument('--name', default=None, required=True, 
                        help='실험(모델) 이름 (config.yml 로드용)')
    parser.add_argument('--output_dir', default='outputs', 
                        help='출력 디렉토리 (config.yml 로드용)')
    parser.add_argument('--test_image_dir', type=str, required=True, 
                        help='Test 이미지 폴더 경로')
    parser.add_argument('--test_mask_dir', type=str, required=True, 
                        help='Test 마스크 폴더 경로')
    parser.add_argument('--num_threads', type=int, default=None, 
                        help='사용할 CPU 스레드 수 (OpenVINO용)')
    args = parser.parse_args()
    return args

def make_test_loader(cfg, test_image_dir, test_mask_dir, img_ext, mask_ext):
    # (IPEX 스크립트와 동일)
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
        mask_dir=test_mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=cfg['num_classes'],
        transform=test_tf
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1, # 평가는 배치 1로 고정
        shuffle=False, 
        num_workers=cfg.get('num_workers', 0),
        pin_memory=False, 
        drop_last=False
    )
    return test_loader

def print_metrics(name, iou_meter, dice_meter, hd95_meter):
    # (IPEX 스크립트와 동일)
    iou = iou_meter.avg
    dice = dice_meter.avg
    hd95 = hd95_meter.avg
    
    print(f"\n--- 결과: {name} ---")
    print(f"  - IoU: {iou:.4f}")
    print(f"  - Dice: {dice:.4f}")
    if hd95 > 0:
        print(f"  - HD95: {hd95:.4f}")

# -----------------------------------------------------------------
#  메인 실행 함수
# -----------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Config 로드
    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # (평가 시 입력 크기를 강제해야 한다면 여기서 수정, 예: config['input_h'] = 512)
    
    print('-'*20)
    print(f"실험 {args.name}의 OpenVINO (INT8) 성능 측정 시작")
    print(f"  - 모델: {args.model_xml}")
    print(f"  - Test Image Dir: {args.test_image_dir}")
    print(f"  - 입력 크기: {config['input_channels']}x{config['input_h']}x{config['input_w']}")
    print('-'*20)

    # 2. 데이터 로더 준비 (IPEX 스크립트 방식)
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

    # 3. OpenVINO 모델 로드
    print("Loading OpenVINO model...")
    ie = ov.Core()
    if args.num_threads:
        try:
             ie.set_property("CPU", {"INFERENCE_NUM_THREADS": args.num_threads})
             print(f"OpenVINO CPU 스레드 수를 {args.num_threads}개로 제한합니다.")
        except Exception:
             ie.set_property({"CPU_THREADS_NUM": str(args.num_threads)})
             print(f"OpenVINO CPU 스레드 수를 {args.num_threads}개로 제한합니다. (레거시 방식)")

    model = ie.read_model(model=args.model_xml)
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    # [수정] 모델의 실제 입력/출력 레이어 가져오기
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print("OpenVINO model compiled.")

    # 4. 성능 측정
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    
    # torch.no_grad() : iou_score 계산 시 불필요한 그래디언트 계산 방지
    with torch.no_grad():
        print(f"Testing (OpenVINO INT8) on {len(test_loader)} images...")
        
        for input_tensor, target_tensor, _ in tqdm(test_loader, desc="OpenVINO INT8"):
            
            # (B, C, H, W) PyTorch 텐서를 (B, C, H, W) NumPy 배열로 변환
            input_data = input_tensor.numpy() 
            
            # [수정] OpenVINO 추론 실행
            # compiled_model([input_data])는 딕셔너리를 반환함
            result_numpy = compiled_model([input_data])[output_layer]
            
            # [수정] 결과(NumPy)를 PyTorch 텐서로 변환 (iou_score 입력을 위해)
            output_tensor = torch.from_numpy(result_numpy)

            try:
                # target_tensor는 이미 PyTorch 텐서
                iou, dice, hd95_ = iou_score(output_tensor, target_tensor)
                iou_avg_meter.update(iou, input_data.shape[0])
                dice_avg_meter.update(dice, input_data.shape[0])
                hd95_avg_meter.update(hd95_, input_data.shape[0])
            except ValueError:
                iou, dice, _ = iou_score(output_tensor, target_tensor)
                iou_avg_meter.update(iou, input_data.shape[0])
                dice_avg_meter.update(dice, input_data.shape[0])
                hd95_avg_meter.update(0, input_data.shape[0])
            
    # 5. 최종 결과 출력
    print_metrics("OpenVINO (INT8)", iou_avg_meter, dice_avg_meter, hd95_avg_meter)


if __name__ == '__main__':
    main()