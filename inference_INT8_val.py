import argparse
import os
import time
import numpy as np
import yaml
import json  # [추가] Validation Split을 읽기 위해
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

# sys.path가 필요할 경우 주석 해제 (archs, metrics, utils 임포트를 위해)
# import sys
# sys.path.append(os.path.abspath('/path/to/your/project_root'))
import archs
from metrics import iou_score 
from utils import AverageMeter

# IPEX 임포트
try:
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import prepare, convert, default_static_qconfig_mapping
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    print("경고: Intel Extension for PyTorch (IPEX)를 찾을 수 없습니다. CPU INT8 테스트가 실패합니다.")


# -----------------------------------------------------------------
#  Dataset 클래스
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
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # (H, W, 3)Q

        # 마스크 읽기 (nii.gz)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        mask = nib.load(mask_path).get_fdata()[:, :]
        mask = mask.astype('uint8') # (H, W)

        # Albumentations 변환 적용
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']  # A.Normalize() -> (H, W, 3) numpy
            mask = augmented['mask'] # (H, W) numpy
        
        # 이미지 HWC -> CHW
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)  # (3, H, W)

        # --- [수정] ---
        # 归一化掩码 (마스크 HWC -> CHW)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)  # 将 HWC 转为 CHW
        # (기존) mask = mask.transpose(2, 0, 1) # (H, W) 2D 배열에 3D transpose 적용 불가
        # (변경)
        # -------------

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
    # (기존과 동일)
    parser = argparse.ArgumentParser(description="CPU (INT8) Performance Script")
    parser.add_argument('--name', default=None, required=True, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리 (config.yml 로드용)')
    parser.add_argument('--test_image_dir', type=str, required=True, help='Test 이미지 폴더 경로')
    parser.add_argument('--test_mask_dir', type=str, required=True, help='Test 마스크 폴더 경로')
    parser.add_argument('--num_threads', type=int, default=None, 
                        help='사용할 CPU 스레드 수 (기본값: 모두 사용)')
    args = parser.parse_args()
    return args

def load_model_weights_cpu(path, model, device='cpu'):
    # (기존과 동일)
    map_location = torch.device(device)
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    
    if any(k.startswith('module.') for k in state.keys()):
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

# --- [신규 추가] ---
def make_val_loader(cfg, img_ext, mask_ext):
    """
    config.yml에 저장된 'splits_final'과 'image_dir'/'mask_dir'를 기반으로
    캘리브레이션에 사용할 Validation Loader를 생성합니다.
    """
    print(f"\n[캘리브레이션용] Validation Split 로드: {cfg['splits_final']}")
    try:
        with open(cfg["splits_final"]) as f:
            sp = json.load(f)
            # train.py와 동일하게 첫 번째 split 사용
            val_ids = sp[0]["val"]
    except KeyError as e:
        print(f"오류: config.yml에 {e} 키가 없습니다. (train.py 설정과 동일해야 함)")
        raise e
    except Exception as e:
        print(f"오류: Validation split JSON '{cfg['splits_final']}' 로드 실패: {e}")
        raise e
    
    print(f"{len(val_ids)}개의 Validation ID를 찾았습니다 (예: {val_ids[0]}).")

    val_tf = A.Compose([
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])

    # config.yml에 저장된 'image_dir'와 'mask_dir' (학습 시 경로) 사용
    val_ds = Dataset(
        img_ids=val_ids,
        img_dir=cfg['image_dir'],
        mask_dir=cfg['mask_dir'],
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=cfg['num_classes'],
        transform=val_tf
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1, # 캘리브레이션은 배치 1로
        shuffle=False, 
        num_workers=cfg.get('num_workers', 0),
        pin_memory=False, 
        drop_last=False
    )
    return val_loader
# --- [신규 추가 완료] ---

def make_test_loader(cfg, test_image_dir, test_mask_dir, img_ext, mask_ext):
    # (기존과 동일)
    print(f"\n[성능측정용] {test_image_dir}에서 '{img_ext}' 확장자를 가진 이미지 파일을 스캔합니다...")
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
        batch_size=1, 
        shuffle=False, 
        num_workers=cfg.get('num_workers', 0),
        pin_memory=False, 
        drop_last=False
    )
    return test_loader

def get_model(config):
    # (기존과 동일)
    return archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'], 
        config['deep_supervision'], 
        embed_dims=config.get('input_list', [128,160,256])
    )

def get_model_path(output_dir, name):
    # (기존과 동일)
    model_path = os.path.join(output_dir, name, 'best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dir, name, 'last.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} 또는 'best.pth'를 찾을 수 없습니다.")
    return model_path

def print_metrics(name, iou_meter, dice_meter, hd95_meter):
    # (기존과 동일)
    iou = iou_meter.avg
    dice = dice_meter.avg
    hd9 = hd95_meter.avg
    
    print(f"\n--- 결과: {name} ---")
    print(f"  - IoU: {iou:.4f}")
    print(f"  - Dice: {dice:.4f}")
    if hd95 > 0:
        print(f"  - HD95: {hd95:.4f}")

# -----------------------------------------------------------------
#  시나리오별 테스트 함수
# -----------------------------------------------------------------

# --- [수정] ---
# val_loader를 인자로 추가
def test_cpu_int8(config, val_loader, test_loader, model_path, num_threads):
    print("\n[시작] 3. CPU (INT8) 성능 측정")
    if not IPEX_AVAILABLE:
        print("Intel Extension for PyTorch (IPEX)가 설치되지 않았습니다. 건너뜁니다.")
        return

    if num_threads:
        torch.set_num_threads(num_threads)
        print(f"CPU 스레드 수를 {num_threads}개로 제한합니다.")

    # 1. CPU로 f32 모델 로드
    model = get_model(config)
    load_model_weights_cpu(model_path, model, device='cpu')
    model.eval()

    # 2. IPEX INT8 정적 양자화
    print("Applying IPEX INT8 Static Quantization...")
    model = model.to(memory_format=torch.channels_last)
    
    qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
    
    # --- [수정] ---
    print("Calibrating model with 10 samples from validation loader...")
    calib_data = []
    # (기존) for i, (input_tensor, _, _) in enumerate(test_loader):
    # (변경) val_loader 사용
    for i, (input_tensor, _, _) in enumerate(val_loader): 
        if i >= 10: # 캘리브레이션 샘플 수는 10개로 제한
            break
        calib_data.append(input_tensor.to(memory_format=torch.channels_last))
    
    if not calib_data:
        print("오류: 캘리브레이션 데이터를 (val_loader에서) 로드할 수 없습니다.")
        return
    
    prepared_model = prepare(model, qconfig_mapping, example_inputs=calib_data[0], inplace=False)
    
    with torch.no_grad():
        for data in tqdm(calib_data, desc="Calibration"):
            prepared_model(data)
            
    model = convert(prepared_model) # INT8 모델로 변환
    print("INT8 Quantization applied.")
    # -----------------------------

    # 3. INT8 성능 측정 (test_loader 사용 - 기존과 동일)
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    
    with torch.no_grad():
        print(f"\nTesting (INT8) on {len(test_loader)} images (from test_loader)...")
        for input_tensor, target_tensor, _ in tqdm(test_loader, desc="CPU INT8 Test"):
            input_tensor = input_tensor.to(memory_format=torch.channels_last)
            
            output = model(input_tensor)
            
            try:
                iou, dice, hd95_ = iou_score(output, target_tensor)
                iou_avg_meter.update(iou, input_tensor.size(0))
                dice_avg_meter.update(dice, input_tensor.size(0))
                hd95_avg_meter.update(hd95_, input_tensor.size(0))
            except ValueError:
                iou, dice, _ = iou_score(output, target_tensor)
                iou_avg_meter.update(iou, input_tensor.size(0))
                dice_avg_meter.update(dice, input_tensor.size(0))
                hd95_avg_meter.update(0, input_tensor.size(0))
            
    print_metrics("CPU (INT8)", iou_avg_meter, dice_avg_meter, hd95_avg_meter)

# -----------------------------------------------------------------
#  메인 실행 함수
# -----------------------------------------------------------------

def main():
    args = parse_args()

    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # [주의] config.yml에 'input_h'/'input_w'가 있어야 합니다.
    # (train.py에서 저장했다면) 1024x1024가 로드됩니다.
    # 만약 512x512로 테스트해야 한다면 이 주석을 해제하세요.
    # config['input_h'] = 512
    # config['input_w'] = 512
    
    print('-'*20)
    print(f"실험 {args.name}의 CPU (INT8) 성능 측정 시작")
    print(f"  - [성능측정용] Test Image Dir: {args.test_image_dir}")
    print(f"  - [캘리브레이션용] Train Split File: {config.get('splits_final', 'N/A')}")
    print(f"  - 입력 크기: {config['input_channels']}x{config['input_h']}x{config['input_w']}")
    print('-'*20)

    # (GFLOPs, 파라미터 측정 - 기존과 동일)
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

    # (데이터셋 확장자 설정 - 기존과 동일)
    img_ext = '_0000.nii.gz'
    mask_ext = '.nii.gz'
    if config['dataset'] == 'busi':
        img_ext = '_0000.nii.gz'
        mask_ext = '.png' 
        print("Busi 데이터셋 감지: img_ext='_0000.nii.gz', mask_ext='.png' 사용")

    # --- [수정] ---
    # 1. 캘리브레이션용 Validation Loader 생성
    # (config.yml에 'image_dir', 'mask_dir', 'splits_final' 등이 있어야 함)
    try:
        val_loader = make_val_loader(
            config,
            img_ext,
            mask_ext
        )
    except Exception as e:
        print(f"\n오류: Validation Loader 생성 실패. config.yml 파일의 경로 설정을 확인하세요.")
        print(f"({e})")
        return

    # 2. 성능 측정용 Test Loader 생성
    test_loader = make_test_loader(
        config, 
        args.test_image_dir, 
        args.test_mask_dir, 
        img_ext, 
        mask_ext
    )

    # 3. 모델 경로 가져오기
    model_path = get_model_path(args.output_dir, args.name)

    # 4. 캘리브레이션 및 테스트 실행
    test_cpu_int8(config, val_loader, test_loader, model_path, args.num_threads)
    # --- [수정 완료] ---


if __name__ == '__main__':
    main()