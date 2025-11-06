import os
import time
import numpy as np
import torch
import yaml
import openvino as ov
import nncf  # NNCF Dataset 생성을 위해 임포트
from nncf import quantize
from nncf.parameters import TargetDevice, ModelType # NNCF 파라미터 임포트
import sys # sys.exit 및 sys.path 확인용

# --- [추가됨] train.py의 의존성 ---
import json
import albumentations as A
from torch.utils.data.distributed import DistributedSampler
sys.path.append(os.path.abspath('/workspace/my/KM_UNet'))
from dataset import Dataset
# dataset.py는 main 함수 내에서 sys.path 추가 후 임포트합니다.
# ---------------------------------


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리')
    parser.add_argument('--num_threads', type=int, default=None, help='CPU 스레드 수')
    args = parser.parse_args()
    return args
def load_model_weights_cpu(path, model, strict=True):
    """
    모델 가중치를 CPU로 불러옵니다.
    """
    ckpt = torch.load(path, map_location='cpu') # CPU로 로드
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    
    if any(k.startswith('module.') for k in state.keys()):
        print("DDP로 학습된 모델을 감지했습니다. 'module.' 접두사를 제거합니다.")
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    
    try:
        model.load_state_dict(state, strict=strict)
        print(f"모델 가중치를 {path}에서 (CPU로) 성공적으로 불러왔습니다.")
        return model
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생: {e}")
        if not strict:
            print("strict=False로 다시 시도합니다.")
            model.load_state_dict(state, strict=False)
            print(f"모델 가중치를 strict=False로 {path}에서 (CPU로) 불러왔습니다.")
        else:
            raise e
def load_config_and_model(args):
    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}가 존재하지 않습니다.")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # sys.path 추가 (중요: 'dataset' 모듈 임포트를 위해 필요)
    sys.path.append(os.path.abspath('/workspace/my/KM_UNet'))
    import archs as archs

    model = archs.__dict__[config['arch']](
        config['num_classes'], 3, config['deep_supervision'], embed_dims=config.get('input_list', [128,160,256])
    )
    model_path = os.path.join(args.output_dir, args.name, 'best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.output_dir, args.name, 'last.pth')
        
    if not os.path.exists(model_path):
        print(f"오류: {args.output_dir}/{args.name}에서 'best.pth' 또는 'last.pth'를 찾을 수 없습니다.")
        sys.exit(1)
    
    model = load_model_weights_cpu(model_path, model, strict=True)
    model.eval()
    return model, config

# --- [추가됨] train.py 에서 복사 ---
def make_dataloaders(cfg, distributed, img_ext='_0000.nii.gz', mask_ext='.png'):
    # (train.py의 make_dataloaders 함수 내용과 동일)
    if cfg['dataset'] == 'busi':
        mask_ext = '_mask.png'
    elif cfg['dataset'] in ['glas','cvc','isic2018','isic2017']:
        mask_ext = '.png'
    elif cfg['dataset'] == 'ngtube':
        mask_ext = '.nii.gz'

    # config.yml에 "splits_final" 키가 있어야 함
    with open(cfg["splits_final"]) as f:
        sp = json.load(f)
        train_ids, val_ids = sp[0]["train"], sp[0]["val"]

    train_tf = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])
    val_tf = A.Compose([
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])

    # Dataset 클래스는 main()에서 임포트된 것을 사용
    train_ds = Dataset(train_ids, cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, num_classes=cfg['num_classes'],
                       transform=train_tf)
    val_ds   = Dataset(val_ids,   cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, num_classes=cfg['num_classes'],
                       transform=val_tf)

    dl_common = dict(batch_size=cfg['batch_size'], pin_memory=True)
    if cfg['num_workers'] > 0:
        dl_common.update(dict(
            num_workers=cfg['num_workers'],
            persistent_workers=True,
            prefetch_factor=8
        ))
    else:
        dl_common.update(dict(num_workers=0))

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_ds, sampler=train_sampler, shuffle=False, drop_last=True, **dl_common
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,   sampler=val_sampler,   shuffle=False, drop_last=False, **dl_common
        )
    else:
        train_sampler = val_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_ds, shuffle=True, drop_last=True, **dl_common
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,   shuffle=False, drop_last=False, **dl_common
        )

    return train_loader, val_loader, train_sampler, val_sampler
# ---------------------------------

def openvino_inference(model_xml, input_data, num_threads=None):
    # (기존 코드와 동일)
    ie = ov.Core() 
    if num_threads:
        try:
             ie.set_property("CPU", {"INFERENCE_NUM_THREADS": num_threads})
        except Exception:
             ie.set_property({"CPU_THREADS_NUM": str(num_threads)})

    model = ie.read_model(model=model_xml)
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    start = time.perf_counter()
    result = compiled_model([input_data])[output_layer]
    end = time.perf_counter()

    print(f"Inference time: {(end - start) * 1000:.2f} ms")
    return result

def main():
    args = parse_args()

    # 1. PyTorch 모델 로드
    # (이 함수 내부에서 /workspace/my/KM_UNet이 sys.path에 추가됨)
    model, config = load_config_and_model(args)
    
    # --- [수정됨] 실제 Validation 데이터 로드 ---
    # 1-1. 'dataset' 모듈 임포트
    try:
        from dataset import Dataset
    except ImportError:
        print("오류: 'dataset.py'를 임포트할 수 없습니다.")
        print(f"현재 sys.path: {sys.path}")
        print("load_config_and_model의 sys.path.append 경로를 확인하세요.")
        sys.exit(1)

    # 1-2. Validation DataLoader 생성
    # config.yml에 train.py의 argparse에 해당하는 키가 모두 있어야 함
    # (예: 'dataset', 'splits_final', 'input_h', 'input_w', 'num_classes', 
    # 'image_dir', 'mask_dir', 'batch_size', 'num_workers')
    print("Loading validation dataset for calibration...")
    try:
        _, val_loader, _, _ = make_dataloaders(config, distributed=False)
    except KeyError as e:
        print(f"오류: config.yml 파일에 필요한 키({e})가 없습니다.")
        print("train.py의 argparse와 config.yml의 내용을 확인하세요.")
        sys.exit(1)
    # ---------------------------------------

    # 2. OpenVINO FP32 모델로 변환 (메모리 상에서)
    print("Converting PyTorch model to OpenVINO FP32 model...")
    dummy_input = torch.randn(1, 3, config['input_h'], config['input_w']) # config 값 사용
    ov_model = ov.convert_model(model, example_input=dummy_input)

    # 3. 보정(Calibration) 데이터 준비
    # --- [수정됨] 더미 데이터 대신 val_loader 사용 ---
    print("Preparing calibration dataset from val_loader...")
    
    calib_data = []
    # 캘리브레이션에 사용할 샘플 수 (예: 100개)
    # config.yml에 'stat_subset_size'를 추가하여 제어할 수 있습니다.
    num_samples_to_take = config.get('stat_subset_size', 100)
    samples_collected = 0

    for x_batch, _, _ in val_loader:
        # x_batch는 (B, C, H, W) torch.Tensor
        # NNCF Dataset은 (1, C, H, W) 형태의 numpy 배열 리스트를 받는 것이 편리함
        
        # 배치(B)를 순회하며 (1, C, H, W) 리스트로 분리
        batch_list = [x_batch[i:i+1].numpy() for i in range(x_batch.size(0))]
        
        calib_data.extend(batch_list)
        samples_collected += len(batch_list)
        
        if samples_collected >= num_samples_to_take:
            break
    
    # 정확히 원하는 샘플 수만큼 자르기
    calib_data = calib_data[:num_samples_to_take]
    
    if not calib_data:
        print("오류: 캘리브레이션 데이터를 val_loader에서 수집하지 못했습니다. 데이터셋 경로를 확인하세요.")
        sys.exit(1)
        
    print(f"Collected {len(calib_data)} samples for calibration (shape: {calib_data[0].shape})")
    # ---------------------------------------

    # 3-1. 변환 함수 정의 (데이터를 모델 입력 형식(dict)으로 변환)
    def transform_fn(data_item):
        """
        calib_data 리스트에서 아이템( (1, C, H, W) numpy array )을 하나씩 받아
        모델의 입력 이름(key)과 텐서(value)의 딕셔너리로 변환합니다.
        """
        input_name = ov_model.inputs[0].any_name
        return {input_name: data_item}

    # 3-2. nncf.Dataset 생성
    calibration_dataset = nncf.Dataset(calib_data, transform_fn)
    # ==============

    # 4. NNCF를 사용하여 INT8 양자화 수행
    print("Applying NNCF INT8 Quantization (PTQ)...")
    
    # U-Net 계열 모델이므로 ModelType.CNN 지정
    quantized_model = quantize(
        ov_model,
        calibration_dataset,
        target_device=TargetDevice.CPU 
    )
    # ==============
    
    # 5. 양자화된 모델 저장
    quantized_model_dir = os.path.join(args.output_dir, args.name, "openvino_nncf_int8")
    os.makedirs(quantized_model_dir, exist_ok=True)
    model_xml = os.path.join(quantized_model_dir, "quantized_model.xml")
    
    ov.save_model(quantized_model, model_xml)
    print(f"NNCF INT8 Quantized model saved to: {model_xml}")

    # 6. 추론 테스트
    # 추론 테스트도 캘리브레이션 데이터의 첫 번째 샘플을 사용
    input_data = calib_data[0] 
    print(f"Running inference test with one sample (shape: {input_data.shape})...")
    openvino_inference(model_xml, input_data, num_threads=args.num_threads)

if __name__ == "__main__":
    main()