import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from thop import profile
import sys
import os
sys.path.append(os.path.abspath('/workspace/my/KM_UNet'))
import archs_fast_cls as archs

# IPEX 임포트 (quantization 포함)
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert, default_static_qconfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리')
    parser.add_argument('--num_threads', type=int, default=None, 
                        help='사용할 CPU 스레드 수 (기본값: PyTorch/IPEX가 모두 사용)')
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
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생: {e}")
        if not strict:
            print("strict=False로 다시 시도합니다.")
            model.load_state_dict(state, strict=False)
            print(f"모델 가중치를 strict=False로 {path}에서 (CPU로) 불러왔습니다.")
        else:
            raise e

def main():
    args = parse_args()

    # --- CPU 스레드 수 설정 ---
    if args.num_threads is not None and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        print(f"CPU 스레드 수를 {args.num_threads}개로 제한합니다.")
    else:
        print("CPU 스레드 수를 제한하지 않습니다 (기본값 사용).")

    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    if not os.path.exists(config_path):
        print(f"오류: {config_path}에서 config 파일을 찾을 수 없습니다.")
        return

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    print(f"실험 {args.name}의 Config 로드 중 (CPU - IPEX INT8 모드)")
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    config['input_channels'] = 3
    # 모델 생성 (CPU, f32)
    model = archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'], 
        config['deep_supervision'], 
        embed_dims=config.get('input_list', [128,160,256])
    )
    model.eval() 

    # 모델 로드 (best.pth 사용, CPU로)
    # model_path = os.path.join(args.output_dir, args.name, 'best.pth')
    # if not os.path.exists(model_path):
    #     model_path = os.path.join(args.output_dir, args.name, 'last.pth')
    
    # load_model_weights_cpu(model_path, model, strict=True)
    model.eval()

    # ----------------------------------------
    # 1. 파라미터 수 측정 (f32)
    # ----------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    
    # f32 모델의 디스크상 크기 (참고용)
    
    print(f"총 파라미터 수 (Total Params) (f32): {total_params / 1e6:.2f} M")
    print(f"학습 가능 파라미터 수 (Trainable Params) (f32): {trainable_params / 1e6:.2f} M")
    print(f"총 모델 크기 (디스크상, f32): {total_bytes / 1e6:.2f} MB")
    print(f"학습 가능 모델 크기 (디스크상, f32): {trainable_bytes / 1e6:.2f} MB")


    # ----------------------------------------
    # 2. GFLOPs 측정 (f32, 최적화 전)
    # ----------------------------------------
    h = 1024
    w = 1024
    c = config['input_channels']
    
    dummy_input = torch.randn(1, c, h, w).cpu() # f32 더미 입력
    
    try:
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False) 
        gflops = flops / 1e9
        print(f"GFLOPs (CPU, f32, 최적화 전, 입력 크기 {c}x{h}x{w}): {gflops:.2f} G")
    except Exception as e:
        print(f"GFLOPs 측정 중 오류 발생: {e}")
        
    # ----------------------------------------
    # IPEX INT8 정적 양자화 적용
    # ----------------------------------------
    print("\nApplying Intel Extension for PyTorch (IPEX) INT8 Static Quantization...")
    
    # INT8 양자화를 위해 Channels Last 메모리 형식을 권장합니다.
    model = model.to(memory_format=torch.channels_last)
    dummy_input = dummy_input.to(memory_format=torch.channels_last)

    try:
        # 1. Quantization Config 정의
        # default_static_qconfig는 CNN에 적합한 'per_channel' 설정을 사용합니다.
        qconfig = ipex.quantization.default_static_qconfig 
        
        # 2. 모델 준비 (Observer 삽입)
        # example_inputs를 제공하여 IPEX가 모델 구조를 추적할 수 있도록 함
        prepared_model = prepare(model, qconfig, example_inputs=dummy_input, inplace=False)
        
        # 3. 보정 (Calibration)
        # 샘플 데이터를 통과시켜 Observer가 값의 범위를 수집하도록 함.
        # 실제 데이터셋을 사용하는 것이 가장 좋지만, 여기서는 더미 입력으로 대체합니다.
        print(f"Calibrating model with 1 iterations...")
        with torch.no_grad():
            for _ in range(1):
                prepared_model(dummy_input)
        
        # 4. 변환 (Convert)
        # 보정된 Observer를 기반으로 모델을 INT8로 변환.
        # 이 함수는 내부적으로 JIT 추적을 포함합니다.
        model = convert(prepared_model)
        
        print("IPEX INT8 Static Quantization applied successfully.")

        # ----------------------------------------
        # [추가된 부분] INT8 모델 파라미터 및 크기 측정
        # ----------------------------------------
        print("\nCalculating parameters/size of the INT8 quantized model...")
        try:
            # 'convert'는 JIT ScriptModule을 반환합니다.
            # .state_dict()를 통해 실제 저장된 요소(INT8 가중치, FP32 스케일/바이어스 등)를 확인합니다.
            
            total_bytes_int8 = 0
            total_params_int8 = 0
            for name, param in model.state_dict().items():
                # param이 텐서인지 확인
                if isinstance(param, torch.Tensor):
                    num_elements = param.numel()
                    total_params_int8 += num_elements
                    element_size = param.element_size()  # 바이트 크기
                    total_bytes_int8 += num_elements * element_size

            print(f"INT8로 변경 후 total_parmter: {total_params_int8 / 1e6:.2f} M")
            print(f"INT8로 변경 후총 모델 크기 : {total_bytes_int8 / 1e6:.2f} MB")

        except Exception as e_param:
            print(f"INT8 모델 파라미터 측정 중 오류 발생: {e_param}")
        # ----------------------------------------

    except Exception as e:
        print(f"IPEX INT8 양자화 중 오류 발생: {e}")
        print("양자화를 건너뛰고 f32 모델로 추론 시간을 측정합니다.")
        # 오류 발생 시, f32 모델 (channels_last 적용됨)로 측정을 계속합니다.
        model = model

    # ----------------------------------------
    # 3. CPU (IPEX INT8) 추론 시간 측정
    # ----------------------------------------
    
    warmup_iterations_cpu = 10
    timing_iterations_cpu = 50
    times_cpu = []

    print(f"\nCPU (IPEX INT8) 추론 시간 측정을 시작합니다 (Warmup: {warmup_iterations_cpu}, Timing: {timing_iterations_cpu})...")

    with torch.no_grad():
        # CPU Warmup
        for _ in range(warmup_iterations_cpu):
            _ = model(dummy_input)

        # CPU Timing
        for _ in range(timing_iterations_cpu):
            start_time_cpu = time.perf_counter()
            _ = model(dummy_input)
            end_time_cpu = time.perf_counter()
            times_cpu.append((end_time_cpu - start_time_cpu) * 1000) # ms 단위

    avg_time_ms_cpu = np.mean(times_cpu)
    std_time_ms_cpu = np.std(times_cpu)
    fps_cpu = 1000.0 / avg_time_ms_cpu

    print('-'*20)
    print(f"CPU (IPEX INT8) 추론 시간 (평균 {timing_iterations_cpu}회): {avg_time_ms_cpu:.2f} ms (± {std_time_ms_cpu:.2f})")
    print(f"CPU (IPEX INT8) FPS (초당 프레임 수): {fps_cpu:.2f}")
    print('-'*20)


if __name__ == '__main__':
    main()