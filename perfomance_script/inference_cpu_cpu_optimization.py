import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from thop import profile
import archs

# IPEX 임포트
import intel_extension_for_pytorch as ipex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리')
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
    # cudnn.benchmark = True # CPU 모드에서는 필요 없음

    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    if not os.path.exists(config_path):
        print(f"오류: {config_path}에서 config 파일을 찾을 수 없습니다.")
        return

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    print(f"실험 {args.name}의 Config 로드 중 (CPU - IPEX JIT f32 모드)")
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    # 모델 생성 (CPU, f32)
    model = archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'], 
        config['deep_supervision'], 
        embed_dims=config.get('input_list', [128,160,256])
    ) 
    model.eval() # JIT 추적 전에 eval 모드로 설정

    # 모델 로드 (best.pth 사용, CPU로)
    model_path = os.path.join(args.output_dir, args.name, 'best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.output_dir, args.name, 'last.pth')
    
    load_model_weights_cpu(model_path, model, strict=True) # CPU 로드 함수 사용

    # ----------------------------------------
    # 1. 파라미터 수 측정
    # ----------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터 수 (Total Params): {total_params / 1e6:.2f} M")
    print(f"학습 가능 파라미터 수 (Trainable Params): {trainable_params / 1e6:.2f} M")

    # ----------------------------------------
    # 2. GFLOPs 측정 (CPU, 최적화 전)
    # ----------------------------------------
    h = 512
    w = 512
    c = config['input_channels']
    
    dummy_input = torch.randn(1, c, h, w).cpu() # f32 더미 입력
    
    try:
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False) 
        gflops = flops / 1e9
        print(f"GFLOPs (CPU, f32, 최적화 전, 입력 크기 {c}x{h}x{w}): {gflops:.2f} G")
    except Exception as e:
        print(f"GFLOPs 측정 중 오류 발생: {e}")
        print("thop 라이브러리가 설치되어 있는지 확인하세요 (pip install thop)")
        
    # ----------------------------------------
    # IPEX JIT 최적화 적용
    # ----------------------------------------
    print("\nApplying Intel Extension for PyTorch (IPEX) JIT optimization (f32)...")
    model.eval() # eval 모드 확인
    
    # JIT 추적을 위해 channels_last로 변경
    model = model.to(memory_format=torch.channels_last) 
    dummy_input = dummy_input.to(memory_format=torch.channels_last) 

    # ipex.optimize 대신 ipex.trace 사용
    try:
        with torch.no_grad():
             # CPU가 BF16 prepack을 지원하지 않으므로 weights_prepack=False 추가
             traced_model = ipex.trace(model, (dummy_input,), dtype=torch.float32, weights_prepack=False)
        model = traced_model # 최적화된 모델로 교체
        print("IPEX JIT optimization applied (f32, weights_prepack=False).")
    except Exception as e:
        print(f"IPEX JIT 추적 중 오류 발생: {e}")
        print("JIT 추적에 실패했습니다. ipex.optimize() 버전으로 테스트를 계속합니다.")
        # JIT 실패 시 ipex.optimize로 fallback (여기도 동일하게 옵션 추가)
        model = ipex.optimize(model, dtype=torch.float32, weights_prepack=False)
        print("IPEX optimization applied (f32, weights_prepack=False).")


    # ----------------------------------------
    # 3. CPU (IPEX JIT) 추론 시간 측정
    # ----------------------------------------
    
    warmup_iterations_cpu = 3
    timing_iterations_cpu = 3
    times_cpu = []

    print(f"\nCPU (IPEX JIT) 추론 시간 측정을 시작합니다 (Warmup: {warmup_iterations_cpu}, Timing: {timing_iterations_cpu})...")

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
    print(f"CPU (IPEX JIT, f32) 추론 시간 (평균 {timing_iterations_cpu}회): {avg_time_ms_cpu:.2f} ms (± {std_time_ms_cpu:.2f})")
    print(f"CPU (IPEX JIT, f32) FPS (초당 프레임 수): {fps_cpu:.2f}")
    print('-'*20)


if __name__ == '__main__':
    main()