import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from thop import profile
import archs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='실험(모델) 이름')
    parser.add_argument('--output_dir', default='outputs', help='출력 디렉토리')
    args = parser.parse_args()
    return args

def load_model_weights(path, model, strict=True):
    """
    DDP 여부에 관계없이 모델 가중치를 불러옵니다. (GPU로 로드)
    """
    ckpt = torch.load(path, map_location='cuda') # GPU로 로드
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    
    if any(k.startswith('module.') for k in state.keys()):
        print("DDP로 학습된 모델을 감지했습니다. 'module.' 접두사를 제거합니다.")
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    
    try:
        model.load_state_dict(state, strict=strict)
        print(f"모델 가중치를 {path}에서 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생: {e}")
        if not strict:
            print("strict=False로 다시 시도합니다.")
            model.load_state_dict(state, strict=False)
            print(f"모델 가중치를 strict=False로 {path}에서 불러왔습니다.")
        else:
            raise e

def main():
    args = parse_args()
    cudnn.benchmark = True # GPU 벤치마크 활성화

    config_path = os.path.join(args.output_dir, args.name, 'config.yml')
    if not os.path.exists(config_path):
        print(f"오류: {config_path}에서 config 파일을 찾을 수 없습니다.")
        return

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    print(f"실험 {args.name}의 Config 로드 중")
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    # 모델 생성
    model = archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'], 
        config['deep_supervision'], 
        embed_dims=config.get('input_list', [128,160,256])
    )
    model = model.cuda() # 모델을 GPU로 이동

    # 모델 로드 (best.pth 사용)
    model_path = os.path.join(args.output_dir, args.name, 'best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.output_dir, args.name, 'last.pth')
    
    load_model_weights(model_path, model, strict=True) # 가중치를 GPU로 로드
    model.eval()

    # ----------------------------------------
    # 1. 파라미터 수 측정
    # ----------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터 수 (Total Params): {total_params / 1e6:.2f} M")
    print(f"학습 가능 파라미터 수 (Trainable Params): {trainable_params / 1e6:.2f} M")

    # ----------------------------------------
    # 2. GFLOPs 측정 (GPU)
    # ----------------------------------------
    h = config['input_h']
    w = config['input_w']
    c = config['input_channels']
    
    dummy_input = torch.randn(1, c, h, w).cuda() # GPU 더미 입력
    
    try:
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        gflops = flops / 1e9
        print(f"GFLOPs (GPU, 입력 크기 {c}x{h}x{w}): {gflops:.2f} G")
    except Exception as e:
        print(f"GFLOPs 측정 중 오류 발생: {e}")
        print("thop 라이브러리가 설치되어 있는지 확인하세요 (pip install thop)")

    # ----------------------------------------
    # 3. GPU 추론 시간 측정
    # ----------------------------------------
    warmup_iterations = 20
    timing_iterations = 100
    times = []

    print(f"\nGPU 추론 시간 측정을 시작합니다 (Warmup: {warmup_iterations}, Timing: {timing_iterations})...")

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
            torch.cuda.synchronize() # GPU 작업 완료 대기

        # Timing
        for _ in range(timing_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = model(dummy_input)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000) # 밀리초(ms) 단위로 저장

    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    fps = 1000.0 / avg_time_ms

    print('-'*20)
    print(f"GPU 추론 시간 (평균 {timing_iterations}회): {avg_time_ms:.2f} ms (± {std_time_ms:.2f})")
    print(f"GPU FPS (초당 프레임 수): {fps:.2f}")
    print('-'*20)


if __name__ == '__main__':
    main()