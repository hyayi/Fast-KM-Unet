import argparse
import os
import time
import numpy as np
import sys
import psutil
import cv2
import albumentations as A

# OpenVINO Core 임포트
import openvino as ov


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVINO Model Inference Speed and Memory Benchmark")
    parser.add_argument('--model_xml', type=str, required=True, 
                        help='평가할 OpenVINO 모델(.xml) 파일 경로')
    
    # 모델 입력 크기
    parser.add_argument('--input_c', type=int, default=3, help='모델 입력 채널 (예: 1)')
    parser.add_argument('--input_h', type=int, default=1024, help='모델 입력 높이 (예: 1024)')
    parser.add_argument('--input_w', type=int, default=1024, help='모델 입력 너비 (예: 1024)')
    
    # 원본 X-ray 크기 (전처리 벤치마크용)
    parser.add_argument('--orig_h', type=int, default=2560, help='원본 X-ray 높이 (기본값: 2560)')
    parser.add_argument('--orig_w', type=int, default=3000, help='원본 X-ray 너비 (기본값: 3000)')

    parser.add_argument('--num_threads', type=int, default=8, 
                        help='사용할 CPU 스크립트 (기본값: OpenVINO가 모두 사용)')
    args = parser.parse_args()
    return args


def preprocess_image(dummy_input_shape, orig_h, orig_w):
    """
    전처리: 원본 16bit(uint16) 이미지를 8bit 범위(0~255)로 변환 후,
           입력 크기로 리사이즈하고 255.0으로 정규화 수행
    dummy_input_shape: (C, H, W)
    orig_h, orig_w: 원본 이미지 크기
    """

    # 1. 원본 16-bit(uint16) 이미지 생성 (0~65535)
    image = np.random.randint(0, 65536, (orig_h, orig_w), dtype=np.uint16)

    start_time = time.time()
    
    # 2. 16-bit (0-65535) -> 8-bit 범위 (0-255)로 스케일링
    #    (Windowing/Leveling을 시뮬레이션)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 3. Resize (8비트 범위로 변환된 이미지를 리사이즈)
    # cv2.resize는 (너비, 높이) 순서
    resized = cv2.resize(image, (dummy_input_shape[2], dummy_input_shape[1]))

    # 4. Normalize (0~1로 정규화 - 8-bit 최대값 255.0 기준)
    normalized = A.Normalize()(image=resized)['image']

    # 5. HWC -> CHW 전환
    input_data = normalized.transpose(2, 0, 1)
    input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가  

    preproc_time = (time.time() - start_time) * 1000  # ms 단위

    return input_data.astype(np.float32), preproc_time


def main():
    args = parse_args()

    print('-'*20)
    print("OpenVINO 모델 추론 속도 및 메모리 사용량 측정을 시작합니다.")
    print(f"  - Model: {args.model_xml}")
    # 수정됨: 전처리 과정을 명시
    print(f"  - Orig Size (H, W): ({args.orig_h}, {args.orig_w}) - [uint16 -> 8bit range -> normalize]") 
    print(f"  - Input Size (C, H, W): ({args.input_c}, {args.input_h}, {args.input_w})")
    print(f"  - CPU Threads: {'Default' if args.num_threads is None else args.num_threads}")
    print('-'*20)

    # OpenVINO Core 생성 및 스레드 설정
    ie = ov.Core()
    if args.num_threads is not None and args.num_threads > 0:
        try:
            ie.set_property("CPU", {"INFERENCE_NUM_THREADS": args.num_threads})
            print(f"OpenVINO CPU 스레드 수를 {args.num_threads}개로 제한합니다.")
        except Exception:
            # 이전 OpenVINO 버전과의 호환성을 위한 fallback
            ie.set_property({"CPU_THREADS_NUM": str(args.num_threads)})
            print(f"OpenVINO CPU 스레드 수를 {args.num_threads}개로 제한합니다. (Fallback)")
    else:
        print("OpenVINO CPU 스레드 수를 제한하지 않습니다 (기본값 사용).")

    # 모델 로드
    try:
        model = ie.read_model(model=args.model_xml)
    except Exception as e:
        print(f"오류: 모델 파일을 읽을 수 없습니다. {e}")
        return

    # 모델 컴파일
    try:
        compiled_model = ie.compile_model(model=model, device_name="CPU")
    except Exception as e:
        print(f"오류: 모델 컴파일에 실패했습니다. {e}")
        return

    output_layer = compiled_model.output(0)
    print("Model compiled successfully.")

    # 전처리 및 더미 입력 생성
    dummy_input_np, preproc_time = preprocess_image(
        (args.input_c, args.input_h, args.input_w), 
        args.orig_h, 
        args.orig_w
    )
    print(dummy_input_np.shape)

    # 프로세스 메모리 사용 측정용
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2  # MB
    peak_mem = mem_before

    warmup_iterations_cpu = 10
    timing_iterations_cpu = 50
    times_cpu = []

    print(f"\n전처리 시간: {preproc_time:.2f} ms")
    print(f"CPU (OpenVINO) 추론 속도 측정을 시작합니다 (Warmup: {warmup_iterations_cpu}, Timing: {timing_iterations_cpu})...")

    try:
        # Warmup
        for _ in range(warmup_iterations_cpu):
            _ = compiled_model([dummy_input_np])[output_layer]
            current_mem = process.memory_info().rss / 1024**2
            if current_mem > peak_mem:
                peak_mem = current_mem

        # Timing
        for _ in range(timing_iterations_cpu):
            start_time_cpu = time.perf_counter()
            _ = compiled_model([dummy_input_np])[output_layer]
            end_time_cpu = time.perf_counter()
            times_cpu.append((end_time_cpu - start_time_cpu) * 1000)  # ms 단위

            current_mem = process.memory_info().rss / 1024**2
            if current_mem > peak_mem:
                peak_mem = current_mem

    except Exception as e:
        print(f"\n추론 중 오류 발생: {e}")
        print("모델의 입력 크기(--input_c/h/w)가 올바른지 확인하세요.")
        return

    avg_time_ms_cpu = np.mean(times_cpu)
    std_time_ms_cpu = np.std(times_cpu)
    fps_cpu = 1000.0 / avg_time_ms_cpu

    print('-'*20)
    print(f"전처리 시간: {preproc_time:.2f} ms")
    print(f"CPU (OpenVINO) 추론 시간 (평균 {timing_iterations_cpu}회): {avg_time_ms_cpu:.2f} ms (± {std_time_ms_cpu:.2f})")
    print(f"CPU (OpenVINO) FPS (초당 프레임 수): {fps_cpu:.2f}")
    print(f"프로세스 메모리 사용량: {mem_before:.2f} MB (측정 시작 시)")
    print(f"피크 메모리 사용량: {peak_mem:.2f} MB (측정 중 최대)")
    print('-'*20)


if __name__ == '__main__':
    main()