"""
downsample_data.py - 데이터 다운샘플링

너무 큰 신호 데이터를 다운샘플링하여 메모리 사용량을 줄입니다.

사용법:
    python downsample_data.py --data_path ./data --target_size 8192
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import signal
from dataset import load_kaggle_gearbox_data


def downsample_signal(signal_data, target_size):
    """
    신호 다운샘플링

    scipy.signal.decimate는 factor>13에서 필터 품질이 떨어진다고 문서에서 권고하므로
    - factor가 크면 resample_poly를 사용해 up/down 분수비로 직접 target_size 맞춤
    - factor가 작으면 decimate(FIR, anti-aliasing) 사용

    Args:
        signal_data (np.ndarray): 원본 신호
        target_size (int): 목표 크기

    Returns:
        np.ndarray: 다운샘플링된 신호
    """
    n = len(signal_data)
    if n <= target_size:
        return signal_data

    factor = n // target_size

    if factor > 13:
        # resample_poly는 up/down 비율로 anti-aliasing 필터 후 리샘플
        from math import gcd
        g = gcd(n, target_size)
        up, down = target_size // g, n // g
        downsampled = signal.resample_poly(signal_data, up, down)
    else:
        downsampled = signal.decimate(signal_data, factor, ftype='fir')

    # 정확한 크기로 보정 (trim or pad)
    if len(downsampled) > target_size:
        downsampled = downsampled[:target_size]
    elif len(downsampled) < target_size:
        downsampled = np.pad(
            downsampled, (0, target_size - len(downsampled)), mode='constant'
        )

    return downsampled


def parse_args():
    parser = argparse.ArgumentParser(description='데이터 다운샘플링')
    parser.add_argument('--data_path', type=str, required=True,
                       help='원본 데이터 경로')
    parser.add_argument('--target_size', type=int, default=8192,
                       help='목표 신호 길이')
    parser.add_argument('--output_path', type=str, default='./data_downsampled',
                       help='다운샘플링된 데이터 저장 경로')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("데이터 다운샘플링")
    print("=" * 70)
    
    # 데이터 로드
    print(f"\n원본 데이터 로딩: {args.data_path}")
    X, y, label_names = load_kaggle_gearbox_data(args.data_path)
    
    print(f"원본 크기: {X.shape}")
    print(f"목표 크기: ({len(X)}, {args.target_size})")
    
    # 다운샘플링
    print("\n다운샘플링 중...")
    X_downsampled = np.zeros((len(X), args.target_size))
    
    for i in range(len(X)):
        X_downsampled[i] = downsample_signal(X[i], args.target_size)
        if (i + 1) % 5 == 0:
            print(f"  진행: {i+1}/{len(X)}")
    
    print("\n다운샘플링 완료!")
    print(f"새 크기: {X_downsampled.shape}")
    
    # 저장
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'X_downsampled.npy', X_downsampled)
    np.save(output_path / 'y.npy', y)
    np.save(output_path / 'label_names.npy', label_names)
    
    print(f"\n저장 완료: {output_path}")
    print(f"  X_downsampled.npy: {X_downsampled.shape}")
    print(f"  y.npy: {y.shape}")
    print(f"  label_names.npy: {len(label_names)} classes")
    
    # 메모리 절약량 계산
    original_size = X.nbytes / (1024**3)
    new_size = X_downsampled.nbytes / (1024**3)
    reduction = (1 - new_size/original_size) * 100
    
    print(f"\n메모리 사용량:")
    print(f"  원본: {original_size:.2f} GB")
    print(f"  다운샘플링: {new_size:.2f} GB")
    print(f"  감소율: {reduction:.1f}%")


if __name__ == '__main__':
    main()
