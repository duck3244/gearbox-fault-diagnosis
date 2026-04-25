"""
visualize.py - 데이터 및 결과 시각화

신호 데이터와 모델 결과를 다양한 방식으로 시각화합니다.

사용법:
    python visualize.py --data_path ./data --output_dir visualizations
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from pathlib import Path

from dataset import load_kaggle_gearbox_data


class SignalVisualizer:
    """신호 시각화 클래스"""
    
    def __init__(self, sampling_rate=20000):
        """
        Args:
            sampling_rate (int): 샘플링 주파수
        """
        self.sampling_rate = sampling_rate
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_time_series(self, signal_data, title='Time Series', 
                        save_path=None, show_stats=True):
        """
        시간 영역 신호 시각화
        
        Args:
            signal_data (np.ndarray): 신호 데이터
            title (str): 제목
            save_path (str): 저장 경로
            show_stats (bool): 통계 표시 여부
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        
        time = np.arange(len(signal_data)) / self.sampling_rate
        ax.plot(time, signal_data, linewidth=0.5, color='steelblue')
        
        if show_stats:
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            ax.axhline(mean, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean:.4f}')
            ax.axhline(mean + std, color='orange', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'±1 STD: {std:.4f}')
            ax.axhline(mean - std, color='orange', linestyle='--', 
                      linewidth=1.5, alpha=0.7)
            ax.legend(loc='upper right')
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_frequency_spectrum(self, signal_data, title='Frequency Spectrum',
                               save_path=None):
        """
        주파수 스펙트럼 시각화
        
        Args:
            signal_data (np.ndarray): 신호 데이터
            title (str): 제목
            save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # FFT
        n = len(signal_data)
        fft_vals = fft(signal_data)
        fft_freq = fftfreq(n, 1/self.sampling_rate)
        
        # 양의 주파수만
        positive_idx = fft_freq > 0
        freq_positive = fft_freq[positive_idx]
        magnitude = np.abs(fft_vals[positive_idx])
        
        # 선형 스케일
        axes[0].plot(freq_positive, magnitude, linewidth=1, color='steelblue')
        axes[0].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{title} (Linear Scale)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 로그 스케일
        axes[1].semilogy(freq_positive, magnitude, linewidth=1, color='coral')
        axes[1].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Magnitude (log)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{title} (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spectrogram(self, signal_data, title='Spectrogram',
                        save_path=None):
        """
        스펙트로그램 시각화
        
        Args:
            signal_data (np.ndarray): 신호 데이터
            title (str): 제목
            save_path (str): 저장 경로
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        f, t, Sxx = scipy_signal.spectrogram(signal_data, self.sampling_rate)
        
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                          shading='gouraud', cmap='viridis')
        
        ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_comparison(self, signals_dict, title='Signal Comparison',
                       save_path=None):
        """
        여러 신호 비교
        
        Args:
            signals_dict (dict): {label: signal_data} 형태의 딕셔너리
            title (str): 제목
            save_path (str): 저장 경로
        """
        n_signals = len(signals_dict)
        fig, axes = plt.subplots(n_signals, 1, figsize=(14, 4*n_signals))
        
        if n_signals == 1:
            axes = [axes]
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_signals))
        
        for idx, (label, signal_data) in enumerate(signals_dict.items()):
            time = np.arange(len(signal_data)) / self.sampling_rate
            axes[idx].plot(time, signal_data, linewidth=0.8, 
                          color=colors[idx], label=label)
            axes[idx].set_ylabel('Amplitude', fontsize=11)
            axes[idx].set_title(label, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(loc='upper right')
        
        axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.001)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_distribution(self, signal_data, title='Amplitude Distribution',
                         save_path=None):
        """
        진폭 분포 시각화
        
        Args:
            signal_data (np.ndarray): 신호 데이터
            title (str): 제목
            save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 히스토그램
        axes[0].hist(signal_data, bins=50, color='steelblue', 
                    alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(signal_data), color='red', 
                       linestyle='--', linewidth=2, label='Mean')
        axes[0].set_xlabel('Amplitude', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Histogram', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        axes[1].boxplot(signal_data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Amplitude', fontsize=12, fontweight='bold')
        axes[1].set_title('Box Plot', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class DatasetVisualizer:
    """데이터셋 시각화 클래스"""
    
    def __init__(self, sampling_rate=20000):
        self.sampling_rate = sampling_rate
        self.signal_viz = SignalVisualizer(sampling_rate)
    
    def visualize_class_samples(self, X, y, label_names, n_samples=3,
                               save_dir=None):
        """
        각 클래스별 샘플 시각화
        
        Args:
            X (np.ndarray): 특징 데이터
            y (np.ndarray): 레이블
            label_names (list): 레이블 이름
            n_samples (int): 클래스당 샘플 수
            save_dir (Path): 저장 디렉토리
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for class_idx, class_name in enumerate(label_names):
            # 해당 클래스의 샘플 인덱스
            class_indices = np.where(y == class_idx)[0]
            
            if len(class_indices) == 0:
                continue
            
            # 랜덤 샘플 선택
            sample_indices = np.random.choice(class_indices, 
                                             min(n_samples, len(class_indices)),
                                             replace=False)
            
            # 각 샘플 시각화
            for i, idx in enumerate(sample_indices):
                signal_data = X[idx]
                
                # 시간 영역
                save_path = save_dir / f'{class_name}_sample{i+1}_time.png' if save_dir else None
                self.signal_viz.plot_time_series(
                    signal_data, 
                    title=f'{class_name} - Sample {i+1}',
                    save_path=save_path
                )
                
                # 주파수 영역
                save_path = save_dir / f'{class_name}_sample{i+1}_freq.png' if save_dir else None
                self.signal_viz.plot_frequency_spectrum(
                    signal_data,
                    title=f'{class_name} - Sample {i+1}',
                    save_path=save_path
                )
        
        print(f"클래스별 샘플 시각화 완료: {save_dir}")
    
    def plot_class_statistics(self, X, y, label_names, save_path=None):
        """
        클래스별 통계 시각화
        
        Args:
            X (np.ndarray): 특징 데이터
            y (np.ndarray): 레이블
            label_names (list): 레이블 이름
            save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 각 클래스별 통계
        class_means = []
        class_stds = []
        class_maxs = []
        class_mins = []
        
        for class_idx in range(len(label_names)):
            class_data = X[y == class_idx]
            class_means.append(np.mean(class_data))
            class_stds.append(np.std(class_data))
            class_maxs.append(np.max(class_data))
            class_mins.append(np.min(class_data))
        
        # Mean
        axes[0].bar(label_names, class_means, color='steelblue', alpha=0.7)
        axes[0].set_title('Mean Amplitude by Class', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Mean', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Standard Deviation
        axes[1].bar(label_names, class_stds, color='coral', alpha=0.7)
        axes[1].set_title('Standard Deviation by Class', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('STD', fontsize=11)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Max
        axes[2].bar(label_names, class_maxs, color='lightgreen', alpha=0.7)
        axes[2].set_title('Maximum Amplitude by Class', fontsize=13, fontweight='bold')
        axes[2].set_ylabel('Max', fontsize=11)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Sample count
        class_counts = [np.sum(y == i) for i in range(len(label_names))]
        axes[3].bar(label_names, class_counts, color='mediumpurple', alpha=0.7)
        axes[3].set_title('Sample Count by Class', fontsize=13, fontweight='bold')
        axes[3].set_ylabel('Count', fontsize=11)
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"클래스 통계 저장: {save_path}")
        else:
            plt.show()


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='Data Visualization')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='시각화 저장 디렉토리')
    parser.add_argument('--n_samples', type=int, default=3,
                       help='클래스당 시각화할 샘플 수')
    parser.add_argument('--sampling_rate', type=int, default=20000,
                       help='샘플링 주파수')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("데이터 시각화")
    print("=" * 70)
    
    # 데이터 로드
    print("\n데이터 로딩 중...")
    X, y, label_names = load_kaggle_gearbox_data(args.data_path)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 시각화
    print("\n시각화 생성 중...")
    visualizer = DatasetVisualizer(args.sampling_rate)
    
    # 클래스별 샘플 시각화
    visualizer.visualize_class_samples(X, y, label_names, 
                                       n_samples=args.n_samples,
                                       save_dir=output_dir / 'class_samples')
    
    # 클래스 통계
    visualizer.plot_class_statistics(X, y, label_names,
                                    save_path=output_dir / 'class_statistics.png')
    
    print(f"\n모든 시각화가 '{output_dir}'에 저장되었습니다.")


if __name__ == '__main__':
    main()
