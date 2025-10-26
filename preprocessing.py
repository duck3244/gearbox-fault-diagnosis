"""
preprocessing.py - 신호 전처리 및 특징 추출

진동 신호 데이터의 전처리와 특징 추출 기능을 제공합니다.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew


class SignalPreprocessor:
    """신호 전처리 클래스"""
    
    def __init__(self, sampling_rate=20000):
        """
        Args:
            sampling_rate (int): 샘플링 주파수 (Hz)
        """
        self.sampling_rate = sampling_rate
    
    def remove_dc_offset(self, signal_data):
        """
        DC 오프셋 제거
        
        Args:
            signal_data (np.ndarray): 입력 신호
        
        Returns:
            np.ndarray: DC 오프셋 제거된 신호
        """
        return signal_data - np.mean(signal_data)
    
    def bandpass_filter(self, signal_data, lowcut=10, highcut=5000, order=5):
        """
        밴드패스 필터 적용
        
        Args:
            signal_data (np.ndarray): 입력 신호
            lowcut (float): 하한 주파수
            highcut (float): 상한 주파수
            order (int): 필터 차수
        
        Returns:
            np.ndarray: 필터링된 신호
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def normalize(self, signal_data, method='minmax'):
        """
        신호 정규화
        
        Args:
            signal_data (np.ndarray): 입력 신호
            method (str): 'minmax', 'zscore', 'maxabs'
        
        Returns:
            np.ndarray: 정규화된 신호
        """
        if method == 'minmax':
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            return (signal_data - min_val) / (max_val - min_val + 1e-8)
        
        elif method == 'zscore':
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            return (signal_data - mean) / (std + 1e-8)
        
        elif method == 'maxabs':
            max_abs = np.max(np.abs(signal_data))
            return signal_data / (max_abs + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def denoise(self, signal_data, wavelet='db4', level=3):
        """
        웨이블릿 기반 노이즈 제거
        
        Args:
            signal_data (np.ndarray): 입력 신호
            wavelet (str): 웨이블릿 타입
            level (int): 분해 레벨
        
        Returns:
            np.ndarray: 노이즈 제거된 신호
        """
        try:
            import pywt
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            
            # 임계값 설정
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
            
            # 소프트 임계값 적용
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            
            # 재구성
            denoised = pywt.waverec(coeffs, wavelet)
            
            return denoised[:len(signal_data)]
        
        except ImportError:
            print("Warning: pywt not installed. Skipping denoising.")
            return signal_data
    
    def segment_signal(self, signal_data, window_size=2048, overlap=0.5):
        """
        신호를 윈도우로 분할
        
        Args:
            signal_data (np.ndarray): 입력 신호
            window_size (int): 윈도우 크기
            overlap (float): 오버랩 비율 (0~1)
        
        Returns:
            np.ndarray: 분할된 신호 (n_segments, window_size)
        """
        step_size = int(window_size * (1 - overlap))
        segments = []
        
        for start_idx in range(0, len(signal_data) - window_size + 1, step_size):
            segment = signal_data[start_idx:start_idx + window_size]
            segments.append(segment)
        
        return np.array(segments)
    
    def preprocess(self, signal_data, remove_dc=True, filter_signal=True, 
                   normalize_signal=True, denoise_signal=False):
        """
        완전한 전처리 파이프라인
        
        Args:
            signal_data (np.ndarray): 입력 신호
            remove_dc (bool): DC 오프셋 제거 여부
            filter_signal (bool): 필터링 여부
            normalize_signal (bool): 정규화 여부
            denoise_signal (bool): 노이즈 제거 여부
        
        Returns:
            np.ndarray: 전처리된 신호
        """
        processed = signal_data.copy()
        
        if remove_dc:
            processed = self.remove_dc_offset(processed)
        
        if filter_signal:
            processed = self.bandpass_filter(processed)
        
        if denoise_signal:
            processed = self.denoise(processed)
        
        if normalize_signal:
            processed = self.normalize(processed, method='zscore')
        
        return processed


class FeatureExtractor:
    """특징 추출 클래스"""
    
    def __init__(self, sampling_rate=20000):
        """
        Args:
            sampling_rate (int): 샘플링 주파수
        """
        self.sampling_rate = sampling_rate
    
    def time_domain_features(self, signal_data):
        """
        시간 영역 특징 추출
        
        Args:
            signal_data (np.ndarray): 입력 신호
        
        Returns:
            dict: 시간 영역 특징
        """
        features = {}
        
        # 기본 통계
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['max'] = np.max(signal_data)
        features['min'] = np.min(signal_data)
        features['ptp'] = np.ptp(signal_data)  # Peak-to-peak
        
        # RMS (Root Mean Square)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        
        # Shape factor
        features['shape_factor'] = features['rms'] / (np.mean(np.abs(signal_data)) + 1e-8)
        
        # Crest factor
        features['crest_factor'] = features['max'] / (features['rms'] + 1e-8)
        
        # Impulse factor
        features['impulse_factor'] = features['max'] / (np.mean(np.abs(signal_data)) + 1e-8)
        
        # Clearance factor
        mean_sqrt = np.mean(np.sqrt(np.abs(signal_data)))**2
        features['clearance_factor'] = features['max'] / (mean_sqrt + 1e-8)
        
        # Kurtosis & Skewness
        features['kurtosis'] = kurtosis(signal_data)
        features['skewness'] = skew(signal_data)
        
        return features
    
    def frequency_domain_features(self, signal_data):
        """
        주파수 영역 특징 추출
        
        Args:
            signal_data (np.ndarray): 입력 신호
        
        Returns:
            dict: 주파수 영역 특징
        """
        features = {}
        
        # FFT
        n = len(signal_data)
        fft_vals = fft(signal_data)
        fft_freq = fftfreq(n, 1/self.sampling_rate)
        
        # 양의 주파수만 사용
        positive_freq_idx = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        # 기본 통계
        features['fft_mean'] = np.mean(fft_magnitude)
        features['fft_std'] = np.std(fft_magnitude)
        features['fft_max'] = np.max(fft_magnitude)
        
        # Spectral centroid (무게 중심 주파수)
        features['spectral_centroid'] = np.sum(fft_freq_positive * fft_magnitude) / (np.sum(fft_magnitude) + 1e-8)
        
        # Spectral spread (주파수 분산)
        features['spectral_spread'] = np.sqrt(
            np.sum(((fft_freq_positive - features['spectral_centroid'])**2) * fft_magnitude) / 
            (np.sum(fft_magnitude) + 1e-8)
        )
        
        # Spectral entropy
        psd = fft_magnitude**2
        psd_normalized = psd / (np.sum(psd) + 1e-8)
        features['spectral_entropy'] = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-8))
        
        # Peak frequency
        peak_idx = np.argmax(fft_magnitude)
        features['peak_frequency'] = fft_freq_positive[peak_idx]
        
        return features
    
    def statistical_features(self, signal_data):
        """
        고급 통계 특징
        
        Args:
            signal_data (np.ndarray): 입력 신호
        
        Returns:
            dict: 통계 특징
        """
        features = {}
        
        # Quantiles
        features['q25'] = np.percentile(signal_data, 25)
        features['q50'] = np.percentile(signal_data, 50)  # Median
        features['q75'] = np.percentile(signal_data, 75)
        features['iqr'] = features['q75'] - features['q25']  # Interquartile range
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # Mean crossing rate
        mean_val = np.mean(signal_data)
        mean_crossings = np.where(np.diff(np.sign(signal_data - mean_val)))[0]
        features['mean_crossing_rate'] = len(mean_crossings) / len(signal_data)
        
        return features
    
    def extract_all_features(self, signal_data):
        """
        모든 특징 추출
        
        Args:
            signal_data (np.ndarray): 입력 신호
        
        Returns:
            dict: 모든 특징
        """
        all_features = {}
        
        # 시간 영역
        time_features = self.time_domain_features(signal_data)
        all_features.update(time_features)
        
        # 주파수 영역
        freq_features = self.frequency_domain_features(signal_data)
        all_features.update(freq_features)
        
        # 통계
        stat_features = self.statistical_features(signal_data)
        all_features.update(stat_features)
        
        return all_features
    
    def extract_features_dataframe(self, signal_list, labels=None):
        """
        여러 신호에서 특징 추출하여 DataFrame 생성
        
        Args:
            signal_list (list): 신호 리스트
            labels (list): 레이블 리스트 (선택사항)
        
        Returns:
            pd.DataFrame: 특징 DataFrame
        """
        features_list = []
        
        for signal_data in signal_list:
            features = self.extract_all_features(signal_data)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        if labels is not None:
            df['label'] = labels
        
        return df


def augment_signal(signal_data, augmentation_type='noise', **kwargs):
    """
    신호 데이터 증강
    
    Args:
        signal_data (np.ndarray): 입력 신호
        augmentation_type (str): 증강 타입
            - 'noise': 가우시안 노이즈 추가
            - 'time_shift': 시간 이동
            - 'amplitude_scale': 진폭 스케일링
            - 'time_warp': 시간 왜곡
        **kwargs: 증강 파라미터
    
    Returns:
        np.ndarray: 증강된 신호
    """
    if augmentation_type == 'noise':
        noise_level = kwargs.get('noise_level', 0.01)
        noise = np.random.normal(0, noise_level, signal_data.shape)
        return signal_data + noise
    
    elif augmentation_type == 'time_shift':
        shift_range = kwargs.get('shift_range', 100)
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(signal_data, shift)
    
    elif augmentation_type == 'amplitude_scale':
        scale_range = kwargs.get('scale_range', (0.8, 1.2))
        scale = np.random.uniform(*scale_range)
        return signal_data * scale
    
    elif augmentation_type == 'time_warp':
        warp_factor = kwargs.get('warp_factor', 0.1)
        n = len(signal_data)
        indices = np.arange(n)
        warped_indices = indices + np.random.uniform(-warp_factor, warp_factor, n) * n
        warped_indices = np.clip(warped_indices, 0, n-1).astype(int)
        return signal_data[warped_indices]
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


if __name__ == '__main__':
    # 테스트
    print("전처리 모듈 테스트\n")
    
    # 더미 신호 생성
    t = np.linspace(0, 1, 1000)
    signal_data = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(1000)
    
    # 전처리
    preprocessor = SignalPreprocessor(sampling_rate=1000)
    processed_signal = preprocessor.preprocess(signal_data)
    
    print(f"원본 신호 길이: {len(signal_data)}")
    print(f"전처리된 신호 길이: {len(processed_signal)}")
    
    # 특징 추출
    extractor = FeatureExtractor(sampling_rate=1000)
    features = extractor.extract_all_features(processed_signal)
    
    print(f"\n추출된 특징 수: {len(features)}")
    print("특징 예시:")
    for key, value in list(features.items())[:5]:
        print(f"  {key}: {value:.4f}")
