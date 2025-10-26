"""
config.py - 프로젝트 설정 파일

모든 하이퍼파라미터와 설정을 관리합니다.
"""

from pathlib import Path


class Config:
    """기본 설정 클래스"""
    
    # 프로젝트 경로
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    
    # 데이터 설정
    TEST_SIZE = 0.15
    VAL_SIZE = 0.18
    RANDOM_SEED = 42
    
    # 모델 설정
    MODEL_TYPE = 'CNN'  # 'CNN', 'ResNet', 'MLP'
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Early Stopping
    PATIENCE = 15
    MIN_DELTA = 0.0
    
    # Learning Rate Scheduler
    LR_FACTOR = 0.5
    LR_PATIENCE = 7
    LR_MIN = 1e-6
    
    # DataLoader
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # 디바이스
    USE_CUDA = True
    
    @classmethod
    def display(cls):
        """설정 출력"""
        print("=" * 70)
        print("현재 설정")
        print("=" * 70)
        
        print("\n[경로 설정]")
        print(f"  프로젝트 루트: {cls.PROJECT_ROOT}")
        print(f"  데이터 디렉토리: {cls.DATA_DIR}")
        print(f"  결과 디렉토리: {cls.RESULTS_DIR}")
        
        print("\n[데이터 설정]")
        print(f"  테스트 비율: {cls.TEST_SIZE}")
        print(f"  검증 비율: {cls.VAL_SIZE}")
        print(f"  랜덤 시드: {cls.RANDOM_SEED}")
        
        print("\n[모델 설정]")
        print(f"  모델 타입: {cls.MODEL_TYPE}")
        
        print("\n[학습 설정]")
        print(f"  배치 크기: {cls.BATCH_SIZE}")
        print(f"  에폭: {cls.EPOCHS}")
        print(f"  학습률: {cls.LEARNING_RATE}")
        print(f"  Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"  Early Stopping Patience: {cls.PATIENCE}")
        
        print("\n[기타]")
        print(f"  워커 수: {cls.NUM_WORKERS}")
        print(f"  CUDA 사용: {cls.USE_CUDA}")
        print("=" * 70)


class CNNConfig(Config):
    """CNN 모델 전용 설정"""
    MODEL_TYPE = 'CNN'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100


class ResNetConfig(Config):
    """ResNet 모델 전용 설정"""
    MODEL_TYPE = 'ResNet'
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    EPOCHS = 150
    PATIENCE = 20


class MLPConfig(Config):
    """MLP 모델 전용 설정"""
    MODEL_TYPE = 'MLP'
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 80


def get_config(model_type='CNN'):
    """
    모델 타입에 따른 설정 반환
    
    Args:
        model_type (str): 모델 타입
    
    Returns:
        Config: 설정 클래스
    """
    configs = {
        'CNN': CNNConfig,
        'ResNet': ResNetConfig,
        'MLP': MLPConfig
    }
    
    if model_type not in configs:
        print(f"경고: 알 수 없는 모델 타입 '{model_type}'. 기본 설정 사용.")
        return Config
    
    return configs[model_type]


if __name__ == '__main__':
    # 설정 테스트
    print("기본 설정:")
    Config.display()
    
    print("\n\nCNN 설정:")
    CNNConfig.display()
    
    print("\n\nResNet 설정:")
    ResNetConfig.display()