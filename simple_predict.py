"""
simple_predict.py - 가장 간단한 예측 예제

학습된 모델로 빠르게 예측하는 방법

사용법:
    python simple_predict.py
"""

import numpy as np
from pathlib import Path
from inference import GearboxPredictor


def find_latest_model():
    """가장 최근 모델 찾기"""
    results_dir = Path('results')
    
    if not results_dir.exists():
        raise FileNotFoundError("results 디렉토리를 찾을 수 없습니다.")
    
    # 모든 타임스탬프 디렉토리 찾기
    model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], 
                       reverse=True)
    
    if not model_dirs:
        raise FileNotFoundError("학습된 모델을 찾을 수 없습니다.")
    
    # 가장 최근 모델
    latest_model = model_dirs[0] / 'best_model.pth'
    
    if not latest_model.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {latest_model}")
    
    return latest_model


def load_test_data():
    """테스트 데이터 로드"""
    
    # 옵션 1: 다운샘플링된 데이터
    downsampled_path = Path('./data_downsampled')
    if (downsampled_path / 'X_downsampled.npy').exists():
        print("다운샘플링된 데이터 로드")
        X = np.load(downsampled_path / 'X_downsampled.npy')
        y = np.load(downsampled_path / 'y.npy')
        label_names = np.load(downsampled_path / 'label_names.npy', allow_pickle=True)
        return X, y, label_names
    
    # 옵션 2: 원본 데이터
    from dataset import load_kaggle_gearbox_data
    print("원본 데이터 로드 (시간이 걸릴 수 있습니다...)")
    X, y, label_names = load_kaggle_gearbox_data('./data')
    return X, y, label_names


def main():
    print("=" * 70)
    print("🎯 간단한 예측 예제")
    print("=" * 70)
    
    # Step 1: 모델 찾기
    print("\n[Step 1] 모델 찾기...")
    try:
        model_path = find_latest_model()
        print(f"✅ 모델 발견: {model_path}")
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("\n먼저 모델을 학습하세요:")
        print("  python main.py train --data_path ./data --model CNN")
        return
    
    # Step 2: 데이터 로드
    print("\n[Step 2] 테스트 데이터 로드...")
    try:
        X, y, label_names = load_test_data()
        print(f"✅ 데이터 로드 완료: {X.shape}")
        print(f"   클래스: {label_names}")
    except Exception as e:
        print(f"❌ 오류: {e}")
        return
    
    # Step 3: 예측기 초기화
    print("\n[Step 3] 예측기 초기화...")
    predictor = GearboxPredictor(str(model_path))
    print("✅ 예측기 준비 완료")
    
    # Step 4: 랜덤 샘플 선택
    print("\n[Step 4] 랜덤 샘플 5개 선택...")
    n_samples = min(5, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    # Step 5: 예측 실행
    print("\n[Step 5] 예측 실행...\n")
    print("=" * 70)
    
    correct = 0
    for i, idx in enumerate(indices):
        signal = X[idx]
        true_label = label_names[y[idx]]
        
        # 예측
        result = predictor.predict_single(signal)
        
        # 결과 출력
        is_correct = result['predicted_class'] == true_label
        if is_correct:
            correct += 1
            status = "✅ 정답"
        else:
            status = "❌ 오답"
        
        print(f"\n샘플 {i+1} (인덱스 {idx}):")
        print(f"  예측: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
        print(f"  실제: {true_label}")
        print(f"  {status}")
        
        # 확률 바 차트
        print("  확률:")
        for class_name, prob in result['probabilities'].items():
            bar_len = int(prob * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(f"    {class_name:12s} {bar} {prob*100:5.1f}%")
    
    # 정확도
    accuracy = (correct / n_samples) * 100
    print("\n" + "=" * 70)
    print(f"테스트 정확도: {correct}/{n_samples} = {accuracy:.1f}%")
    print("=" * 70)
    
    # 추가 정보
    print("\n💡 팁:")
    print("  - 더 많은 샘플 테스트: test_predict.py 사용")
    print("  - CSV 파일 예측: inference.py --input_data file.csv")
    print("  - 배치 예측: NumPy 배열로 여러 샘플 동시 예측")
    
    print("\n✅ 예측 완료!\n")


if __name__ == '__main__':
    main()
