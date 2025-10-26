"""
test_predict.py - 예측 테스트 스크립트

학습된 모델로 쉽게 예측을 테스트할 수 있습니다.

사용법:
    # 테스트 샘플 생성 및 예측
    python test_predict.py --model_path results/*/best_model.pth
    
    # 특정 데이터로 예측
    python test_predict.py --model_path results/*/best_model.pth --data_path ./data
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from inference import GearboxPredictor
from dataset import load_kaggle_gearbox_data


def create_test_samples(data_path, n_samples=5, output_file='test_samples.npy'):
    """
    테스트 샘플 생성
    
    Args:
        data_path: 데이터 경로
        n_samples: 샘플 수
        output_file: 출력 파일명
    """
    print("=" * 70)
    print("테스트 샘플 생성")
    print("=" * 70)
    
    # 다운샘플링된 데이터 로드 시도
    downsampled_path = Path(data_path)
    
    if (downsampled_path / 'X_downsampled.npy').exists():
        print(f"\n다운샘플링된 데이터 로드: {downsampled_path}")
        X = np.load(downsampled_path / 'X_downsampled.npy')
        y = np.load(downsampled_path / 'y.npy')
        label_names = np.load(downsampled_path / 'label_names.npy', allow_pickle=True)
    else:
        print(f"\n원본 데이터 로드: {data_path}")
        X, y, label_names = load_kaggle_gearbox_data(data_path)
    
    print(f"데이터 shape: {X.shape}")
    print(f"클래스: {label_names}")
    
    # 랜덤 샘플 선택
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    test_samples = X[indices]
    test_labels = y[indices]
    
    # 저장
    np.save(output_file, test_samples)
    
    print(f"\n{len(test_samples)}개 테스트 샘플 생성")
    print(f"저장 위치: {output_file}")
    print(f"Shape: {test_samples.shape}")
    
    # 실제 레이블 출력
    print("\n실제 레이블:")
    for i, (idx, label_idx) in enumerate(zip(indices, test_labels)):
        print(f"  샘플 {i+1}: {label_names[label_idx]}")
    
    return output_file, test_labels, label_names


def test_prediction(model_path, test_file, true_labels=None, label_names=None):
    """
    예측 테스트
    
    Args:
        model_path: 모델 경로
        test_file: 테스트 파일
        true_labels: 실제 레이블 (선택사항)
        label_names: 레이블 이름 (선택사항)
    """
    print("\n" + "=" * 70)
    print("예측 실행")
    print("=" * 70)
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 예측기 로드
    predictor = GearboxPredictor(model_path, device=device)
    
    # 예측
    results = predictor.predict_from_npy(test_file)
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("예측 결과")
    print("=" * 70)
    
    correct = 0
    total = len(results) if isinstance(results, list) else 1
    
    if isinstance(results, list):
        for i, result in enumerate(results):
            print(f"\n샘플 {i+1}:")
            print(f"  예측 클래스: {result['predicted_class']}")
            print(f"  신뢰도: {result['confidence']*100:.2f}%")
            
            if true_labels is not None and label_names is not None:
                true_label = label_names[true_labels[i]]
                print(f"  실제 클래스: {true_label}")
                
                if result['predicted_class'] == true_label:
                    print(f"  결과: ✅ 정답")
                    correct += 1
                else:
                    print(f"  결과: ❌ 오답")
            
            print("  클래스별 확률:")
            for class_name, prob in result['probabilities'].items():
                bar_length = int(prob * 50)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"    {class_name:15s} {bar} {prob*100:5.2f}%")
    else:
        result = results
        print(f"\n예측 클래스: {result['predicted_class']}")
        print(f"신뢰도: {result['confidence']*100:.2f}%")
        
        if true_labels is not None and label_names is not None:
            true_label = label_names[true_labels[0]]
            print(f"실제 클래스: {true_label}")
            
            if result['predicted_class'] == true_label:
                print(f"결과: ✅ 정답")
                correct = 1
            else:
                print(f"결과: ❌ 오답")
        
        print("\n클래스별 확률:")
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {class_name:15s} {bar} {prob*100:5.2f}%")
    
    # 정확도 출력
    if true_labels is not None:
        accuracy = (correct / total) * 100
        print("\n" + "=" * 70)
        print(f"테스트 정확도: {correct}/{total} = {accuracy:.2f}%")
        print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='예측 테스트')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='학습된 모델 경로')
    parser.add_argument('--data_path', type=str, default='./data_downsampled',
                       help='데이터 경로 (샘플 생성용)')
    parser.add_argument('--test_file', type=str, default=None,
                       help='테스트 파일 (없으면 자동 생성)')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='생성할 샘플 수')
    parser.add_argument('--create_only', action='store_true',
                       help='샘플만 생성하고 예측 안함')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "🎯" * 35)
    print("예측 테스트 스크립트")
    print("🎯" * 35 + "\n")
    
    # 테스트 파일이 없으면 생성
    if args.test_file is None:
        test_file, true_labels, label_names = create_test_samples(
            args.data_path,
            n_samples=args.n_samples,
            output_file='test_samples.npy'
        )
    else:
        test_file = args.test_file
        true_labels = None
        label_names = None
        print(f"테스트 파일 사용: {test_file}")
    
    # 샘플만 생성
    if args.create_only:
        print("\n샘플 생성 완료! (예측은 실행하지 않음)")
        return
    
    # 예측 실행
    test_prediction(args.model_path, test_file, true_labels, label_names)
    
    print("\n" + "✅" * 35)
    print("테스트 완료!")
    print("✅" * 35 + "\n")


if __name__ == '__main__':
    main()
