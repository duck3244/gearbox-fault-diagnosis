"""
inference.py - 기어박스 고장 진단 추론

사용법:
    python inference.py --model_path results/*/best_model.pth --input_data test.csv
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path


class GearboxPredictor:
    """기어박스 고장 진단 예측기"""

    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path (str): 학습된 모델 경로
            device (str): 디바이스 ('cpu' 또는 'cuda')
        """
        self.device = torch.device(device)

        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.label_names = checkpoint['label_names']
        self.scaler = checkpoint['scaler']

        # 모델 생성 및 가중치 로드
        input_size = self.scaler.mean_.shape[0]
        num_classes = len(self.label_names)

        # 모델 타입 자동 감지
        model_type = self._detect_model_type(checkpoint['model_state_dict'])

        from models import get_model
        self.model = get_model(model_type, input_size, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"모델 로드 완료: {model_path}")
        print(f"모델 타입: {model_type}")
        print(f"클래스: {self.label_names}")

    def _detect_model_type(self, state_dict):
        """
        state_dict에서 모델 타입 자동 감지

        Args:
            state_dict: 모델 가중치 딕셔너리

        Returns:
            str: 모델 타입 ('CNN', 'ResNet', 'MLP')
        """
        keys = list(state_dict.keys())

        # ResNet 감지
        if any('residual_blocks' in key for key in keys):
            return 'ResNet'

        # MLP 감지
        elif any('layers' in key for key in keys):
            return 'MLP'

        # CNN (기본)
        else:
            return 'CNN'

    def predict_single(self, signal_data):
        """단일 샘플 예측"""
        # 전처리
        signal_data = np.array(signal_data).reshape(1, -1)
        signal_data = self.scaler.transform(signal_data)

        # 텐서 변환
        signal_tensor = torch.FloatTensor(signal_data).to(self.device)

        # 예측
        with torch.no_grad():
            outputs = self.model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        return {
            'predicted_class': self.label_names[predicted_class],
            'predicted_index': predicted_class,
            'confidence': confidence,
            'probabilities': {
                name: prob.item()
                for name, prob in zip(self.label_names, probabilities[0])
            }
        }

    def predict_batch(self, signal_data_list):
        """배치 예측"""
        # 전처리
        signal_data = np.array(signal_data_list)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)

        signal_data = self.scaler.transform(signal_data)
        signal_tensor = torch.FloatTensor(signal_data).to(self.device)

        # 예측
        with torch.no_grad():
            outputs = self.model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

        results = []
        for i in range(len(signal_data)):
            result = {
                'predicted_class': self.label_names[predicted_classes[i].item()],
                'predicted_index': predicted_classes[i].item(),
                'confidence': probabilities[i, predicted_classes[i]].item(),
                'probabilities': {
                    name: prob.item()
                    for name, prob in zip(self.label_names, probabilities[i])
                }
            }
            results.append(result)

        return results

    def predict_from_csv(self, csv_path):
        """CSV 파일에서 예측"""
        df = pd.read_csv(csv_path)

        # 레이블 컬럼 제거 (있는 경우)
        label_cols = ['label', 'class', 'target', 'condition', 'state', 'fault', 'status']
        for col in label_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"레이블 컬럼 '{col}' 제거됨")

        signal_data = df.values
        return self.predict_batch(signal_data)

    def predict_from_npy(self, npy_path):
        """NumPy 파일에서 예측"""
        signal_data = np.load(npy_path)

        if signal_data.ndim == 1:
            return self.predict_single(signal_data)
        else:
            return self.predict_batch(signal_data)


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Gearbox Fault Diagnosis Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model_path', type=str, required=True,
                       help='학습된 모델 파일 경로')
    parser.add_argument('--input_data', type=str, default=None,
                       help='입력 데이터 파일 (.csv 또는 .npy)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='예측 결과 저장 파일')
    parser.add_argument('--no_cuda', action='store_true',
                       help='CPU 사용 강제')

    return parser.parse_args()


def interactive_mode(predictor):
    """대화형 예측 모드"""
    print("\n" + "=" * 70)
    print("대화형 예측 모드")
    print("=" * 70)
    print("진동 신호 데이터를 입력하세요 (쉼표로 구분)")
    print("종료하려면 'quit' 입력\n")

    while True:
        user_input = input("신호 데이터: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("종료합니다.")
            break

        try:
            signal_data = [float(x) for x in user_input.split(',')]
            result = predictor.predict_single(signal_data)

            print(f"\n예측 클래스: {result['predicted_class']}")
            print(f"신뢰도: {result['confidence']*100:.2f}%")
            print("클래스별 확률:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob*100:.2f}%")
            print()

        except Exception as e:
            print(f"오류 발생: {e}\n")


def main():
    args = parse_args()

    # 디바이스 설정
    from utils import get_device
    device = get_device(use_cuda=not args.no_cuda)
    
    # 예측기 초기화
    print("\n" + "=" * 70)
    print("기어박스 고장 진단 추론")
    print("=" * 70)
    
    predictor = GearboxPredictor(args.model_path, device=str(device))
    
    # 입력 데이터 처리
    if args.input_data:
        input_path = Path(args.input_data)
        
        if not input_path.exists():
            print(f"오류: 파일을 찾을 수 없습니다: {input_path}")
            return
        
        print(f"\n입력 파일: {input_path}")
        
        # CSV 파일
        if input_path.suffix == '.csv':
            results = predictor.predict_from_csv(input_path)
            
            print("\n" + "=" * 70)
            print(f"예측 완료: {len(results)}개 샘플")
            print("=" * 70)
            
            # 통계 출력
            class_counts = {}
            for result in results:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            print("\n클래스별 예측 통계:")
            for class_name, count in sorted(class_counts.items()):
                percentage = count / len(results) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            # 평균 신뢰도
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"\n평균 신뢰도: {avg_confidence*100:.2f}%")
            
            # 결과 저장
            if args.output:
                output_df = pd.DataFrame([
                    {
                        'Sample': i+1,
                        'Predicted_Class': r['predicted_class'],
                        'Confidence': r['confidence'],
                        **{f'Prob_{k}': v for k, v in r['probabilities'].items()}
                    }
                    for i, r in enumerate(results)
                ])
                output_df.to_csv(args.output, index=False)
                print(f"\n결과 저장: {args.output}")
        
        # NumPy 파일
        elif input_path.suffix == '.npy':
            result = predictor.predict_from_npy(input_path)
            
            if isinstance(result, dict):
                # 단일 샘플
                print("\n" + "=" * 70)
                print("예측 결과")
                print("=" * 70)
                print(f"예측 클래스: {result['predicted_class']}")
                print(f"신뢰도: {result['confidence']*100:.2f}%")
                print("\n클래스별 확률:")
                for class_name, prob in result['probabilities'].items():
                    print(f"  {class_name}: {prob*100:.2f}%")
            else:
                # 다중 샘플
                print("\n" + "=" * 70)
                print(f"예측 완료: {len(result)}개 샘플")
                print("=" * 70)
                
                class_counts = {}
                for r in result:
                    pred_class = r['predicted_class']
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                
                print("\n클래스별 예측 통계:")
                for class_name, count in sorted(class_counts.items()):
                    percentage = count / len(result) * 100
                    print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        else:
            print(f"지원하지 않는 파일 형식: {input_path.suffix}")
            print("지원 형식: .csv, .npy")
    
    else:
        # 대화형 모드
        interactive_mode(predictor)


if __name__ == '__main__':
    main()