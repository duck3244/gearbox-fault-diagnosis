"""
main.py - 메인 실행 파일

간단한 인터페이스로 학습과 추론을 실행할 수 있습니다.

사용법:
    # 학습
    python main.py train --data_path ./data --model CNN
    
    # 추론
    python main.py predict --model_path results/best_model.pth --input_data test.csv
    
    # 설정 확인
    python main.py config
"""

import argparse
import sys
from pathlib import Path

# 로컬 모듈 import
from config import get_config
from train import main as train_main
from inference import main as inference_main


def train_command(args):
    """학습 명령 실행"""
    print("\n" + "=" * 70)
    print("학습 모드")
    print("=" * 70)
    
    # config 파일의 설정 사용
    config = get_config(args.model)
    
    # 명령줄 인자로 오버라이드
    if args.data_path:
        config.DATA_DIR = Path(args.data_path)
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # 설정 표시
    config.display()
    
    # 학습 실행
    sys.argv = ['train.py',
                '--data_path', str(config.DATA_DIR),
                '--model', args.model,
                '--batch_size', str(config.BATCH_SIZE),
                '--epochs', str(config.EPOCHS),
                '--lr', str(config.LEARNING_RATE)]
    
    if args.no_cuda:
        sys.argv.append('--no_cuda')
    
    train_main()


def predict_command(args):
    """추론 명령 실행"""
    print("\n" + "=" * 70)
    print("추론 모드")
    print("=" * 70)
    
    sys.argv = ['inference.py',
                '--model_path', args.model_path]
    
    if args.input_data:
        sys.argv.extend(['--input_data', args.input_data])
    if args.output:
        sys.argv.extend(['--output', args.output])
    if args.no_cuda:
        sys.argv.append('--no_cuda')
    
    inference_main()


def config_command(args):
    """설정 확인 명령"""
    config = get_config(args.model)
    config.display()


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Gearbox Fault Diagnosis - Main Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # CNN 모델 학습
  python main.py train --data_path ./data --model CNN --epochs 100
  
  # ResNet 모델 학습
  python main.py train --data_path ./data --model ResNet --epochs 150
  
  # 추론
  python main.py predict --model_path results/best_model.pth --input_data test.csv
  
  # 대화형 추론
  python main.py predict --model_path results/best_model.pth
  
  # 설정 확인
  python main.py config --model CNN
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='명령')
    
    # Train 명령
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--data_path', type=str, help='데이터 경로')
    train_parser.add_argument('--model', type=str, default='CNN',
                             choices=['CNN', 'ResNet', 'MLP'],
                             help='모델 타입')
    train_parser.add_argument('--batch_size', type=int, help='배치 크기')
    train_parser.add_argument('--epochs', type=int, help='에폭 수')
    train_parser.add_argument('--lr', type=float, help='학습률')
    train_parser.add_argument('--no_cuda', action='store_true', help='CPU 사용')
    
    # Predict 명령
    predict_parser = subparsers.add_parser('predict', help='예측')
    predict_parser.add_argument('--model_path', type=str, required=True,
                               help='모델 파일 경로')
    predict_parser.add_argument('--input_data', type=str,
                               help='입력 데이터 파일')
    predict_parser.add_argument('--output', type=str,
                               help='결과 저장 파일')
    predict_parser.add_argument('--no_cuda', action='store_true', help='CPU 사용')
    
    # Config 명령
    config_parser = subparsers.add_parser('config', help='설정 확인')
    config_parser.add_argument('--model', type=str, default='CNN',
                              choices=['CNN', 'ResNet', 'MLP'],
                              help='모델 타입')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'config':
        config_command(args)
    else:
        print("명령을 선택하세요: train, predict, config")
        print("도움말: python main.py --help")


if __name__ == '__main__':
    main()
