"""
train.py - 기어박스 고장 진단 모델 학습

사용법:
    python train.py --data_path ./data --model CNN --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import get_model
from dataset import load_kaggle_gearbox_data, create_dataloaders
from utils import (
    train_epoch, validate_epoch, save_checkpoint,
    plot_training_history, plot_confusion_matrix,
    print_classification_report, count_parameters,
    set_seed, EarlyStopping, get_device, create_output_directory
)


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Gearbox Fault Diagnosis Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 데이터 관련
    parser.add_argument('--data_path', type=str, default='./data',
                       help='데이터 디렉토리 경로')
    
    # 모델 관련
    parser.add_argument('--model', type=str, default='CNN',
                       choices=['CNN', 'ResNet', 'MLP'],
                       help='모델 타입')
    
    # 학습 관련
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100,
                       help='에폭 수')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (L2 정규화)')
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='데이터 로더 워커 수')
    parser.add_argument('--no_cuda', action='store_true',
                       help='CUDA 사용 안함')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    
    return parser.parse_args()


def main():
    # 인자 파싱
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 디바이스 설정
    device = get_device(use_cuda=not args.no_cuda)
    
    # 출력 디렉토리 생성
    output_dir = create_output_directory(args.output_dir)
    
    print("\n" + "=" * 70)
    print("기어박스 고장 진단 모델 학습")
    print("=" * 70)
    print(f"모델: {args.model}")
    print(f"배치 크기: {args.batch_size}")
    print(f"에폭: {args.epochs}")
    print(f"학습률: {args.lr}")
    print(f"디바이스: {device}")
    
    # 데이터 로드
    print("\n" + "=" * 70)
    print("1. 데이터 로딩")
    print("=" * 70)
    
    X, y, label_names = load_kaggle_gearbox_data(args.data_path)
    
    # 데이터 분할
    print("\n데이터 분할 중...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.18, 
        random_state=args.seed, stratify=y_train_val
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # 정규화
    print("\n데이터 정규화 중...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # DataLoader 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 모델 생성
    print("\n" + "=" * 70)
    print("2. 모델 생성")
    print("=" * 70)
    
    input_size = X_train.shape[1]
    num_classes = len(label_names)
    
    model = get_model(args.model, input_size, num_classes)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, 
        verbose=True, min_lr=1e-6
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # 학습
    print("\n" + "=" * 70)
    print("3. 학습 시작")
    print("=" * 70)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 검증
        val_loss, val_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 기록
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # 출력
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc, label_names, scaler,
                output_dir / 'best_model.pth'
            )
            print(f'  *** 최고 모델 저장 (Val Acc: {val_acc:.2f}%) ***')
        
        # Early stopping 체크
        if early_stopping(val_acc):
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
    
    # 학습 곡선 시각화
    print("\n" + "=" * 70)
    print("4. 학습 결과 시각화")
    print("=" * 70)
    
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        save_path=output_dir / 'training_history.png'
    )
    
    # 최고 모델로 테스트
    print("\n" + "=" * 70)
    print("5. 테스트 평가")
    print("=" * 70)
    
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f'\n테스트 결과:')
    print(f'  Test Loss: {test_loss:.4f}')
    print(f'  Test Accuracy: {test_acc:.2f}%')
    
    # Classification Report
    print_classification_report(test_labels, test_preds, label_names)
    
    # Confusion Matrix
    plot_confusion_matrix(
        test_labels, test_preds, label_names,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    # 최종 결과 저장
    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)
    print(f"최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"최종 테스트 정확도: {test_acc:.2f}%")
    print(f"\n결과 저장 위치: {output_dir}")
    print(f"  - 모델: {output_dir / 'best_model.pth'}")
    print(f"  - 학습 곡선: {output_dir / 'training_history.png'}")
    print(f"  - Confusion Matrix: {output_dir / 'confusion_matrix.png'}")


if __name__ == '__main__':
    main()
