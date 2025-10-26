"""
utils.py - 유틸리티 함수

학습, 평가, 시각화 등의 보조 함수를 제공합니다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


def train_epoch(model, loader, criterion, optimizer, device):
    """
    한 에폭 학습
    
    Args:
        model: 학습할 모델
        loader: 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
    
    Returns:
        tuple: (평균 손실, 정확도)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    """
    한 에폭 검증

    Args:
        model: 검증할 모델
        loader: 데이터 로더
        criterion: 손실 함수
        device: 디바이스

    Returns:
        tuple: (평균 손실, 정확도, 예측값, 실제값)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, val_acc, label_names, scaler, save_path):
    """
    체크포인트 저장

    Args:
        model: 모델
        optimizer: 옵티마이저
        epoch: 현재 에폭
        val_acc: 검증 정확도
        label_names: 레이블 이름 리스트
        scaler: StandardScaler 객체
        save_path: 저장 경로
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'label_names': label_names,
        'scaler': scaler
    }, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    체크포인트 로드

    Args:
        checkpoint_path: 체크포인트 경로
        model: 모델
        optimizer: 옵티마이저 (선택사항)
        device: 디바이스

    Returns:
        dict: 체크포인트 정보
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def plot_training_history(train_losses, train_accs, val_losses, val_accs,
                          save_path='training_history.png'):
    """
    학습 히스토리 시각화

    Args:
        train_losses: 학습 손실 리스트
        train_accs: 학습 정확도 리스트
        val_losses: 검증 손실 리스트
        val_accs: 검증 정확도 리스트
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2, color='#2E86AB')
    axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"학습 히스토리 저장: {save_path}")


def plot_confusion_matrix(y_true, y_pred, label_names,
                          save_path='confusion_matrix.png'):
    """
    Confusion Matrix 시각화

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        label_names: 레이블 이름 리스트
        save_path: 저장 경로
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix 저장: {save_path}")


def print_classification_report(y_true, y_pred, label_names):
    """
    분류 리포트 출력

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        label_names: 레이블 이름 리스트
    """
    print('\n' + '=' * 60)
    print('Classification Report:')
    print('=' * 60)
    print(classification_report(y_true, y_pred,
                                target_names=label_names,
                                digits=4))


def count_parameters(model):
    """
    모델 파라미터 수 계산

    Args:
        model: PyTorch 모델

    Returns:
        tuple: (총 파라미터 수, 학습 가능한 파라미터 수)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def set_seed(seed=42):
    """
    재현성을 위한 시드 설정

    Args:
        seed (int): 시드 값
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early Stopping 클래스"""

    def __init__(self, patience=10, min_delta=0, mode='max'):
        """
        Args:
            patience (int): 개선이 없을 때 기다리는 에폭 수
            min_delta (float): 개선으로 간주할 최소 변화량
            mode (str): 'max' (정확도) 또는 'min' (손실)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Args:
            score: 모니터링할 메트릭 값

        Returns:
            bool: Early stopping 여부
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


def get_device(use_cuda=True):
    """
    사용 가능한 디바이스 반환

    Args:
        use_cuda (bool): CUDA 사용 여부

    Returns:
        torch.device: 디바이스
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def create_output_directory(base_dir='results'):
    """
    출력 디렉토리 생성

    Args:
        base_dir (str): 기본 디렉토리

    Returns:
        Path: 생성된 디렉토리 경로
    """
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"출력 디렉토리 생성: {output_dir}")
    return output_dir


if __name__ == '__main__':
    print("유틸리티 모듈 테스트\n")

    # 시드 설정 테스트
    set_seed(42)
    print("✓ 시드 설정 완료")

    # Early Stopping 테스트
    print("\n✓ Early Stopping 테스트:")
    early_stopping = EarlyStopping(patience=3, mode='max')

    scores = [0.8, 0.82, 0.81, 0.80, 0.79, 0.78]
    for i, score in enumerate(scores):
        if early_stopping(score):
            print(f"  Early stopping at epoch {i+1}")
            break
        else:
            print(f"  Epoch {i+1}: score={score:.2f}, counter={early_stopping.counter}")

    # 디바이스 테스트
    print("\n✓ 디바이스 테스트:")
    device = get_device()
    print(f"  디바이스: {device}")

    # 파라미터 카운트 테스트
    print("\n✓ 파라미터 카운트 테스트:")

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(100, 10)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    total, trainable = count_parameters(model)
    print(f"  총 파라미터: {total:,}")
    print(f"  학습 가능: {trainable:,}")

    print("\n✓ 모든 테스트 완료!")