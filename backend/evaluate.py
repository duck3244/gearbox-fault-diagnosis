"""
evaluate.py - 모델 평가 스크립트

학습된 모델의 성능을 평가하고 상세한 분석을 제공합니다.

사용법:
    python evaluate.py --model_path results/best_model.pth --data_path ./data
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score
)
from pathlib import Path

from sklearn.model_selection import train_test_split

from models import get_model
from dataset import load_kaggle_gearbox_data, GearboxDataset
from torch.utils.data import DataLoader
from utils import get_device


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path (str): 모델 파일 경로
            device (str): 디바이스
        """
        self.device = torch.device(device)

        # 체크포인트 로드 (scaler 등 pickle 객체 포함 → weights_only=False 필수)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.label_names = checkpoint['label_names']
        self.scaler = checkpoint['scaler']

        # 모델 로드
        input_size = self.scaler.mean_.shape[0]
        num_classes = len(self.label_names)

        # state_dict에서 모델 타입 자동 감지
        model_type = self._detect_model_type(checkpoint['model_state_dict'])

        self.model = get_model(model_type, input_size, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"모델 로드 완료: {model_path}")
        print(f"모델 타입: {model_type}")

    @staticmethod
    def _detect_model_type(state_dict):
        """state_dict에서 모델 타입(CNN/ResNet/MLP)을 감지"""
        keys = list(state_dict.keys())
        if any('residual_blocks' in k for k in keys):
            return 'ResNet'
        if any(k.startswith('layers.') for k in keys):
            return 'MLP'
        return 'CNN'
    
    def evaluate(self, test_loader):
        """
        모델 평가
        
        Args:
            test_loader: 테스트 데이터 로더
        
        Returns:
            dict: 평가 결과
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 메트릭 계산
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        results = {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Confusion Matrix 시각화"""
        cm = confusion_matrix(y_true, y_pred)
        
        # 정규화된 confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 절대값
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names, yticklabels=self.label_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        
        # 정규화
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.label_names, yticklabels=self.label_names,
                   ax=axes[1], cbar_kws={'label': 'Ratio'})
        axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion Matrix 저장: {save_path}")
    
    def plot_roc_curves(self, y_true, y_probs, save_path='roc_curves.png'):
        """ROC 곡선 시각화"""
        from sklearn.preprocessing import label_binarize
        
        n_classes = len(self.label_names)

        # 클래스가 2개인 경우 특별 처리
        if n_classes == 2:
            fig, ax = plt.subplots(figsize=(10, 8))

            # 이진 분류의 경우
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, linewidth=2,
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')

            # 대각선
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title('ROC Curve (Binary Classification)', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)

        else:
            # 다중 분류
            # 이진화
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            fig, ax = plt.subplots(figsize=(10, 8))

            # 각 클래스별 ROC 곡선
            for i, class_name in enumerate(self.label_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, linewidth=2,
                       label=f'{class_name} (AUC = {roc_auc:.3f})')

            # 대각선
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC 곡선 저장: {save_path}")

    def plot_precision_recall_curves(self, y_true, y_probs, save_path='pr_curves.png'):
        """Precision-Recall 곡선 시각화"""
        from sklearn.preprocessing import label_binarize

        n_classes = len(self.label_names)

        # 클래스가 2개인 경우 특별 처리
        if n_classes == 2:
            fig, ax = plt.subplots(figsize=(10, 8))

            # 이진 분류
            precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])

            ax.plot(recall, precision, linewidth=2, label='PR Curve')

            ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax.set_title('Precision-Recall Curve (Binary Classification)',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

        else:
            # 다중 분류
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            fig, ax = plt.subplots(figsize=(10, 8))

            for i, class_name in enumerate(self.label_names):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])

                ax.plot(recall, precision, linewidth=2, label=f'{class_name}')

            ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PR 곡선 저장: {save_path}")

    def plot_class_distribution(self, y_true, y_pred, save_path='class_distribution.png'):
        """클래스 분포 비교"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 실제 분포
        true_counts = [np.sum(y_true == i) for i in range(len(self.label_names))]
        axes[0].bar(self.label_names, true_counts, color='steelblue', alpha=0.7)
        axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')

        # 예측 분포
        pred_counts = [np.sum(y_pred == i) for i in range(len(self.label_names))]
        axes[1].bar(self.label_names, pred_counts, color='coral', alpha=0.7)
        axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"클래스 분포 저장: {save_path}")

    def generate_report(self, results, output_dir):
        """
        평가 리포트 생성

        Args:
            results (dict): 평가 결과
            output_dir (Path): 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        y_true = results['labels']
        y_pred = results['predictions']
        y_probs = results['probabilities']

        print("\n" + "=" * 70)
        print("평가 결과")
        print("=" * 70)
        print(f"정확도: {results['accuracy']*100:.2f}%")
        print(f"정밀도: {results['precision']*100:.2f}%")
        print(f"재현율: {results['recall']*100:.2f}%")
        print(f"F1 Score: {results['f1_score']*100:.2f}%")

        # Classification Report
        print("\n" + "=" * 70)
        print("Classification Report")
        print("=" * 70)
        print(classification_report(y_true, y_pred,
                                   target_names=self.label_names,
                                   digits=4))

        # 시각화
        print("\n" + "=" * 70)
        print("시각화 생성 중...")
        print("=" * 70)

        self.plot_confusion_matrix(y_true, y_pred,
                                  output_dir / 'confusion_matrix.png')
        self.plot_roc_curves(y_true, y_probs,
                           output_dir / 'roc_curves.png')
        self.plot_precision_recall_curves(y_true, y_probs,
                                         output_dir / 'pr_curves.png')
        self.plot_class_distribution(y_true, y_pred,
                                    output_dir / 'class_distribution.png')

        print(f"\n모든 결과가 '{output_dir}'에 저장되었습니다.")


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Gearbox Fault Diagnosis Model Evaluation'
    )

    parser.add_argument('--model_path', type=str, required=True,
                       help='학습된 모델 파일 경로')
    parser.add_argument('--data_path', type=str, required=True,
                       help='데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--no_cuda', action='store_true',
                       help='CPU 사용 강제')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='train.py와 동일해야 함 (기본 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='train.py와 동일한 시드로 동일한 test split 재현')
    parser.add_argument('--use_all_data', action='store_true',
                       help='분할 없이 전체 데이터 평가 (데이터 누수 주의)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 디바이스 설정
    device = get_device(use_cuda=not args.no_cuda)

    print("\n" + "=" * 70)
    print("모델 평가")
    print("=" * 70)

    # 평가기 초기화
    evaluator = ModelEvaluator(args.model_path, device=str(device))

    # 데이터 로드
    print("\n데이터 로딩 중...")
    X, y, label_names = load_kaggle_gearbox_data(args.data_path)

    # train.py와 동일한 분할로 테스트 서브셋만 사용 (데이터 누수 방지)
    if args.use_all_data:
        print("⚠️  --use_all_data: 학습 데이터 포함 전체를 평가합니다 (데이터 누수 가능)")
        X_eval, y_eval = X, y
    else:
        _, X_eval, _, y_eval = train_test_split(
            X, y, test_size=args.test_size,
            random_state=args.seed, stratify=y
        )
        print(f"테스트 분할 적용 (test_size={args.test_size}, seed={args.seed})")

    # 데이터 정규화
    X_scaled = evaluator.scaler.transform(X_eval)

    # 데이터셋 및 로더 생성
    test_dataset = GearboxDataset(X_scaled, y_eval)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    print(f"테스트 샘플 수: {len(test_dataset)}")

    # 평가 실행
    print("\n평가 중...")
    results = evaluator.evaluate(test_loader)

    # 리포트 생성
    evaluator.generate_report(results, args.output_dir)


if __name__ == '__main__':
    main()