"""
models.py - 기어박스 고장 진단 모델 정의

다양한 딥러닝 모델 아키텍처를 제공합니다.
"""

import torch
import torch.nn as nn


class GearboxCNN1D(nn.Module):
    """
    1D CNN 모델 - 시계열 진동 신호에 적합
    
    Args:
        input_size (int): 입력 신호 길이
        num_classes (int): 클래스 개수
    """
    
    def __init__(self, input_size, num_classes):
        super(GearboxCNN1D, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block for ResNet"""
    
    def __init__(self, main_path, shortcut):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.shortcut = shortcut
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        out += residual
        out = self.relu(out)
        return out


class GearboxResNet1D(nn.Module):
    """
    ResNet 스타일 1D CNN - 더 깊은 네트워크
    
    Args:
        input_size (int): 입력 신호 길이
        num_classes (int): 클래스 개수
    """
    
    def __init__(self, input_size, num_classes):
        super(GearboxResNet1D, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.Sequential(
            self._make_residual_block(64, 64),
            self._make_residual_block(64, 128, stride=2),
            self._make_residual_block(128, 256, stride=2),
            self._make_residual_block(256, 512, stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Residual Block 생성"""
        layers = []
        
        # 첫 번째 conv
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        
        # 두 번째 conv
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        return ResidualBlock(nn.Sequential(*layers), shortcut)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.residual_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GearboxMLP(nn.Module):
    """
    Multi-Layer Perceptron - 특징 벡터에 적합
    
    Args:
        input_size (int): 입력 특징 차원
        num_classes (int): 클래스 개수
    """
    
    def __init__(self, input_size, num_classes):
        super(GearboxMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


def get_model(model_name, input_size, num_classes):
    """
    모델 팩토리 함수
    
    Args:
        model_name (str): 모델 이름 ('CNN', 'ResNet', 'MLP')
        input_size (int): 입력 크기
        num_classes (int): 클래스 개수
    
    Returns:
        nn.Module: 선택된 모델
    """
    models = {
        'CNN': GearboxCNN1D,
        'ResNet': GearboxResNet1D,
        'MLP': GearboxMLP
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](input_size, num_classes)


if __name__ == '__main__':
    # 모델 테스트
    input_size = 1024
    num_classes = 4
    batch_size = 8
    
    print("=" * 60)
    print("모델 테스트")
    print("=" * 60)
    
    for model_name in ['CNN', 'ResNet', 'MLP']:
        print(f"\n{model_name} 모델:")
        model = get_model(model_name, input_size, num_classes)
        
        # 더미 입력
        x = torch.randn(batch_size, input_size)
        
        # Forward pass
        output = model(x)
        
        print(f"  입력 shape: {x.shape}")
        print(f"  출력 shape: {output.shape}")
        print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
