"""
dataset.py - 데이터셋 클래스 및 데이터 로딩 함수

PyTorch Dataset 클래스와 Kaggle 데이터 로딩 기능을 제공합니다.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


class GearboxDataset(Dataset):
    """
    기어박스 진동 신호 데이터셋
    
    Args:
        features (np.ndarray): 특징 데이터
        labels (np.ndarray): 레이블 데이터
    """
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_csv_data(csv_path):
    """
    CSV 파일에서 데이터 로드
    
    Args:
        csv_path (str): CSV 파일 경로
    
    Returns:
        tuple: (X, y, label_names)
    """
    df = pd.read_csv(csv_path)
    
    print(f"CSV 데이터 shape: {df.shape}")
    print(f"컬럼: {df.columns.tolist()[:10]}...")  # 처음 10개만 출력
    
    # 레이블 컬럼 찾기
    label_candidates = ['label', 'class', 'target', 'condition', 'state', 'fault', 'status']
    label_col = None
    
    for col in df.columns:
        if col.lower() in label_candidates:
            label_col = col
            break
    
    if label_col is None:
        # 마지막 컬럼을 레이블로 가정
        label_col = df.columns[-1]
        print(f"레이블 컬럼 자동 선택: {label_col}")
    else:
        print(f"레이블 컬럼 발견: {label_col}")
    
    # 데이터 분리
    y = df[label_col].values
    X = df.drop(columns=[label_col]).values
    
    # 레이블 인코딩
    if y.dtype == object or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_names = list(le.classes_)
    else:
        label_names = [f'Class_{i}' for i in sorted(np.unique(y))]
    
    return X, y, label_names


def load_folder_structure(data_dir):
    """
    폴더 구조에서 데이터 로드 (각 클래스별 폴더)
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
    
    Returns:
        tuple: (X, y, label_names)
    """
    data_path = Path(data_dir)
    X_list = []
    y_list = []
    label_names = []
    
    print("폴더 구조에서 데이터 로딩...")
    
    for label_idx, class_folder in enumerate(sorted(data_path.iterdir())):
        if not class_folder.is_dir():
            continue
        
        label_names.append(class_folder.name)
        print(f"  로딩 중: {class_folder.name}")
        
        file_count = 0
        
        # CSV 파일 로드
        for file_path in class_folder.glob('*.csv'):
            data = pd.read_csv(file_path).values.flatten()
            X_list.append(data)
            y_list.append(label_idx)
            file_count += 1
        
        # MAT 파일 로드
        for file_path in class_folder.glob('*.mat'):
            mat_data = loadmat(file_path)
            # MAT 파일의 데이터 키 찾기
            for key in mat_data.keys():
                if not key.startswith('__'):
                    data = mat_data[key].flatten()
                    X_list.append(data)
                    y_list.append(label_idx)
                    file_count += 1
                    break
        
        # NumPy 파일 로드
        for file_path in class_folder.glob('*.npy'):
            data = np.load(file_path).flatten()
            X_list.append(data)
            y_list.append(label_idx)
            file_count += 1
        
        print(f"    파일 수: {file_count}")
    
    # 모든 샘플을 같은 길이로 맞추기
    if len(X_list) > 0:
        max_len = max(len(x) for x in X_list)
        X = np.array([
            np.pad(x, (0, max_len - len(x)), mode='constant') 
            if len(x) < max_len else x[:max_len] 
            for x in X_list
        ])
        y = np.array(y_list)
    else:
        raise ValueError("데이터 파일을 찾을 수 없습니다.")
    
    return X, y, label_names


def load_mat_file(mat_path):
    """
    MATLAB 파일에서 데이터 로드
    
    Args:
        mat_path (str): MAT 파일 경로
    
    Returns:
        tuple: (X, y, label_names)
    """
    print(f"MAT 파일에서 데이터 로딩: {mat_path}")
    mat_data = loadmat(mat_path)
    
    # 일반적인 키 이름들
    data_keys = ['X', 'data', 'features', 'signals']
    label_keys = ['y', 'labels', 'targets', 'classes']
    
    X = None
    y = None
    
    # 데이터 찾기
    for key in data_keys:
        if key in mat_data:
            X = mat_data[key]
            print(f"  데이터 키 발견: {key}")
            break
    
    # 레이블 찾기
    for key in label_keys:
        if key in mat_data:
            y = mat_data[key].flatten()
            print(f"  레이블 키 발견: {key}")
            break
    
    if X is None or y is None:
        print("  사용 가능한 키:", [k for k in mat_data.keys() if not k.startswith('__')])
        raise ValueError("MAT 파일에서 데이터 또는 레이블을 찾을 수 없습니다.")
    
    # 레이블 이름
    if 'label_names' in mat_data:
        label_names = [str(name[0]) for name in mat_data['label_names'][0]]
    else:
        label_names = [f'Class_{i}' for i in sorted(np.unique(y))]
    
    return X, y, label_names


def load_kaggle_gearbox_data(data_path):
    """
    Kaggle 기어박스 데이터셋 자동 로드
    여러 형식을 지원하고 자동으로 감지합니다.
    
    Args:
        data_path (str): 데이터 경로
    
    Returns:
        tuple: (X, y, label_names)
    """
    data_path = Path(data_path)
    
    print("=" * 60)
    print("데이터 로딩 중...")
    print("=" * 60)
    
    # 0. NumPy 파일 우선 확인 (다운샘플링된 데이터)
    if (data_path / 'X_downsampled.npy').exists():
        print("다운샘플링된 NumPy 데이터 로드")
        X = np.load(data_path / 'X_downsampled.npy')
        y = np.load(data_path / 'y.npy')
        label_names = np.load(data_path / 'label_names.npy', allow_pickle=True).tolist()

    # 1. CSV 파일 찾기
    elif list(data_path.glob('*.csv')):
        csv_files = list(data_path.glob('*.csv'))
        print(f"CSV 파일 발견: {len(csv_files)}개")
        X, y, label_names = load_csv_data(csv_files[0])

    # 2. MAT 파일 찾기
    elif (data_path / 'data.mat').exists():
        X, y, label_names = load_mat_file(data_path / 'data.mat')

    # 3. 폴더 구조 확인 (클래스별 폴더)
    elif any(d.is_dir() and not d.name.startswith('.') and not d.name.startswith('_')
             and d.name not in ['results', 'data_downsampled', '__pycache__', '.idea']
             for d in data_path.iterdir()):
        X, y, label_names = load_folder_structure(data_path)

    else:
        raise FileNotFoundError(
            f"데이터를 찾을 수 없습니다: {data_path}\n"
            "지원 형식:\n"
            "  1. NumPy 파일 (X_downsampled.npy, y.npy, label_names.npy)\n"
            "  2. CSV 파일\n"
            "  3. MAT 파일\n"
            "  4. 폴더 구조 (클래스별 폴더)\n\n"
            "다운샘플링을 먼저 실행하세요:\n"
            "  python downsample_data.py --data_path ./data --target_size 8192"
        )

    # 데이터 정보 출력
    print("\n" + "=" * 60)
    print("데이터 로드 완료")
    print("=" * 60)
    print(f"전체 샘플 수: {len(X)}")
    print(f"특징 차원: {X.shape[1]}")
    print(f"클래스 수: {len(label_names)}")
    print(f"클래스 이름: {label_names}")

    print("\n클래스별 샘플 수:")
    for i, name in enumerate(label_names):
        count = np.sum(y == i)
        print(f"  {name}: {count} ({count/len(y)*100:.1f}%)")

    return X, y, label_names


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size=32, num_workers=2, train_sampler=None):
    """
    DataLoader 생성

    Args:
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        X_test, y_test: 테스트 데이터
        batch_size (int): 배치 크기
        num_workers (int): 워커 수
        train_sampler: (선택) train 로더용 Sampler. 지정 시 shuffle=False로 강제됨.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = GearboxDataset(X_train, y_train)
    val_dataset = GearboxDataset(X_val, y_val)
    test_dataset = GearboxDataset(X_test, y_test)

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # BatchNorm이 batch_size=1에서 깨지지 않도록
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 테스트
    print("데이터셋 모듈 테스트\n")

    # 더미 데이터 생성
    X = np.random.randn(100, 1024)
    y = np.random.randint(0, 4, 100)

    # 데이터셋 생성
    dataset = GearboxDataset(X, y)

    print(f"데이터셋 크기: {len(dataset)}")
    print(f"첫 번째 샘플 shape: {dataset[0][0].shape}")
    print(f"첫 번째 레이블: {dataset[0][1]}")