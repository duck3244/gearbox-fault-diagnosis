# Gearbox Fault Diagnosis - System Architecture

## 1. 개요

본 프로젝트는 기어박스(Gearbox)의 진동 신호를 기반으로 결함을 진단하는 딥러닝 시스템이다.
1D CNN, ResNet, MLP 세 가지 모델을 학습할 수 있으며, FastAPI 기반 추론 서버와
React + Vite 프론트엔드로 구성된 웹 인터페이스를 제공한다.

- **언어/프레임워크**: Python 3.x (PyTorch), TypeScript (React 18 + Vite)
- **주요 라이브러리**: PyTorch, FastAPI, scikit-learn, NumPy, Pandas, TailwindCSS
- **데이터**: Kaggle 기어박스 진동 데이터셋 (BrokenTooth / Healthy 등 다중 클래스)

---

## 2. 시스템 구조 (High-Level Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser (User)                              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP (Vite dev proxy: /api → :8000)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite + TS)                      │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐    │
│   │ ModelSelector│  │ UploadCard   │  │ PredictionResult       │    │
│   └──────────────┘  └──────────────┘  └────────────────────────┘    │
│   ┌──────────────┐  ┌──────────────────────────────────────────┐    │
│   │ HistoryTable │  │ api/client.ts (fetch wrapper, 타입 정의) │    │
│   └──────────────┘  └──────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ REST (multipart/form-data)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Backend - FastAPI (app/)                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  app/main.py  (CORS, 라우터 등록, uvicorn entrypoint)    │     │
│   └──────────────────────────────────────────────────────────┘     │
│   ┌─────────────┐ ┌──────────────┐ ┌────────────────────────┐     │
│   │ api/health  │ │ api/models   │ │ api/predict            │     │
│   │ /api/health │ │ /api/models  │ │ /api/predict           │     │
│   │             │ │ /.../{id}/   │ │ /api/history           │     │
│   │             │ │   meta       │ │                        │     │
│   └─────────────┘ └──────┬───────┘ └────────┬───────────────┘     │
│                          │                  │                       │
│                          ▼                  ▼                       │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  services/predictor_pool.py                              │     │
│   │  (LRU 캐시, asyncio.Lock 직렬화, 경로 검증)              │     │
│   └──────────────────────┬───────────────────────────────────┘     │
└──────────────────────────┼──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Backend - ML Core (backend/*.py)                    │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │ inference.py   GearboxPredictor                           │    │
│   │   - load checkpoint (model + scaler + label_names)        │    │
│   │   - predict_single / predict_batch                        │    │
│   └────────────────────────┬──────────────────────────────────┘    │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐       │
│   │ models.py    │ │ dataset.py   │ │ preprocessing.py     │       │
│   │ - CNN1D      │ │ - Dataset    │ │ - SignalPreprocessor │       │
│   │ - ResNet1D   │ │ - load_*     │ │ - FeatureExtractor   │       │
│   │ - MLP        │ │ - dataloader │ │ - augment_signal     │       │
│   └──────┬───────┘ └──────┬───────┘ └──────────────────────┘       │
│          │                │                                          │
│          ▼                ▼                                          │
│   ┌──────────────┐ ┌──────────────────────────────────────┐         │
│   │ utils.py     │ │ config.py                            │         │
│   │ - train_*    │ │ - CNNConfig / ResNetConfig / MLPCfg │         │
│   │ - early stop │ │                                      │         │
│   └──────────────┘ └──────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
                           ▲
                           │ load .pth
                           │
┌─────────────────────────────────────────────────────────────────────┐
│                      File System (Storage)                           │
│   data/                       원본 CSV (BrokenTooth, Healthy …)     │
│   data_downsampled/           X_downsampled.npy / y.npy / labels    │
│   results/{YYYYMMDD_HHMMSS}/  best_model.pth + 학습 결과 그림       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 모듈 책임 (Module Responsibilities)

### 3.1 ML Core (`backend/`)

| 파일 | 역할 |
|------|------|
| `config.py` | `Config` 베이스 + `CNNConfig`/`ResNetConfig`/`MLPConfig`. 데이터 분할 비율, 시드, 하이퍼파라미터 일원화. `get_config(model_type)` 팩토리. |
| `models.py` | `GearboxCNN1D`, `GearboxResNet1D`(`ResidualBlock` 포함), `GearboxMLP`. `get_model(name, input_size, num_classes)` 팩토리. |
| `dataset.py` | `GearboxDataset`(PyTorch Dataset), `load_kaggle_gearbox_data`(NPY/CSV/MAT/폴더 자동감지), `create_dataloaders`. |
| `preprocessing.py` | `SignalPreprocessor`(DC offset, bandpass, normalize, denoise, segment), `FeatureExtractor`(time/freq/statistical features), `augment_signal`. |
| `utils.py` | `train_epoch`, `validate_epoch`(AMP 지원), `EarlyStopping`, `save_checkpoint`/`load_checkpoint`, 시각화 유틸. |
| `train.py` | CLI 학습 엔트리포인트. 데이터 분할 → 스케일링 → DataLoader → 학습 루프 → 체크포인트 저장 → 테스트 평가. |
| `evaluate.py` | `ModelEvaluator`. 체크포인트 로드, 모델 타입 자동 감지(state_dict 키 기반), 혼동행렬·ROC·PR 곡선 생성. |
| `inference.py` | `GearboxPredictor`. CLI + 임포트 양용. `predict_single`/`predict_batch`/`predict_from_csv`/`predict_from_npy`. |
| `downsample_data.py` | 원본 CSV → 고정 길이(8192) NPY 변환 스크립트. |

### 3.2 FastAPI Layer (`backend/app/`)

| 파일 | 역할 |
|------|------|
| `app/main.py` | FastAPI 인스턴스 생성, CORS(localhost:5173 허용), 라우터 등록. |
| `app/schemas.py` | Pydantic v2 모델: `HealthResponse`, `ModelMeta`, `PredictionItem`, `PredictResponse`, `HistoryEntry`. |
| `app/api/health.py` | `GET /api/health` — CUDA 가용성, torch 버전. |
| `app/api/models.py` | `GET /api/models` 목록, `GET /api/models/{id}/meta` 상세. |
| `app/api/predict.py` | `POST /api/predict`(파일 업로드, 20MB 캡, 입력 dim 검증, asyncio.Lock 직렬화), `GET /api/history`(in-memory deque). |
| `app/services/predictor_pool.py` | `PredictorPool` LRU 캐시(기본 4개). 모델 ID 정규식 검증, 경로 트래버설 차단, 동시 로드 방지. |

### 3.3 Frontend (`frontend/src/`)

| 파일 | 역할 |
|------|------|
| `main.tsx` | React DOM 루트 마운트. |
| `App.tsx` | 글로벌 상태(health, models, modelId, meta, result, history) 관리, API 호출 오케스트레이션. |
| `api/client.ts` | fetch 래퍼 + 타입 정의(`Health`, `ModelMeta`, `PredictResponse`, `HistoryEntry` 등). |
| `components/ModelSelector.tsx` | 모델 ID 드롭다운 + 메타데이터 표시. |
| `components/UploadCard.tsx` | `.csv`/`.npy` 파일 업로드 UI. |
| `components/PredictionResult.tsx` | 예측 결과 시각화(클래스, 신뢰도, 확률 바). |
| `components/HistoryTable.tsx` | 최근 예측 이력 테이블. |

---

## 4. 학습 파이프라인 (Training Pipeline)

```
data/*.csv  또는  data_downsampled/X_downsampled.npy
        │
        ▼
load_kaggle_gearbox_data()         # 포맷 자동 감지
        │
        ▼
train_test_split (stratified)      # test 15%
        │
        ▼
train_test_split (stratified)      # val 18% of remaining
        │
        ▼
StandardScaler.fit_transform(X_tr) # train으로만 fit
StandardScaler.transform(X_val/X_test)
        │
        ▼
GearboxDataset → DataLoader
        │
        ▼
get_model(model_type, input_size, num_classes)
        │
        ▼
loop (epoch):
   train_epoch()  ── (옵션) AMP autocast + GradScaler
   validate_epoch()
   ReduceLROnPlateau.step(val_loss)
   if best val_acc:  save_checkpoint(best_model.pth)
   EarlyStopping(patience=15)
        │
        ▼
final test evaluation → confusion matrix, training history PNG
        │
        ▼
results/{YYYYMMDD_HHMMSS}/
   ├── best_model.pth        # state_dict + optimizer_state + scaler + label_names + val_acc
   ├── training_history.png
   └── confusion_matrix.png
```

**체크포인트 구조** (`utils.save_checkpoint`):
```python
{
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "epoch": int,
  "val_acc": float,
  "label_names": np.ndarray[str],
  "scaler": StandardScaler,
}
```

---

## 5. 추론 파이프라인 (Inference Pipeline)

```
[Browser]
   UploadCard: 파일 선택 (.csv 또는 .npy)
        │
        ▼
[Frontend api.predict(modelId, file)]
   FormData: model_id + file
        │
        ▼ POST /api/predict (multipart)
[FastAPI - predict.py]
   _read_upload_capped()        # 20MB 제한, 청크 읽기
   _parse_signal()              # dtype/shape 검증, 1D→2D reshape
        │
        ▼
[predictor_pool.get(model_id)]
   - 정규식 검증 (^\d{8}_\d{6}$)
   - path-traversal 차단 (Path.relative_to)
   - LRU 캐시 hit/miss → 필요 시 GearboxPredictor 로드
        │
        ▼
async with cached.lock:          # predictor 단위 직렬화
   await run_in_threadpool(
       predictor.predict_batch(signals)
   )
        │
        ▼
[GearboxPredictor.predict_batch]
   scaler.transform(signals)
   model(tensor) → logits
   softmax + argmax → PredictionItem 리스트
        │
        ▼
PredictResponse 반환
   + HISTORY.appendleft(HistoryEntry)   # in-memory deque, max 100
        │
        ▼
[Frontend]
   PredictionResult 렌더
   HistoryTable 갱신 (api.history)
```

---

## 6. 디렉터리 구조

```
gearbox-fault-diagnosis/
├── README.md
├── LICENSE
├── docs/
│   ├── ARCHITECTURE.md          ← 본 문서
│   └── UML.md                   ← 클래스/시퀀스 다이어그램
├── backend/
│   ├── requirements.txt
│   ├── config.py
│   ├── models.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── downsample_data.py
│   ├── simple_predict.py
│   ├── test_predict.py
│   ├── main.py
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── api/
│   │   │   ├── health.py
│   │   │   ├── models.py
│   │   │   └── predict.py
│   │   └── services/
│   │       └── predictor_pool.py
│   ├── data/                    # 원본 CSV (gitignore 권장)
│   ├── data_downsampled/        # 전처리 NPY
│   ├── results/                 # 학습 결과 (체크포인트 + 그림)
│   ├── evaluation_results/      # 평가 산출물
│   └── visualizations/          # 클래스별 샘플 시각화
└── frontend/
    ├── package.json
    ├── vite.config.ts           # /api → :8000 프록시
    ├── tsconfig.json
    ├── tailwind.config.js
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css
        ├── api/
        │   └── client.ts
        └── components/
            ├── ModelSelector.tsx
            ├── UploadCard.tsx
            ├── PredictionResult.tsx
            └── HistoryTable.tsx
```

---

## 7. 핵심 설계 패턴

| 패턴 | 적용 위치 | 목적 |
|------|----------|------|
| Factory | `models.get_model`, `config.get_config` | 모델/설정 객체 생성 일원화 |
| Strategy | 모델별 forward 구현(CNN/ResNet/MLP) | 동일 인터페이스(`nn.Module`)로 교체 가능 |
| Auto-detect | `evaluate.ModelEvaluator`, `inference.GearboxPredictor` | state_dict 키로 모델 종류 추론 |
| LRU Cache | `services.predictor_pool.PredictorPool` | 반복 `torch.load` 방지 |
| Async Lock | `CachedPredictor.lock` | 단일 모델에 대한 동시 추론 직렬화 |
| Defensive Parsing | `app.api.predict._parse_signal` | dtype/shape 검증, 사이즈 캡 |
| Checkpoint Bundle | `utils.save_checkpoint` | 모델 + scaler + label_names를 단일 `.pth`에 결속 |

---

## 8. 보안 / 운영 고려사항

- **경로 트래버설 차단**: `predictor_pool._resolve()`가 `Path.relative_to(results_dir)`로 검증.
- **모델 ID 정규식 화이트리스트**: `^\d{8}_\d{6}$` 만 허용.
- **업로드 사이즈 제한**: 20MB 청크 누적 검사.
- **CORS 화이트리스트**: localhost:5173 만 허용 (운영 시 도메인 추가 필요).
- **체크포인트 로딩**: `weights_only=False` — scaler 등 비텐서 객체 포함이 필요해서. 신뢰된 파일만 로드해야 함.
- **History 영속화 없음**: 프로세스 재시작 시 초기화. 영속화가 필요하면 SQLite/Redis 도입 고려.

---

## 9. 확장 포인트

1. **새 모델 추가**: `models.py`에 `nn.Module` 정의 → `get_model` 분기 추가 → `evaluate`/`inference`의 자동 감지 로직 갱신.
2. **새 입력 포맷**: `dataset.load_kaggle_gearbox_data` 분기 추가 또는 `app.api.predict._parse_signal` 확장.
3. **History 영속화**: `app.api.predict`의 `HISTORY` deque를 DB 백엔드로 교체.
4. **인증/권한**: 현재는 익명 — FastAPI dependency 로 JWT/API key 추가.
5. **모델 버전 관리**: 현재 디렉터리 타임스탬프(`YYYYMMDD_HHMMSS`)가 사실상 버전. MLflow 등 도입 검토.
