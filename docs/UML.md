# Gearbox Fault Diagnosis - UML Diagrams

본 문서는 시스템의 주요 클래스, 시퀀스, 상태 흐름을 Mermaid 기반 UML로 정리한다.
GitHub/IDE의 Mermaid 렌더러에서 그대로 미리보기 가능하다.

---

## 1. 클래스 다이어그램 — ML Core (`backend/`)

### 1.1 모델 (`models.py`)

```mermaid
classDiagram
    class Module {
        <<torch.nn.Module>>
        +forward(x)
    }

    class GearboxCNN1D {
        -conv_block1 : Sequential
        -conv_block2 : Sequential
        -conv_block3 : Sequential
        -conv_block4 : Sequential
        -classifier  : Sequential
        +__init__(input_size, num_classes)
        +forward(x) Tensor
    }

    class ResidualBlock {
        -conv1 : Conv1d
        -bn1   : BatchNorm1d
        -conv2 : Conv1d
        -bn2   : BatchNorm1d
        -shortcut : Sequential
        +__init__(in_ch, out_ch, stride)
        +forward(x) Tensor
    }

    class GearboxResNet1D {
        -stem : Sequential
        -residual_blocks : ModuleList
        -adaptive_pool : AdaptiveAvgPool1d
        -classifier : Linear
        +__init__(input_size, num_classes)
        +forward(x) Tensor
    }

    class GearboxMLP {
        -layers : Sequential
        +__init__(input_size, num_classes)
        +forward(x) Tensor
    }

    Module <|-- GearboxCNN1D
    Module <|-- ResidualBlock
    Module <|-- GearboxResNet1D
    Module <|-- GearboxMLP
    GearboxResNet1D o-- ResidualBlock : composes
```

> **Factory**: `get_model(model_name: str, input_size: int, num_classes: int) -> nn.Module`
> — `model_name` 값에 따라 위 세 클래스 중 하나를 반환.

---

### 1.2 데이터 / 전처리 (`dataset.py`, `preprocessing.py`)

```mermaid
classDiagram
    class Dataset {
        <<torch.utils.data.Dataset>>
        +__len__()
        +__getitem__(idx)
    }

    class GearboxDataset {
        -features : Tensor
        -labels   : Tensor
        +__init__(X, y)
        +__len__() int
        +__getitem__(idx) Tuple
    }

    class SignalPreprocessor {
        -sampling_rate : int
        +remove_dc_offset(signal) ndarray
        +bandpass_filter(signal, low, high, order) ndarray
        +normalize(signal, method) ndarray
        +denoise(signal, wavelet, level) ndarray
        +segment_signal(signal, win, stride) ndarray
        +preprocess(signal, **opts) ndarray
    }

    class FeatureExtractor {
        -sampling_rate : int
        +time_domain_features(signal) dict
        +frequency_domain_features(signal) dict
        +statistical_features(signal) dict
        +extract_all_features(signal) dict
        +extract_features_dataframe(signals) DataFrame
    }

    Dataset <|-- GearboxDataset

    class load_kaggle_gearbox_data {
        <<function>>
        +load(data_path) Tuple~X, y, label_names~
    }

    class create_dataloaders {
        <<function>>
        +create(X_tr, y_tr, X_val, y_val, X_te, y_te, batch_size) DataLoader×3
    }

    GearboxDataset ..> create_dataloaders : used by
    load_kaggle_gearbox_data ..> GearboxDataset : feeds
```

---

### 1.3 학습 / 평가 / 추론 (`utils.py`, `evaluate.py`, `inference.py`)

```mermaid
classDiagram
    class EarlyStopping {
        -patience : int
        -best_score : float
        -counter : int
        -early_stop : bool
        +__init__(patience, mode)
        +__call__(score) bool
    }

    class ModelEvaluator {
        -model_path : str
        -device : torch.device
        -model : nn.Module
        -scaler : StandardScaler
        -label_names : ndarray
        +__init__(model_path, device)
        -_load_checkpoint()
        -_detect_model_type(state_dict) str
        +evaluate(test_loader) dict
        +plot_confusion_matrix()
        +plot_roc_curves()
        +plot_pr_curves()
        +generate_report()
    }

    class GearboxPredictor {
        -model_path : str
        -device : torch.device
        -model : nn.Module
        -scaler : StandardScaler
        -label_names : ndarray
        +model_type : str
        +input_size : int
        +num_classes : int
        +val_acc : float
        +epoch : int
        +__init__(model_path, device)
        +predict_single(signal) dict
        +predict_batch(signals) list~dict~
        +predict_from_csv(csv_path) list~dict~
        +predict_from_npy(npy_path) list~dict~
    }

    class TrainUtils {
        <<module utils.py>>
        +train_epoch(model, loader, criterion, optimizer, device, amp) Tuple
        +validate_epoch(model, loader, criterion, device) Tuple
        +save_checkpoint(state, path)
        +load_checkpoint(path, device) dict
        +plot_training_history(history, path)
        +plot_confusion_matrix(cm, labels, path)
        +set_seed(seed, deterministic)
        +get_device() device
        +create_output_directory(base) Path
    }

    ModelEvaluator ..> TrainUtils : uses load_checkpoint
    GearboxPredictor ..> TrainUtils : uses load_checkpoint
    ModelEvaluator --> "1" StandardScaler
    GearboxPredictor --> "1" StandardScaler
```

---

### 1.4 설정 (`config.py`)

```mermaid
classDiagram
    class Config {
        +PROJECT_ROOT : Path
        +DATA_DIR : Path
        +RESULTS_DIR : Path
        +TEST_SIZE : float
        +VAL_SIZE  : float
        +RANDOM_SEED : int
        +BATCH_SIZE : int
        +NUM_WORKERS : int
    }

    class CNNConfig {
        +EPOCHS : int
        +LEARNING_RATE : float
        +PATIENCE : int
    }

    class ResNetConfig {
        +EPOCHS : int
        +LEARNING_RATE : float
        +PATIENCE : int
    }

    class MLPConfig {
        +EPOCHS : int
        +LEARNING_RATE : float
        +PATIENCE : int
    }

    Config <|-- CNNConfig
    Config <|-- ResNetConfig
    Config <|-- MLPConfig
```

> **Factory**: `get_config(model_type: str) -> type[Config]`

---

## 2. 클래스 다이어그램 — FastAPI Layer (`backend/app/`)

```mermaid
classDiagram
    class FastAPI {
        <<framework>>
    }

    class App {
        <<app/main.py>>
        +app : FastAPI
        +CORS middleware
        +include_router(health, models, predict)
    }

    class HealthRouter {
        <<app/api/health.py>>
        +get_health() HealthResponse
    }

    class ModelsRouter {
        <<app/api/models.py>>
        +list_models() list~str~
        +get_meta(model_id) ModelMeta
    }

    class PredictRouter {
        <<app/api/predict.py>>
        -HISTORY : deque~HistoryEntry~
        +predict(model_id, file) PredictResponse
        +get_history(limit) list~HistoryEntry~
        -_read_upload_capped(file)
        -_parse_signal(bytes, ext) ndarray
    }

    class PredictorPool {
        <<services/predictor_pool.py>>
        -results_dir : Path
        -max_size : int
        -_cache : dict~str, CachedPredictor~
        -_order : list~str~
        -_registry_lock : asyncio.Lock
        +list_models() list~str~
        +get(model_id) CachedPredictor
        -_resolve(model_id) Path
        -_evict_if_needed()
    }

    class CachedPredictor {
        <<dataclass>>
        +predictor : GearboxPredictor
        +lock : asyncio.Lock
    }

    class HealthResponse {
        <<pydantic>>
        +status : str
        +cuda : bool
        +torch_version : str
    }

    class ModelMeta {
        <<pydantic>>
        +model_id : str
        +model_type : str
        +input_size : int
        +num_classes : int
        +label_names : list~str~
        +val_acc : float?
        +epoch : int?
    }

    class PredictionItem {
        <<pydantic>>
        +sample_index : int
        +predicted_class : str
        +predicted_index : int
        +confidence : float
        +probabilities : dict~str, float~
    }

    class PredictResponse {
        <<pydantic>>
        +model_id : str
        +model_type : str
        +input_size : int
        +n_samples : int
        +predictions : list~PredictionItem~
        +elapsed_ms : float
    }

    class HistoryEntry {
        <<pydantic>>
        +timestamp : str
        +model_id : str
        +n_samples : int
        +top_class : str
        +elapsed_ms : float
    }

    FastAPI <|-- App
    App o-- HealthRouter
    App o-- ModelsRouter
    App o-- PredictRouter
    ModelsRouter ..> PredictorPool : depends on
    PredictRouter ..> PredictorPool : depends on
    PredictorPool o-- "0..*" CachedPredictor
    CachedPredictor --> GearboxPredictor : wraps
    HealthRouter ..> HealthResponse : returns
    ModelsRouter ..> ModelMeta : returns
    PredictRouter ..> PredictResponse : returns
    PredictRouter ..> HistoryEntry : appends
    PredictResponse o-- PredictionItem
```

---

## 3. 컴포넌트 다이어그램 — Frontend (`frontend/src/`)

```mermaid
classDiagram
    class App {
        <<App.tsx>>
        +health : Health?
        +models : string[]
        +modelId : string?
        +meta : ModelMeta?
        +result : PredictResponse?
        +history : HistoryEntry[]
        +error : string?
        +busy : boolean
        +handleSubmit(file)
    }

    class ApiClient {
        <<api/client.ts>>
        +health() Health
        +listModels() string[]
        +modelMeta(id) ModelMeta
        +predict(modelId, file) PredictResponse
        +history(limit) HistoryEntry[]
    }

    class ModelSelector {
        <<component>>
        +models : string[]
        +modelId : string?
        +meta : ModelMeta?
        +onSelect(id)
    }

    class UploadCard {
        <<component>>
        +busy : boolean
        +onSubmit(file)
    }

    class PredictionResult {
        <<component>>
        +result : PredictResponse?
    }

    class HistoryTable {
        <<component>>
        +rows : HistoryEntry[]
    }

    App o-- ModelSelector
    App o-- UploadCard
    App o-- PredictionResult
    App o-- HistoryTable
    App ..> ApiClient : uses
```

---

## 4. 시퀀스 다이어그램 — 학습 (Training)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as train.py
    participant DS  as dataset.py
    participant SK  as StandardScaler
    participant DL  as DataLoader
    participant M   as get_model()
    participant U   as utils.py
    participant FS  as FileSystem

    User->>CLI: python train.py --model CNN ...
    CLI->>DS: load_kaggle_gearbox_data(path)
    DS-->>CLI: (X, y, label_names)
    CLI->>CLI: train_test_split (test 15%)
    CLI->>CLI: train_test_split (val 18%)
    CLI->>SK: fit_transform(X_train)
    SK-->>CLI: X_train scaled
    CLI->>SK: transform(X_val), transform(X_test)
    CLI->>DS: create_dataloaders(...)
    DS-->>CLI: (train_dl, val_dl, test_dl)
    CLI->>M: get_model(name, in_size, n_classes)
    M-->>CLI: nn.Module instance

    loop epoch in range(EPOCHS)
        CLI->>U: train_epoch(model, train_dl, ...)
        U-->>CLI: train_loss, train_acc
        CLI->>U: validate_epoch(model, val_dl, ...)
        U-->>CLI: val_loss, val_acc
        CLI->>CLI: scheduler.step(val_loss)
        alt val_acc improved
            CLI->>U: save_checkpoint(state, best_model.pth)
            U->>FS: torch.save(state)
        end
        CLI->>U: EarlyStopping(val_acc)
        break early stop triggered
            CLI-->>CLI: exit training loop
        end
    end

    CLI->>U: validate_epoch(model, test_dl, ...)
    U-->>CLI: test metrics
    CLI->>FS: save training_history.png, confusion_matrix.png
```

---

## 5. 시퀀스 다이어그램 — 추론 (Web Inference)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as Frontend (App.tsx)
    participant API as POST /api/predict
    participant Pool as PredictorPool
    participant FS as FileSystem
    participant Pred as GearboxPredictor
    participant Model as nn.Module
    participant Hist as HISTORY deque

    User->>FE: 모델 선택 + 파일 업로드 클릭
    FE->>API: multipart (model_id, file)
    API->>API: _read_upload_capped (≤20MB)
    API->>API: _parse_signal (dtype/shape)
    API->>Pool: get(model_id)

    alt cache hit
        Pool-->>API: CachedPredictor
    else cache miss
        Pool->>Pool: _resolve(model_id) (정규식 + 경로검증)
        Pool->>FS: torch.load(best_model.pth)
        FS-->>Pool: checkpoint dict
        Pool->>Pred: GearboxPredictor(path)
        Pred->>Model: get_model() + load_state_dict
        Pool->>Pool: _evict_if_needed (LRU)
        Pool-->>API: CachedPredictor
    end

    API->>API: shape[1] vs predictor.input_size 검증
    API->>Pred: async with lock: run_in_threadpool(predict_batch)
    Pred->>Pred: scaler.transform(signals)
    Pred->>Model: forward(tensor)
    Model-->>Pred: logits
    Pred->>Pred: softmax + argmax
    Pred-->>API: List[PredictionItem]
    API->>Hist: appendleft(HistoryEntry)
    API-->>FE: PredictResponse (json)
    FE->>FE: setResult / setHistory
    FE-->>User: 예측 결과 렌더링
```

---

## 6. 시퀀스 다이어그램 — 모델 메타 조회

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as Frontend
    participant API as /api/models...
    participant Pool as PredictorPool
    participant FS as FileSystem

    User->>FE: 페이지 로드
    FE->>API: GET /api/models
    API->>Pool: list_models()
    Pool->>FS: scan results/*/best_model.pth
    FS-->>Pool: dirs
    Pool-->>API: [model_id, ...]  (newest-first)
    API-->>FE: 200 OK, list~str~

    User->>FE: 모델 선택
    FE->>API: GET /api/models/{id}/meta
    API->>Pool: get(id)
    Pool-->>API: CachedPredictor (load if miss)
    API-->>FE: ModelMeta (model_type, input_size, label_names, val_acc, epoch)
```

---

## 7. 상태 다이어그램 — `EarlyStopping`

```mermaid
stateDiagram-v2
    [*] --> Watching
    Watching --> Watching : score improved\n(counter=0, best=score)
    Watching --> Patience : score not improved\n(counter+=1)
    Patience --> Watching : score improved
    Patience --> Patience : score not improved\n(counter < patience)
    Patience --> Stopped : counter >= patience
    Stopped --> [*]
```

---

## 8. 상태 다이어그램 — `PredictorPool` 캐시 엔트리

```mermaid
stateDiagram-v2
    [*] --> Absent
    Absent --> Loading : get(id) cache miss
    Loading --> Cached : torch.load 성공
    Loading --> Failed : 검증/IO 오류
    Failed --> [*]
    Cached --> Cached : get(id) cache hit\n(_order 갱신)
    Cached --> Evicted : LRU eviction\n(cache > max_size)
    Evicted --> [*]
```

---

## 9. 배포/런타임 컴포넌트 (Deployment)

```mermaid
flowchart LR
    subgraph Browser
        UI[React SPA<br/>localhost:5173]
    end

    subgraph DevHost
        Vite[Vite Dev Server<br/>:5173<br/>proxy /api → :8000]
        Uvicorn[Uvicorn<br/>:8000<br/>app.main:app]
        Models[(results/<br/>YYYYMMDD_HHMMSS/<br/>best_model.pth)]
        Data[(data_downsampled/<br/>X.npy, y.npy)]
    end

    UI -->|HTTP| Vite
    Vite -->|proxy| Uvicorn
    Uvicorn -->|read| Models
    Uvicorn -. optional .-> Data

    subgraph Trainer
        TrainCLI[train.py CLI]
    end

    TrainCLI -->|read| Data
    TrainCLI -->|write| Models
```

---

## 10. 인터페이스 요약 (REST API)

| Method | Path | Request | Response | 비고 |
|--------|------|---------|----------|------|
| GET | `/api/health` | — | `HealthResponse` | CUDA / torch 버전 |
| GET | `/api/models` | — | `list[str]` | 최신순 model_id |
| GET | `/api/models/{model_id}/meta` | path param | `ModelMeta` | 모델 상세 |
| POST | `/api/predict` | multipart: `model_id`, `file` (.csv/.npy ≤20MB) | `PredictResponse` | per-model lock + threadpool |
| GET | `/api/history?limit=N` | query | `list[HistoryEntry]` | in-memory deque (max 100) |

---

## 11. 다이어그램 렌더링 안내

- 본 문서의 모든 다이어그램은 [Mermaid](https://mermaid.js.org/) 문법으로 작성되어 있다.
- GitHub, GitLab, IntelliJ/PyCharm 최신 버전, VS Code(Markdown Preview Mermaid Support 확장)에서 자동 렌더링된다.
- 별도 PNG/SVG 추출이 필요하면 `mermaid-cli`(`mmdc`)를 사용:
  ```bash
  npx -y @mermaid-js/mermaid-cli -i docs/UML.md -o docs/uml.svg
  ```
