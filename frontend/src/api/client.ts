export type Health = {
  status: 'ok';
  cuda: boolean;
  torch_version: string;
};

export type ModelType = 'CNN' | 'ResNet' | 'MLP';

export type ModelMeta = {
  model_id: string;
  model_type: ModelType;
  input_size: number;
  num_classes: number;
  label_names: string[];
  val_acc: number | null;
  epoch: number | null;
};

export type PredictionItem = {
  sample_index: number;
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  probabilities: Record<string, number>;
};

export type PredictResponse = {
  model_id: string;
  model_type: string;
  input_size: number;
  n_samples: number;
  predictions: PredictionItem[];
  elapsed_ms: number;
};

export type HistoryEntry = {
  timestamp: string;
  model_id: string;
  n_samples: number;
  top_class: string;
  elapsed_ms: number;
};

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? res.statusText);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => fetch('/api/health').then((r) => handle<Health>(r)),
  listModels: () => fetch('/api/models').then((r) => handle<string[]>(r)),
  modelMeta: (id: string) =>
    fetch(`/api/models/${encodeURIComponent(id)}/meta`).then((r) => handle<ModelMeta>(r)),
  predict: (modelId: string, file: File) => {
    const fd = new FormData();
    fd.append('model_id', modelId);
    fd.append('file', file);
    return fetch('/api/predict', { method: 'POST', body: fd }).then((r) =>
      handle<PredictResponse>(r),
    );
  },
  history: (limit = 20) =>
    fetch(`/api/history?limit=${limit}`).then((r) => handle<HistoryEntry[]>(r)),
};
