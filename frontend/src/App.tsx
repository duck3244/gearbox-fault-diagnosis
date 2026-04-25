import { useEffect, useState } from 'react';
import {
  api,
  type Health,
  type HistoryEntry,
  type ModelMeta,
  type PredictResponse,
} from './api/client';
import { HistoryTable } from './components/HistoryTable';
import { ModelSelector } from './components/ModelSelector';
import { PredictionResult } from './components/PredictionResult';
import { UploadCard } from './components/UploadCard';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [modelId, setModelId] = useState<string>('');
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // 초기 로드
  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e)));
    api.listModels().then((ids) => {
      setModels(ids);
      if (ids[0]) setModelId(ids[0]);
    }).catch((e) => setError(String(e)));
    api.history(5).then(setHistory).catch(() => { /* no-op on first load */ });
  }, []);

  // 모델 변경 시 메타 조회
  useEffect(() => {
    if (!modelId) {
      setMeta(null);
      return;
    }
    api.modelMeta(modelId).then(setMeta).catch((e) => setError(String(e)));
  }, [modelId]);

  const handleSubmit = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.predict(modelId, file);
      setResult(r);
      const h = await api.history(5);
      setHistory(h);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <h1 className="text-xl font-semibold">Gearbox Fault Diagnosis</h1>
          {health && (
            <div className="flex items-center gap-2 text-sm">
              <span
                className={`h-2 w-2 rounded-full ${
                  health.cuda ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
              />
              <span className="text-slate-500">
                torch {health.torch_version} · {health.cuda ? 'CUDA' : 'CPU'}
              </span>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-4 px-6 py-6">
        {error && (
          <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-700">
            {error}
          </div>
        )}
        <ModelSelector
          models={models}
          value={modelId}
          meta={meta}
          onChange={setModelId}
        />
        <UploadCard disabled={!modelId || busy} onSubmit={handleSubmit} />
        <PredictionResult data={result} />
        <HistoryTable rows={history} />
      </main>
    </div>
  );
}
