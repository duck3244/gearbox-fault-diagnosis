import type { PredictResponse } from '../api/client';

export function PredictionResult({ data }: { data: PredictResponse | null }) {
  if (!data) return null;
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-800">예측 결과</h2>
        <span className="text-xs text-slate-500">
          {data.model_type} · {data.input_size}-dim · {data.elapsed_ms.toFixed(1)} ms
        </span>
      </div>
      <ul className="divide-y divide-slate-100">
        {data.predictions.map((p) => (
          <li key={p.sample_index} className="py-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-sm font-medium text-slate-700">
                Sample #{p.sample_index + 1}
              </span>
              <span className="text-sm text-slate-500">
                {p.predicted_class} · {(p.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="space-y-1">
              {Object.entries(p.probabilities).map(([name, prob]) => (
                <div key={name} className="flex items-center gap-2">
                  <span className="w-24 text-xs text-slate-500">{name}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full bg-indigo-500"
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <span className="w-12 text-right text-xs tabular-nums text-slate-500">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
