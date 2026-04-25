import type { HistoryEntry } from '../api/client';

export function HistoryTable({ rows }: { rows: HistoryEntry[] }) {
  if (rows.length === 0) return null;
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="mb-3 text-lg font-semibold text-slate-800">최근 예측</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-slate-500">
            <th className="pb-2 font-normal">시각 (UTC)</th>
            <th className="pb-2 font-normal">모델</th>
            <th className="pb-2 font-normal">샘플</th>
            <th className="pb-2 font-normal">다수 클래스</th>
            <th className="pb-2 text-right font-normal">소요</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-slate-100 last:border-none">
              <td className="py-2 text-slate-600">{r.timestamp.slice(11, 19)}</td>
              <td className="py-2 text-slate-600">{r.model_id}</td>
              <td className="py-2 text-slate-600">{r.n_samples}</td>
              <td className="py-2 text-slate-600">{r.top_class}</td>
              <td className="py-2 text-right tabular-nums text-slate-600">
                {r.elapsed_ms.toFixed(0)} ms
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
