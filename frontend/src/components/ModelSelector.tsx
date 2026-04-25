import type { ModelMeta } from '../api/client';

type Props = {
  models: string[];
  value: string;
  meta: ModelMeta | null;
  onChange: (id: string) => void;
};

export function ModelSelector({ models, value, meta, onChange }: Props) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="mb-3 text-lg font-semibold text-slate-800">모델</h2>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm"
      >
        {models.length === 0 && <option value="">등록된 모델이 없습니다</option>}
        {models.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
      {meta && (
        <dl className="mt-4 grid grid-cols-2 gap-y-1 text-sm">
          <dt className="text-slate-500">타입</dt>
          <dd className="text-slate-800">{meta.model_type}</dd>
          <dt className="text-slate-500">입력 차원</dt>
          <dd className="text-slate-800">{meta.input_size}</dd>
          <dt className="text-slate-500">클래스</dt>
          <dd className="text-slate-800">{meta.label_names.join(', ')}</dd>
          {meta.val_acc !== null && (
            <>
              <dt className="text-slate-500">Val Acc</dt>
              <dd className="text-slate-800">{meta.val_acc.toFixed(2)}%</dd>
            </>
          )}
          {meta.epoch !== null && (
            <>
              <dt className="text-slate-500">Epoch</dt>
              <dd className="text-slate-800">{meta.epoch}</dd>
            </>
          )}
        </dl>
      )}
    </section>
  );
}
