import { useState } from 'react';

type Props = {
  disabled?: boolean;
  onSubmit: (file: File) => void;
};

export function UploadCard({ disabled, onSubmit }: Props) {
  const [file, setFile] = useState<File | null>(null);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="mb-3 text-lg font-semibold text-slate-800">신호 업로드</h2>
      <p className="mb-4 text-sm text-slate-500">
        <code className="rounded bg-slate-100 px-1">.csv</code> 또는{' '}
        <code className="rounded bg-slate-100 px-1">.npy</code> (최대 20MB). 1D 또는 2D
        배열이며 특징 차원은 모델과 일치해야 합니다.
      </p>
      <div className="flex items-center gap-3">
        <input
          type="file"
          accept=".csv,.npy"
          disabled={disabled}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-sm file:mr-3 file:rounded-md file:border file:border-slate-300 file:bg-slate-50 file:px-3 file:py-1.5 file:text-sm file:text-slate-700 hover:file:bg-slate-100"
        />
        <button
          type="button"
          disabled={!file || disabled}
          onClick={() => file && onSubmit(file)}
          className="rounded-md bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          예측 실행
        </button>
      </div>
    </section>
  );
}
