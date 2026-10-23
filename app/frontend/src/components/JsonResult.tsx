import { useApi } from "../api";
import { DataTree } from "./DataTree";
import { ErrorBox, SkeletonBlock } from "./ui";

export function useResultJson<T = unknown>(path: string | null) {
  return useApi<T>(path ? `/api/results/${path}` : null);
}

export function JsonResult({ path, title }: { path: string; title?: string }) {
  const { data, error, loading, reload } = useResultJson(path);
  return (
    <div className="card p-4">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-[13px] font-semibold">{title ?? path}</div>
        {title && title !== path ? <code className="font-mono text-[11px] text-muted">{path}</code> : null}
      </div>
      {loading ? <SkeletonBlock rows={4} /> : error ? <ErrorBox message={error} onRetry={reload} /> : <DataTree value={data} />}
    </div>
  );
}
