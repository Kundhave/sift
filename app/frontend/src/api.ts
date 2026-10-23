import { useCallback, useEffect, useState } from "react";
import type { Analysis, ModelInfo, ResultFile, SampleSummary } from "./types";

export class ApiFail extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try { r = await fetch(path, init); }
  catch { throw new ApiFail(0, "Cannot reach the Sift backend. Is it running on port 8000?"); }
  const ct = r.headers.get("content-type") ?? "";
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    if (ct.includes("json")) {
      const j = await r.json().catch(() => null);
      if (j?.error) msg = String(j.error);
      else if (j?.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail).slice(0, 300);
    } else if (r.status >= 500) msg = "Backend unreachable or crashed (" + r.status + ")";
    throw new ApiFail(r.status, msg);
  }
  if (!ct.includes("json")) throw new ApiFail(502, "Backend returned a non-JSON response");
  return r.json();
}

export const getModels = () => api<{ default: string; models: ModelInfo[] }>("/api/models");
export const getSamples = () => api<SampleSummary[]>("/api/samples");
export const getResultsIndex = () => api<ResultFile[]>("/api/results");
export const analyze = (body: object) =>
  api<Analysis>("/api/analyze", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

export interface Async<T> { data: T | null; error: string | null; loading: boolean; reload: () => void }

export function useApi<T>(path: string | null, deps: unknown[] = []): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!path);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!path) { setLoading(false); return; }
    let live = true;
    setLoading(true); setError(null);
    api<T>(path).then((d) => live && (setData(d), setLoading(false)))
      .catch((e: Error) => live && (setError(e.message), setData(null), setLoading(false)));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, tick, ...deps]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

/** All result files that exist right now. Pages poll this so they fill in automatically. */
export function useResultIndex(pollMs = 15000) {
  const [files, setFiles] = useState<ResultFile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    const load = () => getResultsIndex().then((f) => live && (setFiles(f), setError(null)))
      .catch((e: Error) => live && setError(e.message));
    load();
    const id = setInterval(load, pollMs);
    return () => { live = false; clearInterval(id); };
  }, [pollMs]);
  return { files, error, loading: files === null && error === null };
}

export function matchFiles(files: ResultFile[] | null, re: RegExp, ext?: string): ResultFile[] {
  return (files ?? []).filter((f) => re.test(f.path) && (!ext || f.path.toLowerCase().endsWith(ext)));
}
