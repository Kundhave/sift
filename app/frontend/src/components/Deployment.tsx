import { useApi } from "../api";
import { Caption, ErrorBox, SkeletonBlock } from "./ui";
import { f3 } from "../format";

type Row = { id: string; size: number | null; f1?: number; aucR?: number; aucH?: number; ndcg?: number; ece?: number; p50?: number; p95?: number; thr?: number };

/** Size and CPU latency of the exported students, straight from results/latency_size_onnx.json. */
export function Deployment({ path = "latency_size_onnx.json" }: { path?: string }) {
  const { data, error, loading, reload } = useApi<any>(`/api/results/${path}`);
  if (loading) return <SkeletonBlock rows={3} />;
  if (error) return <ErrorBox message={error} onRetry={reload} />;
  const rows: Row[] = Object.entries<any>(data ?? {}).filter(([, v]) => v && typeof v === "object" && "size_mb" in v).map(([id, v]) => ({
    id, size: v.size_mb ?? null, thr: v.threshold, f1: v.test?.macro_f1, aucR: v.test?.pairwise_auc?.random?.auc, aucH: v.test?.pairwise_auc?.hard?.auc,
    ndcg: v.test?.["ndcg@5"], ece: v.test?.ece, p50: v.latency_ms?.p50_ms, p95: v.latency_ms?.p95_ms,
  }));
  const t = data?.targets ?? {}, met = data?.targets_met ?? {};
  const ms = (x?: number) => (x == null ? "-" : x.toFixed(1));
  return (
    <div className="space-y-2">
      <div className="card overflow-x-auto" data-testid="deployment">
        <table className="w-full min-w-[640px] text-[12.5px] tabular-nums">
          <thead className="text-left text-muted"><tr className="border-b border-line">
            {["Export", "Size MB", "Macro-F1", "AUC random", "AUC hard", "NDCG@5", "ECE", "p50 ms", "p95 ms"].map((h, i) => <th key={h} className={"px-3 py-2 font-medium " + (i ? "text-right" : "")}>{h}</th>)}
          </tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-line/60 last:border-0">
                <td className="px-3 py-2 font-medium">{r.id.replace(/^student_/, "").replace(/_/g, " ")}</td>
                <td className="px-3 py-2 text-right">{r.size?.toFixed(1) ?? "-"}</td><td className="px-3 py-2 text-right">{f3(r.f1)}</td>
                <td className="px-3 py-2 text-right">{f3(r.aucR)}</td><td className="px-3 py-2 text-right">{f3(r.aucH)}</td>
                <td className="px-3 py-2 text-right">{f3(r.ndcg)}</td><td className="px-3 py-2 text-right">{f3(r.ece)}</td>
                <td className="px-3 py-2 text-right">{ms(r.p50)}</td><td className="px-3 py-2 text-right">{ms(r.p95)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Caption>
        {data?.cpu?.cpu ? `Measured on ${data.cpu.cpu}, ${data.threads} threads. ` : ""}
        {t.size_mb_lt != null && <>Targets (planned): under {t.size_mb_lt} MB{met.size != null ? ` (${met.size ? "met" : "not met"})` : ""}, p95 under {t.p95_ms_lt} ms{met.p95 != null ? ` (${met.p95 ? "met" : "not met"})` : ""}. </>}
        {data?.note}
      </Caption>
    </div>
  );
}
