import { useEffect, useState } from "react";
import { api } from "../api";

export const COLORS = ["#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#64748b", "#22c55e", "#0ea5e9"];
export const axisProps = { stroke: "rgb(var(--muted))", fontSize: 11, tickLine: false } as const;
export const gridStroke = "rgb(var(--line))";
export const tooltipStyle = {
  contentStyle: { background: "rgb(var(--surface))", border: "1px solid rgb(var(--line))", borderRadius: 8, fontSize: 12, color: "rgb(var(--fg))" },
  labelStyle: { color: "rgb(var(--fg))", fontWeight: 600 }, itemStyle: { color: "rgb(var(--fg))" }, cursor: { fill: "rgb(var(--raised))" },
};

/** Fetch many result JSON files at once (keyed by published path). Re-fetches when the path list changes. */
export function useMultiJson(paths: string[]) {
  const key = paths.join("|");
  const [data, setData] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(paths.length > 0);
  useEffect(() => {
    let live = true;
    if (!paths.length) { setData({}); setLoading(false); return; }
    setLoading(true);
    Promise.all(paths.map((p) => api<any>(`/api/results/${p}`).then((d) => [p, d, null] as const).catch((e: Error) => [p, null, e.message] as const)))
      .then((rs) => {
        if (!live) return;
        setData(Object.fromEntries(rs.filter((r) => r[1] != null).map((r) => [r[0], r[1]])));
        setErrors(Object.fromEntries(rs.filter((r) => r[2]).map((r) => [r[0], r[2]!])));
        setLoading(false);
      });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return { data, errors, loading };
}

export interface ModelRow {
  id: string; name: string; group: "Main" | "Student" | "Ablation" | "Baseline" | "Non-neural" | "GAN";
  m: Record<string, number | null>;   // metrics for the chosen subset
  aucRandom: number | null; aucHard: number | null; ndcg5: number | null;
  full: Record<string, any> | null;   // full test block (for confusion matrix)
}

const num = (v: unknown): number | null => (typeof v === "number" && isFinite(v) ? v : null);

export function groupOf(name: string): ModelRow["group"] {
  if (/^full$/.test(name)) return "Main";
  if (/abl|nodenoise|no_|hunk_only|intent_only|nodistill/i.test(name)) return "Ablation";
  if (/student|onnx|int8/i.test(name)) return "Student";
  return "Baseline";
}

/** Build comparison rows from eval_*.json (subset = key in test{}), baselines.json and gan.json. Only real file content. */
const RUN_NAMES: Record<string, string> = {
  full: "Main: ModernBERT-base", abl_hunk_only: "Ablation: hunk-only", abl_intent_only: "Ablation: intent-only",
  abl_no_consolidation: "Ablation: no consolidation", abl_no_focal: "Ablation: no focal loss", abl_no_multitask: "Ablation: alignment head only",
  abl_no_denoise: "Ablation: no denoising", student: "Distilled MiniLM student", student_nodistill: "MiniLM student, no distillation",
  minilm_ce: "MiniLM-L6 cross-encoder", codebert: "CodeBERT cross-encoder", bilstm: "Siamese BiLSTM",
};
export function buildRows(evals: Record<string, any>, baselines: any | null, gan: any | null, subset: string): ModelRow[] {
  const rows: ModelRow[] = [];
  for (const [path, d] of Object.entries(evals)) {
    const t = d?.test?.[subset]; const tf = d?.test?.full;
    if (!tf) continue;
    const run = path.replace(/^eval_|\.json$/g, "");
    const name = String(d.name ?? RUN_NAMES[run] ?? run);
    const src = t ?? null;
    const pa = (b: any, tier: string) => num(b?.pairwise_auc?.[tier]?.auc ?? b?.pairwise_auc?.[tier]);
    rows.push({
      id: path, name, group: groupOf(run), full: tf,
      m: src ? { accuracy: num(src.accuracy), macro_f1: num(src.macro_f1), precision: num(src.precision), recall: num(src.recall), specificity: num(src.specificity), roc_auc: num(src.roc_auc), pr_auc: num(src.pr_auc), ece: num(src.ece), mrr: num(src.mrr) } : {},
      aucRandom: pa(d.test.tier_random ?? tf, "random") ?? pa(tf, "random"), aucHard: pa(d.test.tier_hard ?? tf, "hard") ?? pa(tf, "hard"),
      ndcg5: num(src?.["ndcg@5"]),
    });
  }
  if (baselines?.ndcg5 || baselines?.["ndcg@5"]) {
    const nd = baselines["ndcg@5"] ?? baselines.ndcg5;
    for (const [name, v] of Object.entries<number>(nd)) {
      const pw = baselines.pairwise_auc?.[name];
      const isB5 = name.startsWith("B5") && baselines.B5_classification_test;
      const c = isB5 ? baselines.B5_classification_test : null;
      rows.push({
        id: "base:" + name, name, group: "Non-neural", full: null,
        m: subset === "full" && c ? { accuracy: num(c.accuracy), macro_f1: num(c.macro_f1), precision: num(c.precision), recall: num(c.recall), specificity: num(c.specificity) } : {},
        aucRandom: num(pw?.random), aucHard: num(pw?.hard), ndcg5: num(v),
      });
    }
  }
  if (gan?.variants) {
    for (const [name, v] of Object.entries<any>(gan.variants)) {
      const t = v.test; if (!t) continue;
      rows.push({
        id: "gan:" + name, name: "GAN: " + name.replace(/_/g, " "), group: "GAN", full: null,
        m: subset === "full" ? { accuracy: num(t.accuracy), macro_f1: num(t.macro_f1), precision: num(t.precision), recall: num(t.recall), specificity: num(t.specificity), ece: num(t.ece) } : {},
        aucRandom: num(t.pairwise_auc?.random?.auc), aucHard: num(t.pairwise_auc?.hard?.auc), ndcg5: num(t["ndcg@5"]),
      });
    }
  }
  return rows;
}
