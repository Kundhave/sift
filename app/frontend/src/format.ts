export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const f3 = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? "-" : x.toFixed(3));
export function fmtVal(v: unknown): string {
  if (v == null) return "-";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString("en-US");
    const a = Math.abs(v);
    return a !== 0 && (a < 0.001 || a >= 10000) ? v.toExponential(2) : v.toFixed(a >= 100 ? 1 : 4);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}
export const label = (k: string) => k.replace(/_/g, " ").replace(/@/g, "@");
export const DRIFT_LABEL: Record<string, string> = {
  collateral: "Collateral", scope_expansion: "Scope-Expansion", purpose_drift: "Purpose-Drift",
};
export const driftName = (d: string | null | undefined) => (d ? DRIFT_LABEL[d] ?? d : "");
export const REASON_LABEL: Record<string, string> = {
  chit_chat: "Chit-chat", retraction: "Retracted request", repeated_turn: "Repeated turn",
  retracted_by_later_turn: "Retracted by a later turn",
};
