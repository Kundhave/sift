export interface ModelInfo {
  id: string; name: string; kind: string; description: string; available: boolean; reason: string | null;
  loaded: boolean; encoder?: string; backend?: string; max_len?: number; temperature?: number;
  threshold?: number; size_mb?: number | null; supports_explain?: boolean; source_run?: string;
}
export interface Attributions {
  method: string; n_steps: number;
  intent_words: [string, number][]; hunk_lines: [string, number][];
}
export interface GroundTruth { label: string; drift_type: string | null; difficulty: string | null; noise_types: string | null }
export interface SimilarHunk {
  sample_id: string; label: string; similarity: number; repo?: string | null; file?: string | null;
  drift_type?: string | null; text?: string;
}
export interface HunkResult {
  index: number; file: string; header: string; p_misaligned: number; verdict: string;
  drift_type: string | null; drift_probs: Record<string, number> | number[] | null;
  confidence_head: number | null; text: string; rank: number; confidence: number;
  cluster_id: number | null; cluster_prior?: number | null; similar?: SimilarHunk[]; attributions: Attributions | null;
  ground_truth?: GroundTruth | null; correct?: boolean;
}
export interface Analysis {
  model: string; intent: string; turns: string[]; dropped_turns: { turn: string; reason: string }[];
  threshold: number; temperature: number; n_hunks: number;
  notes: { file: string; kind: string; detail: string }[];
  model_latency_ms: number; total_ms: number; ranking: number[]; hunks: HunkResult[]; sample_id?: string;
}
export interface SampleSummary {
  id: string; tag: string; repo: string; language: string; n_hunks: number; n_misaligned: number;
  turns: string[]; drift_types: string[];
}
export interface ResultFile { path: string; source: string; bytes: number; type: string; url: string }
export type Verdict = "aligned" | "misaligned";
export type Decision = "accept" | "flag" | "discuss";
