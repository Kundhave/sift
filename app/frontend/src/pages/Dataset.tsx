import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../api";
import type { SampleSummary } from "../types";
import { COLORS, axisProps, gridStroke, tooltipStyle } from "../components/charts";
import { DiffView } from "../components/DiffView";
import { Badge, Empty, ErrorBox, PageHeader, Section, Skeleton, SkeletonBlock, Stat } from "../components/ui";
import { Highlights } from "../components/Highlights";
import { driftName } from "../format";

type Split = { hunks: number; sessions: number; aligned_clean: number; misaligned_clean: number; drift: Record<string, number>; difficulty: Record<string, number>; label_flips: number; languages: Record<string, number>; repos: string[] };
interface Stats { train: Split; val: Split; test: Split; noise_rows: Record<string, number>; rows_with_any_noise: number; non_semantic_reasons: Record<string, number>; intent_noise_sessions: Record<string, number>; label_flip_directions: Record<string, number>; approx_tokens: Record<string, number>; hunks_per_session: Record<string, number>; commit_date_range: string[]; distinct_source_commits: number }
interface SampleFull { id: string; tag: string; repo: string; language: string; turns: string[]; hunks: { index: number; file: string; label: string; drift_type: string | null; difficulty: string | null; noise_types: string | null; text: string }[] }

const NOISE_LABEL: Record<string, string> = { intent_turn: "Intent-turn noise", duplicate_hunk: "Duplicate hunk", non_semantic_hunk: "Non-semantic hunk", label_flip: "Label flip (train only)", truncation: "Truncation (> 1,024 tokens)" };

export default function DatasetPage() {
  const st = useApi<Stats>("/api/dataset/stats");
  const sm = useApi<SampleSummary[]>("/api/samples");
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const detail = useApi<SampleFull>(open ? `/api/samples/${open}` : null);
  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase();
    return (sm.data ?? []).filter((s) => !t || [s.id, s.repo, s.tag, s.language, ...s.turns, ...s.drift_types].join(" ").toLowerCase().includes(t));
  }, [sm.data, q]);

  const d = st.data;
  const splits = d ? (["train", "val", "test"] as const).map((k) => ({ k, ...d[k] })) : [];
  const total = splits.reduce((a, s) => a + s.hunks, 0);

  return (
    <div className="space-y-8">
      <PageHeader title="Dataset explorer" sub="Sift-IA v1.0: hunk-intent pairs from open-source repositories. Splits are by repository, so no repository appears in two splits." />
      <Highlights page="dataset" />
      {st.error && <ErrorBox message={st.error} onRetry={st.reload} />}
      {st.loading && <SkeletonBlock rows={6} />}
      {d && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Hunk-intent pairs" value={total.toLocaleString()} />
            <Stat label="Sessions" value={splits.reduce((a, s) => a + s.sessions, 0).toLocaleString()} sub={`median ${d.hunks_per_session.median} hunks, max ${d.hunks_per_session.max}`} />
            <Stat label="Source commits" value={d.distinct_source_commits.toLocaleString()} sub={`${d.commit_date_range[0]} to ${d.commit_date_range[1]}`} />
            <Stat label="Tokens per pair" value={`${d.approx_tokens.median}`} sub={`median · p95 ${d.approx_tokens.p95} · max ${d.approx_tokens.max.toLocaleString()}`} />
          </div>

          <Section title="Splits" hint="Counts use clean labels.">
            <div className="card overflow-x-auto">
              <table className="w-full min-w-[640px] text-[12.5px] tabular-nums">
                <thead className="text-left text-muted"><tr className="border-b border-line">{["Split", "Repositories", "Sessions", "Hunks", "Aligned", "Misaligned", "Languages"].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr></thead>
                <tbody>{splits.map((s) => (
                  <tr key={s.k} className="border-b border-line/60 align-top last:border-0">
                    <td className="px-3 py-2 font-medium capitalize">{s.k === "val" ? "validation" : s.k}</td>
                    <td className="px-3 py-2">{s.repos.join(", ")}</td><td className="px-3 py-2">{s.sessions.toLocaleString()}</td><td className="px-3 py-2">{s.hunks.toLocaleString()}</td>
                    <td className="px-3 py-2">{s.aligned_clean.toLocaleString()}</td><td className="px-3 py-2">{s.misaligned_clean.toLocaleString()}</td>
                    <td className="px-3 py-2">{Object.entries(s.languages).map(([l, n]) => `${l} ${n.toLocaleString()}`).join(" · ")}</td>
                  </tr>))}</tbody>
              </table>
            </div>
          </Section>

          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Misaligned drift types and difficulty" hint="Clean labels, per split.">
              <div className="card p-3">
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={splits.map((s) => ({ split: s.k, ...Object.fromEntries(Object.entries(s.drift).map(([k, v]) => [driftName(k), v])), random: s.difficulty.random, hard: s.difficulty.hard }))} margin={{ left: -10 }}>
                    <CartesianGrid vertical={false} stroke={gridStroke} /><XAxis dataKey="split" {...axisProps} /><YAxis {...axisProps} />
                    <Tooltip {...tooltipStyle} /><Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Collateral" fill={COLORS[0]} /><Bar dataKey="Scope-Expansion" fill={COLORS[1]} /><Bar dataKey="Purpose-Drift" fill={COLORS[2]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[12.5px] text-muted">Difficulty tiers (misaligned only): {splits.map((s) => `${s.k} random ${s.difficulty.random.toLocaleString()} / hard ${s.difficulty.hard.toLocaleString()}`).join(" · ")}.</p>
            </Section>
            <Section title="Noise shipped with ground truth" hint={`${d.rows_with_any_noise.toLocaleString()} rows carry at least one noise type.`}>
              <div className="card p-3">
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart layout="vertical" data={Object.entries(d.noise_rows).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ name: NOISE_LABEL[k] ?? k, rows: v }))} margin={{ left: 20, right: 20 }}>
                    <CartesianGrid horizontal={false} stroke={gridStroke} /><XAxis type="number" {...axisProps} /><YAxis type="category" dataKey="name" width={150} {...axisProps} />
                    <Tooltip {...tooltipStyle} /><Bar dataKey="rows" radius={[0, 4, 4, 0]}>{Object.keys(d.noise_rows).map((_, i) => <Cell key={i} fill={COLORS[3]} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[12.5px] text-muted">
                Non-semantic: {Object.entries(d.non_semantic_reasons).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(", ")}. Intent-noise sessions: {Object.entries(d.intent_noise_sessions).filter(([k]) => k !== "none").map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(", ")}. Label flips: {Object.entries(d.label_flip_directions).map(([k, v]) => `${k} ${v}`).join(", ")}.
              </p>
            </Section>
          </div>
        </>
      )}

      <Section title="Browse curated test samples" hint="These are the examples the Review page can load. Labels come from dataset construction, not manual judgement."
        right={<input className="input !w-64" placeholder="Search id, repo, intent, drift..." aria-label="Search samples" value={q} onChange={(e) => setQ(e.target.value)} />}>
        {sm.error && <ErrorBox message={sm.error} onRetry={sm.reload} />}
        {sm.loading ? <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-14" />)}</div> : filtered.length === 0 ? <Empty title="No samples match">Try a different search term.</Empty> : (
          <ul className="space-y-2">
            {filtered.map((s) => (
              <li key={s.id} className="card">
                <button className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 p-3 text-left" onClick={() => setOpen(open === s.id ? null : s.id)} aria-expanded={open === s.id}>
                  <span className="font-mono text-[12.5px] font-semibold">{s.id}</span>
                  <Badge>{s.repo}</Badge><Badge>{s.language}</Badge><Badge tone="accent">{s.tag}</Badge>
                  <span className="text-[12px] text-muted">{s.n_hunks} hunks, {s.n_misaligned} misaligned{s.drift_types.length ? ` (${s.drift_types.map(driftName).join(", ")})` : ""}</span>
                  <span className="basis-full truncate text-[13px] text-muted">{s.turns.join("  |  ")}</span>
                </button>
                {open === s.id && (
                  <div className="space-y-3 border-t border-line p-3">
                    {detail.loading ? <SkeletonBlock rows={4} /> : detail.error ? <ErrorBox message={detail.error} /> : detail.data?.hunks.map((h) => (
                      <div key={h.index} className="space-y-1.5">
                        <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
                          <span className="font-mono font-medium">{h.file}</span>
                          <Badge tone={h.label === "misaligned" ? "bad" : "ok"}>{h.label}</Badge>
                          {h.drift_type && h.drift_type !== "none" && <Badge tone="accent">{driftName(h.drift_type)}</Badge>}
                          {h.difficulty && h.difficulty !== "none" && <Badge>{h.difficulty}</Badge>}
                          {h.noise_types && h.noise_types !== "none" && <Badge tone="warn">noise: {h.noise_types}</Badge>}
                        </div>
                        <DiffView text={h.text} file={h.file} maxLines={30} />
                      </div>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </Section>
    </div>
  );
}
