import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { analyze, api, getModels, getSamples, ApiFail } from "../api";
import type { Analysis, Decision, HunkResult, ModelInfo, SampleSummary } from "../types";
import { Badge, Empty, ErrorBox, Icon, Skeleton, SkeletonBlock } from "../components/ui";
import { DiffView } from "../components/DiffView";
import { REASON_LABEL, driftName, pct } from "../format";

const EXAMPLE_TURNS = "Fix the off-by-one error in pagination";
const DEC_LABEL: Record<Decision, string> = { accept: "Accepted", flag: "Flagged", discuss: "Needs discussion" };


function attrColor(s: number, max: number): string | undefined {
  const a = Math.min(0.6, (Math.abs(s) / (max || 1)) * 0.6);
  if (a < 0.05) return undefined;
  return s >= 0 ? `rgb(var(--bad) / ${a})` : `rgb(var(--accent) / ${a})`;
}

const INPUTS_KEY = "sift-inputs-open";
const DESK = "(min-width: 1024px)";

function useDesktop() {
  const [desk, setDesk] = useState(() => window.matchMedia(DESK).matches);
  useEffect(() => {
    const mq = window.matchMedia(DESK);
    const on = () => setDesk(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return desk;
}

export default function ReviewPage() {
  const desk = useDesktop();
  // Desktop state is remembered; the mobile section is open before an analysis and collapses after one.
  const [openDesk, setOpenDesk] = useState(() => {
    try { return localStorage.getItem(INPUTS_KEY) !== "0"; } catch { return true; }
  });
  const [openMob, setOpenMob] = useState(true);
  const inputsOpen = desk ? openDesk : openMob;
  const toggleInputs = useCallback(() => {
    if (window.matchMedia(DESK).matches) {
      setOpenDesk((o) => {
        try { localStorage.setItem(INPUTS_KEY, o ? "0" : "1"); } catch { /* storage blocked */ }
        return !o;
      });
    } else setOpenMob((o) => !o);
  }, []);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelErr, setModelErr] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleSummary[] | null>(null);
  const [model, setModel] = useState("full");
  const [turnsText, setTurnsText] = useState(EXAMPLE_TURNS);
  const [diffText, setDiffText] = useState("");
  const [sampleId, setSampleId] = useState<string>("");
  const [thr, setThr] = useState<number | null>(null);
  const [explain, setExplain] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<Analysis | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [dec, setDec] = useState<Record<number, Decision>>({});
  const [lat, setLat] = useState<Record<string, number>>({});
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getModels().then((m) => { setModels(m.models); setModel(m.default); }).catch((e: Error) => setModelErr(e.message));
    getSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  const info = models.find((m) => m.id === model);
  const effThr = thr ?? res?.threshold ?? info?.threshold ?? 0.5;

  const hunks = useMemo(() => (res ? [...res.hunks].sort((a, b) => a.rank - b.rank) : []), [res]);
  const selected = hunks.find((h) => h.index === sel) ?? null;

  const run = useCallback(async (override?: { model?: string; explain?: boolean }) => {
    const m = override?.model ?? model;
    const ex = override?.explain ?? explain;
    setBusy(true); setError(null);
    try {
      const body = sampleId
        ? { sample_id: sampleId, model: m, explain: ex }
        : { turns: turnsText.split("\n").map((t) => t.trim()).filter(Boolean), diff: diffText, model: m, explain: ex };
      const r = await analyze(body);
      setRes(r); setDec({}); setThr(null);
      setSel(r.hunks.length ? [...r.hunks].sort((a, b) => a.rank - b.rank)[0].index : null);
      setLat((l) => ({ ...l, [m]: r.model_latency_ms }));
      if (!window.matchMedia(DESK).matches) setOpenMob(false);
    } catch (e) {
      setError(e instanceof ApiFail ? e.message : String(e));
    } finally { setBusy(false); }
  }, [model, explain, sampleId, turnsText, diffText]);

  const loadSample = async (id: string) => {
    setSampleId(id); setRes(null); setError(null);
    if (!id) return;
    try {
      const s = await api<{ turns: string[]; diff: string }>(`/api/samples/${id}`);
      setTurnsText(s.turns.join("\n")); setDiffText(s.diff);
    } catch (e) { setError((e as Error).message); }
  };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    if (f.size > 1_000_000) { setError("File is larger than 1 MB, the backend limit for a diff."); return; }
    setSampleId(""); setDiffText(await f.text()); setRes(null);
  };

  const step = useCallback((d: number) => {
    if (!hunks.length) return;
    const i = Math.max(0, hunks.findIndex((h) => h.index === sel));
    const n = hunks[Math.min(hunks.length - 1, Math.max(0, i + d))];
    setSel(n.index);
    document.getElementById(`hunk-${n.index}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [hunks, sel]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "[") { e.preventDefault(); toggleInputs(); }
      else if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); step(1); }
      else if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); step(-1); }
      else if (sel != null && "afd".includes(e.key) && e.key.length === 1) {
        const d: Decision = e.key === "a" ? "accept" : e.key === "f" ? "flag" : "discuss";
        setDec((p) => ({ ...p, [sel]: d }));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, sel, toggleInputs]);

  const flaggedByModel = hunks.filter((h) => h.p_misaligned >= effThr).length;
  const needAttention = hunks.filter((h) => dec[h.index] === "flag" || dec[h.index] === "discuss" || (dec[h.index] == null && h.p_misaligned >= effThr)).length;
  const count = (d: Decision) => hunks.filter((h) => dec[h.index] === d).length;

  return (
    <>
    <div className={clsx("grid gap-4 lg:transition-[grid-template-columns] lg:duration-200 lg:ease-out",
      openDesk ? "lg:grid-cols-[320px_minmax(0,1fr)] xl:grid-cols-[320px_minmax(0,1fr)_400px]" : "lg:grid-cols-[44px_minmax(0,1fr)] xl:grid-cols-[44px_minmax(0,1fr)_480px]")}>
      {/* LEFT: intent + controls (collapsible) */}
      <aside aria-label="Inputs" className={clsx("lg:sticky lg:top-16 lg:self-start lg:max-h-[calc(100vh-5rem)] lg:overflow-y-auto lg:overflow-x-hidden", inputsOpen ? "lg:pr-1" : "")}>
        <button type="button" onClick={toggleInputs} data-testid="inputs-toggle" aria-expanded={inputsOpen} aria-controls="inputs-panel"
          title={`${inputsOpen ? "Hide" : "Show"} the inputs panel (shortcut: [)`}
          className={clsx("btn mb-3 w-full", !inputsOpen && "lg:mb-0 lg:h-36 lg:w-11 lg:flex-col lg:px-0 lg:py-3")}>
          <Icon name={inputsOpen ? "panelOpen" : "panelClosed"} className="h-4 w-4 shrink-0" />
          <span className={clsx(!inputsOpen && "lg:[writing-mode:vertical-rl]")}>{inputsOpen ? "Hide inputs" : "Show inputs"}</span>
        </button>
        <div id="inputs-panel" data-testid="inputs-panel"
          className={clsx("grid transition-[grid-template-rows,opacity] duration-200 ease-out", inputsOpen ? "grid-rows-[1fr] opacity-100" : "invisible grid-rows-[0fr] opacity-0")}>
         <div className="min-h-0 overflow-hidden">
          <div className="space-y-4 lg:w-[308px]">
        <div className="card space-y-3 p-3.5">
          <div>
            <label className="label" htmlFor="sample" title="Curated from the held-out test split. Ground truth is shown after analysis.">Test-set example</label>
            <select id="sample" className="input" value={sampleId} onChange={(e) => loadSample(e.target.value)} disabled={!samples}>
              <option value="">{samples ? "Custom input" : "Loading examples..."}</option>
              {samples?.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.repo} · {s.tag} · {s.n_hunks} hunks</option>)}
            </select>
            
          </div>
          <div>
            <div className="flex items-center justify-between"><label className="label" htmlFor="turns">Session intent (one turn per line)</label></div>
            <textarea id="turns" rows={3} className="input resize-y font-mono text-[12px]" value={turnsText}
              onChange={(e) => { setTurnsText(e.target.value); setSampleId(""); }} placeholder="What did you ask the agent to do?" />
          </div>
          <div>
            <div className="flex items-center justify-between">
              <label className="label" htmlFor="diff">Unified diff</label>
              <button className="mb-1 inline-flex items-center gap-1 text-[12px] font-medium text-accent hover:underline" onClick={() => fileRef.current?.click()}>
                <Icon name="upload" className="h-3 w-3" />Upload .diff / .patch
              </button>
              <input ref={fileRef} type="file" accept=".diff,.patch,.txt" hidden onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
            </div>
            <textarea id="diff" rows={6} className="input resize-y font-mono text-[11.5px]" value={diffText} spellCheck={false}
              onChange={(e) => { setDiffText(e.target.value); setSampleId(""); }} placeholder={"diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,3 @@ ..."} />
          </div>
          <div>
            <label className="label" htmlFor="model">Model</label>
            {modelErr ? <ErrorBox message={modelErr} /> : (
              <select id="model" className="input" value={model} onChange={(e) => { setModel(e.target.value); setThr(null); if (res) run({ model: e.target.value }); }}>
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name}{m.size_mb ? ` · ${m.size_mb} MB` : ""}{lat[m.id] != null ? ` · ${lat[m.id]} ms` : ""}{!m.available ? " (unavailable)" : ""}
                  </option>
                ))}
              </select>
            )}
            {info && (
              <details className="mt-1 text-[11.5px] text-muted">
                <summary className="cursor-pointer select-none">Model details</summary>
                <p className="mt-1">{info.description}.{" "}{lat[model] != null ? <>Last run: <b className="text-fg">{lat[model]} ms</b> (model only, this machine).</> : "Latency shows after the first run."}</p>
              </details>
            )}
            {models.filter((m) => !m.available).map((m) => (
              <p key={m.id} className="mt-1 text-[11.5px] text-warn">{m.name}: {m.reason}</p>
            ))}
          </div>
          <label className="flex items-start gap-2 text-[12.5px]">
            <input type="checkbox" className="mt-0.5 accent-[rgb(var(--accent))]" checked={explain} onChange={(e) => setExplain(e.target.checked)} disabled={info ? !info.supports_explain : false} />
            <span title="Integrated Gradients on the top 3 hunks. PyTorch models only; slow on CPU.">Explain top hunks <span className="text-[11.5px] text-muted">(slow on CPU)</span></span>
          </label>
          <button className="btn btn-primary w-full" onClick={() => run()} disabled={busy || !info?.available || (!sampleId && (!diffText.trim() || !turnsText.trim()))}>
            <Icon name="play" className="h-3.5 w-3.5" />{busy ? "Analysing..." : "Analyse"}
          </button>
        </div>

        {/* intent panel */}
        <div className="card space-y-3 p-3.5" aria-live="polite">
          <h2 className="h-sec text-[14px]">Consolidated intent</h2>
          {busy ? <SkeletonBlock rows={3} /> : !res ? (
            <p className="text-[12.5px] text-muted">Shown after you press Analyse.</p>
          ) : (
            <>
              <blockquote className="rounded-lg border-l-2 border-accent bg-raised p-2.5 text-[13px]">{res.intent}</blockquote>
              <div>
                <div className="label">Dropped as noise ({res.dropped_turns.length})</div>
                {res.dropped_turns.length === 0 ? <p className="text-[12.5px] text-muted">No turns dropped.</p> : (
                  <ul className="space-y-1.5">
                    {res.dropped_turns.map((d, i) => (
                      <li key={i} className="rounded-lg border border-line p-2 text-[12.5px]">
                        <Badge tone="warn">{REASON_LABEL[d.reason] ?? d.reason}</Badge>
                        <div className="mt-1 text-muted line-through decoration-muted/50">{d.turn}</div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {res.notes.length > 0 && (
                <div>
                  <div className="label">Diff notes</div>
                  <ul className="space-y-1 text-[12px] text-muted">{res.notes.map((n, i) => <li key={i}><b className="text-fg">{n.kind}</b> {n.file}: {n.detail}</li>)}</ul>
                </div>
              )}
            </>
          )}
        </div>
          </div>
         </div>
        </div>
      </aside>

      {/* CENTRE: ranked hunks */}
      <section className="min-w-0 space-y-3" aria-label="Ranked hunks">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-1 pb-1">
          <div className="min-w-[160px]">
                        <div className="text-[17px] font-semibold" data-testid="summary" title={res ? `${flaggedByModel} flagged by the model at threshold ${effThr.toFixed(2)}` : undefined}>
              {res ? <><span className={needAttention ? "text-accent" : ""}>{needAttention}</span> of {hunks.length} hunks need attention</> : "Ready to review"}
            </div>
            {res && (count("accept") + count("flag") + count("discuss") > 0) && <div className="text-[12px] text-muted">{count("accept")} accepted · {count("flag")} flagged · {count("discuss")} to discuss</div>}
          </div>
          <div className="ml-auto w-full sm:w-52">
            <div className="flex items-center justify-between">
              <label className="label !mb-0" htmlFor="thr">Threshold</label>
              <span className="font-mono text-[12px]">{effThr.toFixed(2)}</span>
            </div>
            <input id="thr" type="range" min={0.01} max={0.99} step={0.01} value={effThr} onChange={(e) => setThr(Number(e.target.value))} className="w-full accent-[rgb(var(--accent))]" />
            {thr != null && <div className="text-right text-[11px]"><button className="text-accent hover:underline" onClick={() => setThr(null)}>reset to {info?.threshold ?? "default"}</button></div>}
          </div>
        </div>

        {error && <ErrorBox message={error} onRetry={() => run()} />}

        {busy && !res ? (
          <div className="space-y-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-40" />)}</div>
        ) : !res ? (
          <Empty title="Nothing analysed yet" icon={<Icon name="review" className="h-6 w-6" />}>
            Choose a test-set example or paste an intent and a diff, then press Analyse. The likeliest misaligned hunks come first.
          </Empty>
        ) : hunks.length === 0 ? (
          <Empty title="No hunks found">The backend parsed the diff but found no reviewable hunks.</Empty>
        ) : (
          <>
            <ul className={clsx("space-y-3 transition-opacity", busy && "opacity-60")}>
              {hunks.map((h) => (
                <HunkCard key={h.index} h={h} thr={effThr} active={h.index === sel} decision={dec[h.index]} onSelect={() => setSel(h.index)}
                  onDecide={(d) => setDec((p) => ({ ...p, [h.index]: d }))} />
              ))}
            </ul>
            <details className="text-center text-[11.5px] text-muted">
              <summary className="cursor-pointer select-none">Keyboard shortcuts</summary>
              <p className="mt-1">j/k step hunks · a accept · f flag · d discuss · [ hide/show inputs</p>
            </details>
          </>
        )}
      </section>

      {/* RIGHT: detail */}
      <aside className="min-w-0 lg:col-span-2 xl:col-span-1 xl:sticky xl:top-16 xl:max-h-[calc(100vh-5rem)] xl:self-start xl:overflow-y-auto" aria-label="Hunk detail">
        {selected && res ? (
          <Detail h={selected} thr={effThr} res={res} decision={dec[selected.index]} onDecide={(d) => setDec((p) => ({ ...p, [selected.index]: d }))}
            onExplain={() => { setExplain(true); run({ explain: true }); }} canExplain={!!info?.supports_explain} busy={busy} />
        ) : (
          null
        )}
      </aside>
    </div>
    </>
  );
}

function Decisions({ value, onChange }: { value?: Decision; onChange: (d: Decision) => void }) {
  const opts: { d: Decision; label: string; key: string; cls: string }[] = [
    { d: "accept", label: "Accept", key: "a", cls: "data-[on=true]:bg-ok/15 data-[on=true]:border-ok/50 data-[on=true]:text-ok" },
    { d: "flag", label: "Flag", key: "f", cls: "data-[on=true]:bg-bad/15 data-[on=true]:border-bad/50 data-[on=true]:text-bad" },
    { d: "discuss", label: "Needs discussion", key: "d", cls: "data-[on=true]:bg-warn/15 data-[on=true]:border-warn/50 data-[on=true]:text-warn" },
  ];
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Review decision">
      {opts.map((o) => (
        <button key={o.d} data-on={value === o.d} aria-pressed={value === o.d} className={clsx("btn !px-2.5 !py-1 text-[12px]", o.cls)} onClick={(e) => { e.stopPropagation(); onChange(o.d); }}>
          {o.label}<kbd className="ml-1 text-[10px] opacity-60">{o.key}</kbd>
        </button>
      ))}
    </div>
  );
}

function HunkCard({ h, thr, active, decision, onSelect, onDecide }: { h: HunkResult; thr: number; active: boolean; decision?: Decision; onSelect: () => void; onDecide: (d: Decision) => void }) {
  const flagged = h.p_misaligned >= thr;
  return (
    <li id={`hunk-${h.index}`}>
      <article onClick={onSelect} tabIndex={0} onFocus={onSelect} aria-current={active} aria-label={`Hunk ${h.rank}, ${h.file}, ${pct(h.p_misaligned, 0)} misaligned`}
        className={clsx("card cursor-pointer overflow-hidden transition-shadow fade-in", active && "ring-2 ring-accent/50")}>
        <header className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3.5 pb-2 pt-3">
          <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] font-medium" title={h.file}>{h.rank}. {h.file}</span>
          <span className={clsx("text-[13px] font-semibold tabular-nums", flagged ? "text-accent" : "text-muted")}>
            {flagged ? "Likely misaligned" : "Likely aligned"} · {pct(h.p_misaligned, 0)}
          </span>
          {flagged && h.drift_type && <Badge tone="accent" title="Predicted drift type">{driftName(h.drift_type)}</Badge>}
          {decision && <Badge tone="neutral">{DEC_LABEL[decision]}</Badge>}
        </header>
        <div className="px-3.5 pb-2"><DiffView text={h.text} file={h.file} maxLines={24} /></div>
        <footer className="px-3.5 pb-3"><Decisions value={decision} onChange={onDecide} /></footer>
      </article>
    </li>
  );
}

function Detail({ h, thr, res, onExplain, canExplain, busy }: { h: HunkResult; thr: number; res: Analysis; decision?: Decision; onDecide: (d: Decision) => void; onExplain: () => void; canExplain: boolean; busy: boolean }) {
  const flagged = h.p_misaligned >= thr;
  const at = h.attributions;
  const maxW = at ? Math.max(0.0001, ...at.intent_words.map(([, s]) => Math.abs(s))) : 1;
  const maxL = at ? Math.max(0.0001, ...at.hunk_lines.map(([, s]) => Math.abs(s))) : 1;
  const lineScores = useMemo(() => new Map((at?.hunk_lines ?? []).map(([l, s]) => [l, s / maxL] as [string, number])), [at, maxL]);
  const dp = h.drift_probs && !Array.isArray(h.drift_probs) ? Object.entries(h.drift_probs) : null;
  const g = h.ground_truth;
  return (
    <div className="card space-y-3 p-4 fade-in">
      <div>
        <div className="break-all font-mono text-[12.5px] font-medium">{h.file}</div>
        <div className="mt-1.5 flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <span className={clsx("text-3xl font-semibold tabular-nums", flagged ? "text-accent" : "text-fg")}>{pct(h.p_misaligned, 0)}</span>
          <span className="text-[13px] text-muted">{flagged ? "likely misaligned" : "likely aligned"}</span>
          {flagged && h.drift_type && <Badge tone="accent">{driftName(h.drift_type)}</Badge>}
        </div>
        {g && <p className="mt-1.5 text-[12px] text-muted" title="Ground truth from dataset construction">Truth: <b className="text-fg">{g.label}{g.drift_type && g.drift_type !== "none" ? `, ${driftName(g.drift_type)}` : ""}</b> · model {h.correct ? "correct" : "wrong"}</p>}
      </div>

      {at ? (
        <div className="space-y-3">
          <div>
            <div className="mb-1 text-[12px] text-muted">Intent words <span className="opacity-70">(red pushes toward misaligned, blue toward aligned)</span></div>
            <p className="rounded-lg bg-raised p-2 text-[13px] leading-7">
              {at.intent_words.map(([w, s], i) => <span key={i} className="rounded px-0.5" style={{ background: attrColor(s, maxW) }} title={s.toFixed(4)}>{w} </span>)}
            </p>
          </div>
          <div>
            <div className="mb-1 text-[12px] text-muted">Hunk lines</div>
            <DiffView text={h.text} file={h.file} lineScores={lineScores} />
          </div>
          <p className="text-[11.5px] text-muted">{at.method.replace("_", " ")}, {at.n_steps} steps. Evidence about the model, not proof of the cause.</p>
        </div>
      ) : canExplain ? (
        <button className="btn w-full" onClick={onExplain} disabled={busy} title="Integrated Gradients on the top 3 hunks (slow on CPU)">Explain this hunk</button>
      ) : null}

      <details className="text-[12px]">
        <summary className="cursor-pointer select-none text-muted">Details</summary>
        <div className="mt-2 space-y-3">
          <p className="text-muted">Calibrated confidence {pct(h.confidence, 0)}{h.confidence_head != null ? ` · confidence head ${h.confidence_head.toFixed(2)}` : ""} · {res.model_latency_ms} ms model, {res.total_ms} ms total · T={res.temperature.toFixed(2)} · threshold {thr.toFixed(2)}</p>
          {dp && dp.length > 0 && (
            <div>
              <div className="label">Drift-type probabilities</div>
              {dp.map(([k, v]) => (
                <div key={k} className="mb-1 flex items-center gap-2">
                  <span className="w-28 text-muted">{driftName(k)}</span>
                  <div className="h-1.5 flex-1 rounded-full bg-raised"><div className="h-full rounded-full bg-accent" style={{ width: pct(Number(v), 0) }} /></div>
                  <span className="w-10 text-right tabular-nums">{pct(Number(v), 0)}</span>
                </div>
              ))}
            </div>
          )}
          <div>
            <div className="label">Similar flagged hunks</div>
            {h.cluster_id == null ? (
              <p className="text-muted" data-testid="no-cluster">Only available for the main model (the k-means index uses its embeddings).</p>
            ) : (
              <div className="space-y-2">
                <p>Cluster <b>{h.cluster_id}</b>{h.cluster_prior != null && <>; {pct(h.cluster_prior, 0)} of training hunks in it are misaligned</>}.</p>
                {(h.similar ?? []).map((s) => (
                  <details key={s.sample_id} className="rounded-lg border border-line bg-raised/50 p-2">
                    <summary className="cursor-pointer select-none">
                      <span className="font-mono">{s.sample_id}</span>
                      <span className="text-muted"> · {s.repo ?? "train"} · cos {s.similarity.toFixed(2)}{s.drift_type ? ` · ${driftName(s.drift_type)}` : ""}</span>
                    </summary>
                    {s.text && <div className="mt-2"><DiffView text={s.text} file={s.file ?? ""} maxLines={16} /></div>}
                  </details>
                ))}
              </div>
            )}
          </div>
        </div>
      </details>
    </div>
  );
}
