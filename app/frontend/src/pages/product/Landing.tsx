import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "../../api";
import { Icon } from "../../components/ui";
import { Reveal } from "../../components/Reveal";

interface Proof { main: number; base: number; baseName: string; size: number | null; p95: number | null; cpu: string | null }

/** All proof numbers are read from the result files served by /api/results/*. Nothing is typed in. */
function useProof() {
  const [p, setP] = useState<Proof | null>(null);
  useEffect(() => {
    let live = true;
    Promise.all([api<any>("/api/results/master_comparison.json"), api<any>("/api/results/latency_size_onnx.json").catch(() => null)])
      .then(([m, o]) => {
        if (!live) return;
        const rows: any[] = m.rows ?? [];
        const main = rows.find((r) => r.group === "main");
        const simple = rows.filter((r) => r.group === "non_neural" && typeof r.auc_hard === "number");
        if (!main || !simple.length) return;
        const b = simple.reduce((a, c) => (c.auc_hard > a.auc_hard ? c : a));
        const i8 = o?.student_onnx_int8;
        setP({ main: main.auc_hard, base: b.auc_hard, baseName: b.model.replace(/^B\d\s*/, ""), size: i8?.size_mb ?? null, p95: i8?.latency_ms?.p95_ms ?? null, cpu: o?.cpu?.cpu ?? null });
      }).catch(() => { /* proof row simply stays hidden */ });
    return () => { live = false; };
  }, []);
  return p;
}

const STEPS = [
  { icon: "filter", title: "Consolidate intent", body: "Sift reads your whole conversation with the agent, drops chit-chat and retracted requests, and keeps what you actually asked for." },
  { icon: "search", title: "Classify every hunk", body: "A cross-encoder reads the intent and one hunk together and decides whether that change serves the request." },
  { icon: "list", title: "Rank and explain", body: "Hunks are ordered by calibrated misalignment probability, with a drift type and the words and lines that drove the call." },
];

const FEATURES = [
  { icon: "layers", title: "Hunk-level ranking", body: "A diff is many decisions. Sift scores each hunk, so the one stray change in a big diff rises to the top." },
  { icon: "gauge", title: "Calibrated confidence", body: "Probabilities are temperature-scaled, so 80% means roughly 80%. Move the threshold to suit how careful you want to be." },
  { icon: "eye", title: "Explanations", body: "Integrated Gradients highlight the intent words and diff lines behind each verdict. It is evidence, not proof." },
  { icon: "bolt", title: "Fast and local", body: "An INT8 student model runs on a laptop CPU in milliseconds. No GPU, no account, no upload." },
];

const DRIFT = [
  ["Collateral", "An unrequested change next to the requested one, like reformatting a module while fixing one function."],
  ["Scope-Expansion", "An extra feature or behaviour nobody asked for."],
  ["Purpose-Drift", "A change that serves a different purpose than the one you described."],
];

function Eyebrow({ children }: { children: string }) {
  return <div className="mb-3 text-[12.5px] font-semibold uppercase tracking-[0.14em] text-accent">{children}</div>;
}

function Shot() {
  return (
    <div className="shot overflow-hidden rounded-2xl border border-line bg-surface">
      <div className="flex items-center gap-1.5 border-b border-line bg-raised px-3.5 py-2.5" aria-hidden>
        <span className="h-2.5 w-2.5 rounded-full bg-line" /><span className="h-2.5 w-2.5 rounded-full bg-line" /><span className="h-2.5 w-2.5 rounded-full bg-line" />
        <span className="ml-3 rounded-md bg-bg px-3 py-0.5 text-[11px] text-muted">sift / workspace</span>
      </div>
      <img src="/shots/workspace-light.png" alt="The Sift workspace: a ranked list of diff hunks with misalignment probability, drift type and an explanation panel" className="block w-full dark:hidden" width={1440} height={820} />
      <img src="/shots/workspace-dark.png" alt="" aria-hidden className="hidden w-full dark:block" width={1440} height={820} />
    </div>
  );
}

export default function Landing() {
  const proof = useProof();
  return (
    <>
      {/* hero */}
      <section className="relative overflow-hidden">
        <div className="hero-bg absolute inset-0" aria-hidden />
        <div className="relative mx-auto max-w-6xl px-4 pb-12 pt-16 text-center sm:px-6 sm:pt-24">
          <div className="rise mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-line bg-surface/80 px-3 py-1 text-[12.5px] text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />Local-first review for AI coding agents
          </div>
          <h1 className="rise rise-2 mx-auto max-w-3xl text-balance text-[40px] font-semibold leading-[1.05] tracking-tight sm:text-[64px]">
            Review the hunks <span className="text-accent">that matter first.</span>
          </h1>
          <p className="rise rise-3 mx-auto mt-6 max-w-2xl text-[17px] leading-relaxed text-muted sm:text-[19px]">
            Sift reads what you asked your AI coding agent to do and ranks every diff hunk by how likely it drifted from that intent. It runs on your machine.
          </p>
          <div className="rise rise-4 mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link to="/app" className="btn btn-primary btn-lg w-full sm:w-auto">Try Sift<Icon name="arrow" className="h-4 w-4" /></Link>
            <a href="#how" className="btn btn-lg w-full sm:w-auto">See how it works</a>
          </div>
          <div className="rise rise-4 mx-auto mt-14 max-w-5xl text-left"><Shot /></div>
        </div>
      </section>

      {/* how it works */}
      <section id="how" className="mx-auto max-w-6xl scroll-mt-20 px-4 py-20 sm:px-6">
        <Reveal><Eyebrow>How it works</Eyebrow><h2 className="max-w-xl text-3xl font-semibold tracking-tight sm:text-4xl">From a long diff to a short list worth reading.</h2></Reveal>
        <ol className="mt-10 grid gap-4 md:grid-cols-3">
          {STEPS.map((s, i) => (
            <Reveal key={s.title} delay={i * 90}>
              <li className="card relative h-full list-none p-6">
                <div className="flex items-center justify-between">
                  <span className="grid h-10 w-10 place-items-center rounded-xl bg-accent/10 text-accent"><Icon name={s.icon === "filter" ? "sparkle" : s.icon} className="h-5 w-5" /></span>
                  <span className="font-mono text-[13px] text-muted">0{i + 1}</span>
                </div>
                <h3 className="mt-5 text-lg font-semibold tracking-tight">{s.title}</h3>
                <p className="mt-2 text-[14.5px] leading-relaxed text-muted">{s.body}</p>
              </li>
            </Reveal>
          ))}
        </ol>
      </section>

      {/* features */}
      <section className="border-y border-line bg-raised/40">
        <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
          <Reveal><Eyebrow>Why Sift</Eyebrow><h2 className="max-w-xl text-3xl font-semibold tracking-tight sm:text-4xl">Built for the way you review agent output.</h2></Reveal>
          <div className="mt-10 grid gap-x-10 gap-y-9 sm:grid-cols-2">
            {FEATURES.map((f, i) => (
              <Reveal key={f.title} delay={(i % 2) * 90}>
                <div className="flex gap-4">
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-line bg-surface text-accent"><Icon name={f.icon} className="h-5 w-5" /></span>
                  <div><h3 className="font-semibold tracking-tight">{f.title}</h3><p className="mt-1 text-[14.5px] leading-relaxed text-muted">{f.body}</p></div>
                </div>
              </Reveal>
            ))}
          </div>
          <Reveal className="mt-12">
            <div className="card p-6 sm:p-8">
              <h3 className="text-lg font-semibold tracking-tight">Three kinds of drift, named</h3>
              <div className="mt-5 grid gap-6 sm:grid-cols-3">
                {DRIFT.map(([t, d]) => (
                  <div key={t}><div className="inline-block rounded-md border border-accent/30 bg-accent/10 px-2 py-0.5 text-[12.5px] font-semibold text-accent">{t}</div><p className="mt-2 text-[14px] leading-relaxed text-muted">{d}</p></div>
                ))}
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* proof */}
      {proof && (
        <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6" aria-labelledby="proof-h">
          <Reveal><Eyebrow>Measured, not claimed</Eyebrow><h2 id="proof-h" className="max-w-xl text-3xl font-semibold tracking-tight sm:text-4xl">Numbers from the test set.</h2></Reveal>
          <div className="mt-10 grid gap-4 md:grid-cols-3" data-testid="proof">
            <Reveal><div className="card h-full p-6">
              <div className="text-[12px] font-semibold uppercase tracking-wider text-muted">Ranking quality</div>
              <div className="mt-2 text-5xl font-semibold tabular-nums tracking-tight" data-testid="proof-auc">{proof.main.toFixed(3)}</div>
              <p className="mt-2 text-[14px] text-muted">Hard-tier pairwise AUC, against <span className="tabular-nums text-fg" data-testid="proof-baseline">{proof.base.toFixed(3)}</span> for the best simple baseline ({proof.baseName.toLowerCase()}).</p>
            </div></Reveal>
            {proof.size != null && <Reveal delay={90}><div className="card h-full p-6">
              <div className="text-[12px] font-semibold uppercase tracking-wider text-muted">Size</div>
              <div className="mt-2 text-5xl font-semibold tabular-nums tracking-tight"><span data-testid="proof-size">{proof.size.toFixed(1)}</span><span className="ml-1 text-2xl text-muted">MB</span></div>
              <p className="mt-2 text-[14px] text-muted">On-disk size of the INT8 model you run locally.</p>
            </div></Reveal>}
            {proof.p95 != null && <Reveal delay={180}><div className="card h-full p-6">
              <div className="text-[12px] font-semibold uppercase tracking-wider text-muted">Latency</div>
              <div className="mt-2 text-5xl font-semibold tabular-nums tracking-tight"><span data-testid="proof-p95">{proof.p95.toFixed(0)}</span><span className="ml-1 text-2xl text-muted">ms</span></div>
              <p className="mt-2 text-[14px] text-muted">p95 per hunk on a laptop CPU{proof.cpu ? ` (${proof.cpu})` : ""}.</p>
            </div></Reveal>}
          </div>
          <p className="mt-5 text-[12.5px] text-muted">Measured on the Sift-IA test set, single seed; see the <Link className="text-accent underline-offset-2 hover:underline" to="/dev">developer console</Link>.</p>
        </section>
      )}

      {/* final CTA */}
      <section className="mx-auto max-w-6xl px-4 pb-24 sm:px-6">
        <Reveal>
          <div className="relative overflow-hidden rounded-3xl border border-line bg-surface px-6 py-14 text-center sm:py-20">
            <div className="hero-bg absolute inset-0 opacity-70" aria-hidden />
            <div className="relative">
              <h2 className="mx-auto max-w-xl text-3xl font-semibold tracking-tight sm:text-4xl">Open a diff. See what drifted.</h2>
              <p className="mx-auto mt-3 max-w-md text-muted">Try it on a curated example, or paste your own intent and diff.</p>
              <Link to="/app" className="btn btn-primary btn-lg mt-8">Try Sift<Icon name="arrow" className="h-4 w-4" /></Link>
            </div>
          </div>
        </Reveal>
      </section>
    </>
  );
}
