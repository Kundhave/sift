import { type ReactNode } from "react";
import clsx from "clsx";

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton", className)} aria-hidden />;
}
export function SkeletonBlock({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-2" role="status" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => <Skeleton key={i} className="h-5" />)}
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start justify-between gap-3 rounded-xl border border-bad/40 bg-bad/10 p-3 text-[13px]">
      <div><span className="font-semibold text-bad">Error. </span>{message}</div>
      {onRetry && <button className="btn shrink-0" onClick={onRetry}>Retry</button>}
    </div>
  );
}

export function Empty({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-line p-8 text-center">
      {icon && <div className="mb-1 text-muted">{icon}</div>}
      <div className="font-medium">{title}</div>
      {children && <div className="max-w-md text-[13px] text-muted">{children}</div>}
    </div>
  );
}

/** Shown wherever a result artifact has not been produced yet. Driven by the API (file index), never by data. */
export function NotAvailable({ what, expects }: { what: string; expects?: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-dashed border-line bg-raised/50 p-4" data-testid="not-available">
      <span className="mt-0.5 inline-block h-2 w-2 shrink-0 rounded-full bg-warn" aria-hidden />
      <div className="text-[13px]">
        <div className="font-medium">{what}: not yet available</div>
        <div className="text-muted">
          This result has not been produced yet. It appears here automatically once
          {expects ? <> <code className="rounded bg-raised px-1 font-mono text-[12px]">{expects}</code> is written to results/</> : " its file is written to results/"}.
        </div>
      </div>
    </div>
  );
}

export function Badge({ children, tone = "neutral", title }: { children: ReactNode; tone?: "neutral" | "ok" | "warn" | "bad" | "accent"; title?: string }) {
  const t = {
    neutral: "border-line bg-raised text-muted", ok: "border-ok/30 bg-ok/10 text-ok", warn: "border-warn/30 bg-warn/10 text-warn",
    bad: "border-bad/30 bg-bad/10 text-bad", accent: "border-accent/30 bg-accent/10 text-accent",
  }[tone];
  return <span title={title} className={clsx("inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-4", t)}>{children}</span>;
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card p-3.5">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums tracking-tight">{value}</div>
      {sub && <div className="mt-0.5 text-[12px] text-muted">{sub}</div>}
    </div>
  );
}

export function Section({ title, hint, children, right }: { title: string; hint?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div><h2 className="h-sec">{title}</h2>{hint && <p className="text-[13px] text-muted">{hint}</p>}</div>
        {right}
      </div>
      {children}
    </section>
  );
}

export function PageHeader({ title, sub }: { title: string; sub?: ReactNode }) {
  return (
    <header className="mb-5">
      <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
      {sub && <p className="mt-0.5 max-w-3xl text-[13px] text-muted">{sub}</p>}
    </header>
  );
}

export function Caption({ children }: { children: ReactNode }) {
  return <p className="mt-2 text-[12.5px] text-muted"><span className="font-medium text-fg">What this means: </span>{children}</p>;
}

export function Segmented<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { v: T; label: string }[] }) {
  return (
    <div role="tablist" className="inline-flex rounded-lg border border-line bg-raised p-0.5">
      {options.map((o) => (
        <button key={o.v} role="tab" aria-selected={value === o.v} onClick={() => onChange(o.v)}
          className={clsx("rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors",
            value === o.v ? "bg-surface text-fg shadow-sm" : "text-muted hover:text-fg")}>{o.label}</button>
      ))}
    </div>
  );
}

export function Limits({ items }: { items: string[] }) {
  return (
    <div className="rounded-xl border border-warn/30 bg-warn/5 p-4">
      <div className="mb-1.5 text-[13px] font-semibold text-warn">Honest limitations</div>
      <ul className="list-disc space-y-1 pl-5 text-[13px]">{items.map((i) => <li key={i}>{i}</li>)}</ul>
    </div>
  );
}

export const LIMITS = [
  "Labels in Sift-IA v1.0 are construction-determined (a misaligned hunk is a donor hunk from a different commit). No human labels exist yet.",
  "The test split is repository-disjoint but still drawn from open-source Python and JavaScript repositories.",
  "Because misaligned hunks are constructed, a model can learn construction artefacts. The hard tier mitigates this but does not remove it.",
  "Latency and size come only from the machine they were measured on.",
];

export function Icon({ name, className = "h-4 w-4" }: { name: string; className?: string }) {
  const p: Record<string, string> = {
    review: "M4 6h16M4 12h10M4 18h6", chart: "M4 20V10m6 10V4m6 16v-7m4 7H2", flask: "M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3",
    db: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
    check: "M5 12l5 5L20 7", info: "M12 8h.01M11 12h1v5h1M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", sun: "M12 3v2m0 14v2M3 12h2m14 0h2M5.6 5.6 7 7m10 10 1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z",
    moon: "M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z", upload: "M12 16V4m0 0L7 9m5-5 5 5M4 20h16", play: "M7 4l13 8-13 8V4Z", menu: "M4 6h16M4 12h16M4 18h16",
    book: "M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2V5Zm0 14a2 2 0 0 1 2-2h13", panelOpen: "M3 5h18v14H3V5Zm6 0v14M15 10l-2 2 2 2", panelClosed: "M3 5h18v14H3V5Zm6 0v14M13 10l2 2-2 2",
    arrow: "M5 12h14m-6-6 6 6-6 6", bolt: "M13 3 5 14h6l-1 7 8-11h-6l1-7Z", lock: "M6 11h12v9H6v-9Zm3 0V8a3 3 0 0 1 6 0v3",
    layers: "m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5", gauge: "M4 17a8 8 0 1 1 16 0M12 17l4-5", list: "M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01", search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14Zm9 16-4-4",
    sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z", target: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 5a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z",
    eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Zm10-3a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z", close: "M6 6l12 12M18 6 6 18",
  };
  return <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d={p[name] ?? ""} /></svg>;
}
