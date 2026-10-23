import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import clsx from "clsx";
import { Icon } from "../components/ui";
import { useApi } from "../api";
import { useTheme } from "../theme";

const NAV = [
  { to: "/dev", label: "Overview", icon: "review", end: true },
  { to: "/dev/results", label: "Results", icon: "chart" },
  { to: "/dev/experiments", label: "Experiments", icon: "flask" },
  { to: "/dev/dataset", label: "Dataset", icon: "db" },
  { to: "/dev/testing", label: "Testing", icon: "check" },
  { to: "/dev/related-work", label: "Related work", icon: "book" },
];

function Health() {
  const { data, error } = useApi<{ status: string; models_available: string[] }>("/api/health");
  const ok = !!data && !error;
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[11px] text-muted" title={error ?? `Models available: ${data?.models_available.join(", ")}`}>
      <span className={clsx("h-2 w-2 rounded-full", ok ? "bg-ok" : error ? "bg-bad" : "bg-muted")} />
      {ok ? "backend online" : error ? "backend offline" : "connecting"}
    </span>
  );
}

export default function DevLayout() {
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo(0, 0); }, [loc.pathname]);
  useEffect(() => { document.title = "Sift developer console"; }, []);
  return (
    <div className="dev flex min-h-full flex-col lg:flex-row">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-accent focus:px-3 focus:py-1 focus:text-white">Skip to content</a>
      <aside className={clsx("z-30 border-line bg-surface lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-56 lg:shrink-0 lg:flex-col lg:border-r", "border-b lg:border-b-0")}>
        <div className="flex h-12 items-center gap-2 px-3">
          <span className="grid h-6 w-6 place-items-center rounded-md bg-accent text-white dark:text-bg"><Icon name="menu" className="h-3.5 w-3.5" /></span>
          <div className="leading-tight"><div className="text-[13px] font-semibold tracking-tight">Sift</div><div className="font-mono text-[10px] uppercase tracking-wider text-muted">Developer console</div></div>
          <button className="btn ml-auto px-2 lg:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu" aria-expanded={open}><Icon name="menu" /></button>
        </div>
        <nav className={clsx("flex-1 flex-col gap-0.5 px-2 pb-2 lg:flex", open ? "flex" : "hidden")} aria-label="Console">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) => clsx("flex items-center gap-2 rounded-md border-l-2 px-2.5 py-1.5 text-[13px] font-medium transition-colors", isActive ? "border-accent bg-raised text-fg" : "border-transparent text-muted hover:bg-raised/60 hover:text-fg")}>
              <Icon name={n.icon} className="h-3.5 w-3.5" />{n.label}
            </NavLink>
          ))}
        </nav>
        <div className={clsx("space-y-2 border-t border-line p-3 lg:block", open ? "block" : "hidden")}>
          <Health />
          <div className="flex items-center gap-2">
            <Link to="/" className="btn flex-1 text-[12px]">&larr; Back to product</Link>
            <button className="btn px-2" onClick={toggle} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} title="Toggle theme"><Icon name={dark ? "sun" : "moon"} /></button>
          </div>
        </div>
      </aside>
      <main id="main" className="min-w-0 flex-1 px-4 py-5 lg:px-6">
        <div className="mx-auto max-w-[1500px]"><Outlet /></div>
      </main>
    </div>
  );
}
