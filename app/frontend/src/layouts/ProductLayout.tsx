import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import clsx from "clsx";
import { Icon } from "../components/ui";
import { useTheme } from "../theme";

export function Logo({ className }: { className?: string }) {
  return (
    <Link to="/" className={clsx("flex items-center gap-2 font-semibold tracking-tight", className)} aria-label="Sift home">
      <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-white dark:text-bg"><Icon name="menu" className="h-4 w-4" /></span>
      <span className="text-[17px]">Sift</span>
    </Link>
  );
}

const LINKS = [
  { to: "/how-it-works", label: "How it works" },
  { to: "/about", label: "About" },
];

export default function ProductLayout() {
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const workspace = loc.pathname === "/app";
  useEffect(() => { setOpen(false); if (!workspace) window.scrollTo(0, 0); }, [loc.pathname, workspace]);
  useEffect(() => {
    document.title = workspace ? "Sift workspace" : "Sift - review the hunks that matter first";
  }, [workspace]);
  const themeBtn = (
    <button className="btn px-2" onClick={toggle} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} title="Toggle theme"><Icon name={dark ? "sun" : "moon"} /></button>
  );
  return (
    <div className="product flex min-h-full flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-accent focus:px-3 focus:py-1 focus:text-white">Skip to content</a>
      <header className="sticky top-0 z-30 border-b border-line/70 bg-bg/80 backdrop-blur-md">
        <div className={clsx("mx-auto flex items-center gap-4 px-4 sm:px-6", workspace ? "h-12 max-w-[1700px]" : "h-16 max-w-6xl")}>
          <Logo />
          {workspace && <span className="hidden border-l border-line pl-4 text-[13px] font-medium text-muted sm:inline">Workspace</span>}
          <nav className="ml-4 hidden items-center gap-1 md:flex" aria-label="Primary">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} className={({ isActive }) => clsx("rounded-lg px-3 py-1.5 text-[14px] font-medium transition-colors", isActive ? "text-fg" : "text-muted hover:text-fg")}>{l.label}</NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <Link to="/dev" className="hidden text-[12px] text-muted hover:text-fg sm:inline" title="Metrics, experiments and tests">Developers</Link>
            {themeBtn}
            {!workspace && <Link to="/app" className="btn btn-primary hidden sm:inline-flex">Try Sift</Link>}
            <button className="btn px-2 md:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu" aria-expanded={open} aria-controls="mobile-nav"><Icon name={open ? "close" : "menu"} /></button>
          </div>
        </div>
        {open && (
          <nav id="mobile-nav" className="border-t border-line bg-bg px-4 py-3 md:hidden" aria-label="Mobile">
            <div className="flex flex-col gap-1 text-[15px]">
              {LINKS.map((l) => <NavLink key={l.to} to={l.to} className="rounded-lg px-3 py-2.5 font-medium hover:bg-raised">{l.label}</NavLink>)}
              <Link to="/dev" className="rounded-lg px-3 py-2.5 text-muted hover:bg-raised">Developers</Link>
              <Link to="/app" className="btn btn-primary mt-1 py-3 text-[15px]">Try Sift</Link>
            </div>
          </nav>
        )}
      </header>
      <main id="main" className={clsx("w-full flex-1", workspace && "mx-auto max-w-[1700px] px-4 py-5")}>
        <Outlet />
      </main>
      {!workspace && <Footer />}
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-12 sm:px-6 md:grid-cols-[1.4fr_1fr_1fr]">
        <div>
          <Logo />
          <p className="mt-3 max-w-xs text-[13.5px] text-muted">A local verification layer for AI coding agents. Your code never leaves your machine.</p>
          <div className="mt-4 flex gap-2">
            <span className="rounded-md border border-line px-2 py-0.5 text-[11.5px] font-medium text-muted">SDG 9</span>
            <span className="rounded-md border border-line px-2 py-0.5 text-[11.5px] font-medium text-muted">SDG 8</span>
          </div>
        </div>
        <div className="text-[13.5px]">
          <div className="mb-2 font-semibold">Product</div>
          <ul className="space-y-1.5 text-muted">
            <li><Link className="hover:text-fg" to="/app">Try Sift</Link></li>
            <li><Link className="hover:text-fg" to="/how-it-works">How it works</Link></li>
            <li><Link className="hover:text-fg" to="/about">About</Link></li>
            <li><Link className="hover:text-fg" to="/dev">Developers</Link></li>
          </ul>
        </div>
        <div className="text-[13.5px] text-muted">
          <div className="mb-2 font-semibold text-fg">Project</div>
          <p>Sift, a course project for 23CSE473 Neural Networks and Deep Learning.</p>
          <p className="mt-1">Amrita Vishwa Vidyapeetham, Department of Computer Science and Engineering.</p>
        </div>
      </div>
    </footer>
  );
}
