import { useEffect, useState } from "react";

const KEY = "sift-theme";
const mq = () => window.matchMedia("(prefers-color-scheme: dark)");

function stored(): "dark" | "light" | null {
  try { const v = localStorage.getItem(KEY); return v === "dark" || v === "light" ? v : null; } catch { return null; }
}

/** Follows the system until the user toggles; after that the explicit choice is remembered. */
export function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);
  useEffect(() => {
    const m = mq();
    const on = () => { if (!stored()) setDark(m.matches); };
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  const toggle = () => setDark((d) => {
    try { localStorage.setItem(KEY, d ? "light" : "dark"); } catch { /* storage blocked */ }
    return !d;
  });
  return { dark, toggle };
}
