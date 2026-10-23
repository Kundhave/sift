import { useEffect, useRef, useState, type ReactNode } from "react";
import clsx from "clsx";

/** Fades content in once when it scrolls into view. Content is visible without IntersectionObserver. */
export function Reveal({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [on, setOn] = useState(typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const el = ref.current;
    if (!el || on) return;
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setOn(true); io.disconnect(); } }, { rootMargin: "0px 0px -8% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [on]);
  return <div ref={ref} style={{ transitionDelay: `${delay}ms` }} className={clsx("reveal", on && "in", className)}>{children}</div>;
}
