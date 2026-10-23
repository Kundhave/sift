import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { api } from "../api";
import { Badge, Icon } from "./ui";
import { FILES, MACHINE_FILE, buildItems, type Data, type PageKey, type Tag } from "./highlightData";

const KEY = "sift-highlights-collapsed";
const KEY_COMPACT = "sift-highlights-review-open";
const TAG_TONE: Record<Tag, "accent" | "neutral"> = { helped: "accent", "no clear effect": "neutral", "did not help": "neutral" };

function useCollapsed(compact: boolean) {
  const key = compact ? KEY_COMPACT : KEY;
  // The compact strip (Review) is collapsed unless the user opened it; the full strip is open unless the user hid it.
  const [c, setC] = useState(() => { try { const v = localStorage.getItem(key); return compact ? v !== "1" : v === "1"; } catch { return compact; } });
  const set = (v: boolean) => { setC(v); try { localStorage.setItem(key, compact ? (v ? "0" : "1") : (v ? "1" : "0")); } catch { /* storage blocked */ } };
  return [c, set] as const;
}

function useFiles(names: string[]) {
  const [data, setData] = useState<Data | null>(null);
  const key = names.join("|");
  useEffect(() => {
    let live = true;
    setData(null);
    Promise.all(names.map((n) => api<unknown>(n.startsWith("/") ? n : `/api/results/${n}`).catch(() => null)))
      .then((v) => live && setData(Object.fromEntries(names.map((n, i) => [n, v[i]]))));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return data;
}

/** Shared "What worked best" strip. Values are computed in the browser from the results files. */
export function Highlights({ page, compact = false }: { page: PageKey; compact?: boolean }) {
  const [collapsed, setCollapsed] = useCollapsed(compact);
  const names = useMemo(() => [...FILES[page], MACHINE_FILE], [page]);
  const data = useFiles(names);
  const items = useMemo(() => (data ? buildItems(page, data) : []), [page, data]);
  const cpu = data?.[MACHINE_FILE]?.cpu_info?.cpu as string | undefined;
  const modelPage = (["review", "results", "experiments", "about"] as PageKey[]).includes(page);
  const foot = !modelPage ? `Computed from the recorded result files${cpu ? `; measured on this machine (${cpu})` : ""}.` : `Single seed (473), measured on this machine${cpu ? ` (${cpu})` : ""}. Differences of about 0.01 to 0.02 macro-F1 are within noise.`;

  if (!data) return <div className={clsx("card skeleton", compact ? "h-8" : "h-16")} data-testid="highlights-loading" aria-hidden />;
  if (!items.length) return null;
  if (compact) {
    const line = items.slice(0, 2).map((i) => `${i.label.replace(/^Best /, "").toLowerCase()}: ${i.value}`).join(" · ");
    return (
      <aside className="text-[12.5px]" data-testid="highlights" aria-label="What worked best">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-semibold text-accent">What worked best</span>
          <span className="min-w-0 text-muted sm:truncate" title={items.map((i) => `${i.label}: ${i.value} (${i.criterion})`).join("\n")}>{line}</span>
          <button className="text-accent hover:underline" data-testid="highlights-toggle" aria-expanded={!collapsed} onClick={() => setCollapsed(!collapsed)}>{collapsed ? "Why" : "Hide"}</button>
        </div>
        {!collapsed && (
          <div className="mt-2 space-y-2 rounded-lg border border-line bg-raised/60 p-3" data-testid="highlights-body">
            {items.map((it) => (
              <div key={it.id}>
                <b>{it.label}: {it.value}</b>{it.detail ? <span>. {it.detail}</span> : null}
                <div className="text-[11.5px] italic text-muted">Criterion: {it.criterion}{it.note ? `. ${it.note}` : ""}</div>
              </div>
            ))}
            <p className="text-[11.5px] text-muted">{foot}</p>
          </div>
        )}
      </aside>
    );
  }
  return (
    <aside className="card p-3.5" data-testid="highlights" aria-label="What worked best">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-[13px] font-semibold tracking-tight text-accent">What worked best</h2>
          <p className="text-[12px] text-muted" title="Computed in the browser from the results files">{foot}</p>
        </div>
        <button className="btn shrink-0 px-2 py-1 text-[12px]" data-testid="highlights-toggle" aria-expanded={!collapsed}
          onClick={() => setCollapsed(!collapsed)} title={collapsed ? "Show highlights" : "Hide highlights"}>
          <Icon name={collapsed ? "panelClosed" : "panelOpen"} className="h-3.5 w-3.5" />{collapsed ? "Show" : "Hide"}
        </button>
      </div>
      {!collapsed && (
        <div className={clsx("mt-3 grid gap-2.5 sm:grid-cols-2", items.length >= 3 && "xl:grid-cols-3", items.length >= 4 && "2xl:grid-cols-4")} data-testid="highlights-body">
          {items.map((it) => (
            <div key={it.id} className="min-w-0 rounded-lg border border-line bg-raised/60 p-3" data-testid={`hl-${it.id}`}>
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">{it.label}</div>
              <div className="mt-0.5 break-words text-[14px] font-semibold leading-snug">{it.value}</div>
              {it.detail && <div className="mt-0.5 break-words text-[12.5px]">{it.detail}</div>}
              {it.lines?.map((l, i) => (
                <div key={i} className="mt-1.5 flex items-start gap-1.5 text-[12.5px]">
                  <span className="shrink-0"><Badge tone={TAG_TONE[l.tag]}>{l.tag}</Badge></span><span className="min-w-0 break-words">{l.text}</span>
                </div>
              ))}
              {it.note && <div className="mt-1 break-words text-[12px] text-muted">{it.note}</div>}
              <div className="mt-1.5 break-words text-[11.5px] italic text-muted">Criterion: {it.criterion}</div>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
