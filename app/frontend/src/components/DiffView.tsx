import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import javascript from "highlight.js/lib/languages/javascript";
import typescript from "highlight.js/lib/languages/typescript";
import json from "highlight.js/lib/languages/json";
import yaml from "highlight.js/lib/languages/yaml";
import markdown from "highlight.js/lib/languages/markdown";
import xml from "highlight.js/lib/languages/xml";
import css from "highlight.js/lib/languages/css";
import bash from "highlight.js/lib/languages/bash";
import { useMemo } from "react";
import clsx from "clsx";

hljs.registerLanguage("python", python); hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("typescript", typescript); hljs.registerLanguage("json", json);
hljs.registerLanguage("yaml", yaml); hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("xml", xml); hljs.registerLanguage("css", css); hljs.registerLanguage("bash", bash);

const EXT: Record<string, string> = {
  py: "python", js: "javascript", mjs: "javascript", cjs: "javascript", jsx: "javascript", ts: "typescript", tsx: "typescript",
  json: "json", yml: "yaml", yaml: "yaml", md: "markdown", html: "xml", xml: "xml", css: "css", sh: "bash", toml: "yaml", cfg: "yaml",
};
export const langOf = (file: string) => EXT[file.split(".").pop()?.toLowerCase() ?? ""];

function hl(code: string, lang?: string): string {
  if (!lang) return code.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!));
  try { return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value; }
  catch { return code.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!)); }
}

/** Unified-diff hunk with per-line syntax highlighting, +/- gutters, and optional per-line attribution shading. */
export function DiffView({ text, file, lineScores, maxLines }: { text: string; file: string; lineScores?: Map<string, number>; maxLines?: number }) {
  const lang = langOf(file);
  const rows = useMemo(() => {
    const ls = text.replace(/\n$/, "").split("\n");
    return ls.map((l) => {
      const kind = l.startsWith("@@") ? "hdr" : l[0] === "+" ? "add" : l[0] === "-" ? "del" : "ctx";
      const body = kind === "hdr" ? l : l.slice(1);
      return { raw: l, kind, html: kind === "hdr" ? "" : hl(body, lang) };
    });
  }, [text, lang]);
  const shown = maxLines ? rows.slice(0, maxLines) : rows;
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-surface font-mono text-[12px] leading-[1.55]" role="region" aria-label={`Diff of ${file}`}>
      <table className="w-full border-collapse">
        <tbody>
          {shown.map((r, i) => {
            const s = lineScores?.get(r.raw);
            const bg = s == null ? undefined : s >= 0 ? `rgb(var(--bad) / ${Math.min(0.5, Math.abs(s) * 0.5)})` : `rgb(var(--accent) / ${Math.min(0.5, Math.abs(s) * 0.5)})`;
            return (
              <tr key={i} style={bg ? { boxShadow: `inset 0 0 0 999px ${bg}` } : undefined}
                className={clsx(r.kind === "add" && "bg-ok/10", r.kind === "del" && "bg-bad/10", r.kind === "hdr" && "bg-raised text-muted")}>
                <td className={clsx("w-5 select-none px-1.5 text-center", r.kind === "add" && "text-ok", r.kind === "del" && "text-bad", r.kind !== "add" && r.kind !== "del" && "text-muted")}>
                  {r.kind === "add" ? "+" : r.kind === "del" ? "-" : ""}
                </td>
                <td className="whitespace-pre pr-3">
                  {r.kind === "hdr" ? r.raw : <span dangerouslySetInnerHTML={{ __html: r.html || "&nbsp;" }} />}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {maxLines && rows.length > maxLines && <div className="border-t border-line bg-raised px-3 py-1 text-[11px] text-muted">{rows.length - maxLines} more lines</div>}
    </div>
  );
}
