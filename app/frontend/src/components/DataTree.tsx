import { fmtVal, label } from "../format";

type J = unknown;
const isObj = (v: J): v is Record<string, J> => typeof v === "object" && v !== null && !Array.isArray(v);
const isScalar = (v: J) => v == null || typeof v !== "object";

function Table({ cols, rows }: { cols: string[]; rows: (J[])[] }) {
  return (
    <div className={"overflow-x-auto rounded-lg border border-line" + (rows.length > 12 ? " max-h-[28rem] overflow-y-auto" : "")}>
      <table className="w-full text-[12.5px] tabular-nums">
        <thead className="sticky top-0 bg-raised text-left text-muted">
          <tr>{cols.map((c) => <th key={c} className="whitespace-nowrap px-2.5 py-1.5 font-medium">{label(c)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line">
              {r.map((c, j) => <td key={j} className={"px-2.5 py-1.5 align-top " + (j === 0 ? "whitespace-nowrap font-medium " : "") + (typeof c === "string" && c.length > 40 ? "min-w-[15rem] " : "")}>{isScalar(c) ? fmtVal(c) : <DataTree value={c} depth={9} />}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Generic, honest renderer for any results JSON: shows exactly what the file contains. */
export function DataTree({ value, depth = 0, maxRows = 40 }: { value: J; depth?: number; maxRows?: number }) {
  if (isScalar(value)) return <span>{fmtVal(value)}</span>;
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-muted">empty</span>;
    if (value.every(isScalar)) {
      const v = value.slice(0, 30).map((x) => fmtVal(x)).join(", ");
      return <span className="font-mono text-[12px]">[{v}{value.length > 30 ? `, ... (${value.length} values)` : ""}]</span>;
    }
    if (value.every(isObj)) {
      const rows = value as Record<string, J>[];
      const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(r)))).filter((c) => rows.some((r) => isScalar(r[c]))).slice(0, 14);
      return (
        <div className="space-y-1">
          <Table cols={cols} rows={rows.slice(0, maxRows).map((r) => cols.map((c) => r[c]))} />
          {rows.length > maxRows && <div className="text-[12px] text-muted">Showing {maxRows} of {rows.length} rows.</div>}
        </div>
      );
    }
    return <span className="text-muted">{value.length} items</span>;
  }
  const entries = Object.entries(value as Record<string, J>);
  const scalars = entries.filter(([, v]) => isScalar(v));
  const nested = entries.filter(([, v]) => !isScalar(v));
  // object of same-shaped objects with scalar fields -> one comparison table
  if (scalars.length === 0 && nested.length > 1 && nested.every(([, v]) => isObj(v))) {
    const cols = Array.from(new Set(nested.flatMap(([, v]) => Object.entries(v as Record<string, J>).filter(([, x]) => isScalar(x)).map(([k]) => k))));
    if (cols.length >= 2) {
      return <Table cols={["", ...cols]} rows={nested.map(([k, v]) => [label(k), ...cols.map((c) => (v as Record<string, J>)[c])])} />;
    }
  }
  return (
    <div className={"space-y-2.5 " + (depth > 0 ? "border-l border-line pl-3" : "")}>
      {scalars.length > 0 && (
        <dl className="grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-x-4 gap-y-1.5 text-[12.5px]">
          {scalars.map(([k, v]) => (
            typeof v === "string" && v.length > 36 ? (
              <div key={k} className="col-span-full border-b border-line/60 pb-1">
                <dt className="text-muted">{label(k)}</dt><dd className="break-words font-medium">{v}</dd>
              </div>
            ) : (
              <div key={k} className="flex items-baseline justify-between gap-2 border-b border-line/60 pb-1">
                <dt className="text-muted">{label(k)}</dt><dd className="break-words text-right font-medium tabular-nums">{fmtVal(v)}</dd>
              </div>
            )
          ))}
        </dl>
      )}
      {nested.map(([k, v]) => (
        <details key={k} open={depth < 1} className="group">
          <summary className="cursor-pointer select-none text-[12.5px] font-semibold text-fg/90 hover:text-accent">{label(k)}</summary>
          <div className="mt-1.5">{depth > 4 ? <span className="text-muted">...</span> : <DataTree value={v} depth={depth + 1} maxRows={maxRows} />}</div>
        </details>
      ))}
    </div>
  );
}
