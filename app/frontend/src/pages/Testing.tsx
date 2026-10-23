import { useState } from "react";
import { useResultIndex, useApi } from "../api";
import { DataTree } from "../components/DataTree";
import { JsonResult } from "../components/JsonResult";
import { ErrorBox, NotAvailable, PageHeader, Section, SkeletonBlock, Stat } from "../components/ui";
import { Highlights } from "../components/Highlights";

interface TestsResp { available: boolean; reason?: string; source?: string; summary?: any; coverage?: any }

function findDefects(o: any): any[] | null {
  if (!o || typeof o !== "object") return null;
  for (const [k, v] of Object.entries(o)) {
    if (/defect/i.test(k) && Array.isArray(v) && v.length && typeof v[0] === "object") return v as any[];
  }
  for (const v of Object.values(o)) { const r = typeof v === "object" ? findDefects(v) : null; if (r) return r; }
  return null;
}

export default function TestingPage() {
  const t = useApi<TestsResp>("/api/tests");
  const [raw, setRaw] = useState(false);
  const { files } = useResultIndex();
  const hasMeas = !!files?.some((f) => f.path === "testing_measurements.json");
  const s = t.data?.summary;
  const defects = findDefects(s);
  const totals = s?.totals ?? s?.total ?? null;
  const covTotal = t.data?.coverage?.totals?.percent_covered ?? t.data?.coverage?.percent_covered ?? s?.coverage?.percent_covered ?? null;

  return (
    <div className="space-y-8">
      <PageHeader title="Testing" sub="Test results by category, coverage and the defect register. Aligned with the six sections of the DoD CDAO test-and-evaluation framework." />
      <Highlights page="testing" />
      {t.loading && <SkeletonBlock rows={5} />}
      {t.error && <ErrorBox message={t.error} onRetry={t.reload} />}
      {t.data && !t.data.available && (
        <NotAvailable what="Test summary and coverage" expects="test_summary.json" />
      )}
      {t.data?.available && (
        <>
          <p className="text-[12px] text-muted">Source: <code className="font-mono">{t.data.source}</code></p>
          {totals && typeof totals === "object" && (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
              {Object.entries(totals).filter(([, v]) => typeof v === "number").map(([k, v]) => <Stat key={k} label={k.replace(/_/g, " ")} value={(v as number).toLocaleString()} />)}
              {typeof covTotal === "number" && <Stat label="Line coverage" value={`${covTotal.toFixed(1)}%`} />}
            </div>
          )}
          {s?.categories && typeof s.categories === "object" && (
            <Section title="Results by category">
              <div className="card overflow-x-auto">
                <table className="w-full text-left text-[12.5px]">
                  <thead className="border-b border-line text-[11px] uppercase tracking-wider text-muted">
                    <tr><th className="p-2.5">Category</th><th className="p-2.5">Framework section</th><th className="p-2.5 text-right">Passed</th><th className="p-2.5 text-right">xfailed</th><th className="p-2.5 text-right">Failed</th><th className="p-2.5 text-right">Total</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries<any>(s.categories).map(([k, c]) => (
                      <tr key={k} className="border-b border-line/60 last:border-0">
                        <td className="p-2.5 font-medium">{k}</td><td className="p-2.5 text-muted">{c.framework_section ?? ""}</td>
                        {["passed", "xfailed", "failed", "total"].map((f) => <td key={f} className="p-2.5 text-right tabular-nums">{c[f] ?? "-"}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}
          <Section title="Defect register" hint="Only defects the tests actually found. Known ones are strict expected-failure cases.">
            {defects ? (
              <div className="card overflow-x-auto">
                <table className="w-full text-left text-[12.5px]">
                  <thead className="border-b border-line text-[11px] uppercase tracking-wider text-muted">
                    <tr><th className="p-2.5">ID</th><th className="p-2.5">Description</th><th className="p-2.5">Severity</th><th className="p-2.5">How found</th><th className="p-2.5">Mitigation</th></tr>
                  </thead>
                  <tbody>
                    {defects.map((d, i) => (
                      <tr key={d.id ?? i} className="border-b border-line/60 align-top last:border-0">
                        <td className="whitespace-nowrap p-2.5 font-mono">{d.id}</td><td className="p-2.5">{d.description}</td>
                        <td className="p-2.5">{d.severity}</td><td className="p-2.5 text-muted">{d.how_found}</td><td className="p-2.5 text-muted">{d.mitigation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <NotAvailable what="Defect register" expects="a `defects` list in test_summary.json" />}
          </Section>
          <div>
            <button className="btn" onClick={() => setRaw((r) => !r)} aria-expanded={raw} data-testid="raw-toggle">{raw ? "Hide" : "Show"} raw data</button>
            {raw && (
              <div className="mt-3 space-y-4" data-testid="raw-tree">
                <div className="card p-4"><DataTree value={s} maxRows={60} /></div>
                {t.data.coverage && <div className="card p-4"><DataTree value={t.data.coverage.files ? Object.fromEntries(Object.entries<any>(t.data.coverage.files).map(([k, v]) => [k, v.summary ?? v])) : t.data.coverage} /></div>}
              </div>
            )}
          </div>
        </>
      )}
      {hasMeas && (
        <Section title="Measured robustness checks" hint="Metamorphic relations and adversarial attacks run against the main model (testing_measurements.json).">
          <JsonResult path="testing_measurements.json" />
        </Section>
      )}
    </div>
  );
}
