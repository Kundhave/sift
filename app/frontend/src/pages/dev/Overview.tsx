import { Link } from "react-router-dom";
import { LIMITS, Limits, PageHeader, Section } from "../../components/ui";
import { Highlights } from "../../components/Highlights";
import { Architecture } from "../../components/Architecture";

const PAGES = [
  ["/dev/results", "Results", "Master comparison, per-tier AUC, ROC/PR, calibration, latency and size."],
  ["/dev/experiments", "Experiments", "Ablation, noise robustness, confident learning, clustering, XAI."],
  ["/dev/dataset", "Dataset", "Sift-IA v1.0 statistics, splits, noise manifest."],
  ["/dev/testing", "Testing", "Test totals, coverage, categories and the defect register."],
  ["/dev/related-work", "Related work", "Baselines on Sift-IA and the literature, with verification status."],
];

export default function DevOverview() {
  return (
    <div className="space-y-7">
      <PageHeader title="Overview" sub="Everything measured for Sift, computed from the result files in results/. Single seed (473); see each page for caveats." />
      <Highlights page="about" />
      <Section title="Console pages">
        <div className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
          {PAGES.map(([to, t, d]) => (
            <Link key={to} to={to} className="card p-3.5 transition-colors hover:bg-raised">
              <div className="font-semibold">{t}</div><p className="mt-0.5 text-[12.5px] text-muted">{d}</p>
            </Link>
          ))}
        </div>
      </Section>
      <Section title="Pipeline"><Architecture /></Section>
      <Limits items={[...LIMITS, "Seed commits may be tangled, and a donor hunk may coincidentally suit the intent.", "Only Python, JavaScript and TypeScript are covered.", "Attribution highlights are evidence about the model, not proof of the cause."]} />
    </div>
  );
}
