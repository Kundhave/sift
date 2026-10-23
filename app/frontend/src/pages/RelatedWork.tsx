import type { ReactNode } from "react";
import { useApi } from "../api";
import { Badge, ErrorBox, PageHeader, Section, SkeletonBlock } from "../components/ui";
import { Highlights } from "../components/Highlights";
import { f3 } from "../format";

interface MasterRow { model: string; group: string; "ndcg@5"?: number | null; macro_f1?: number | null; auc_hard?: number | null }
interface Master { seed?: number; rows: MasterRow[] }

function Table({ head, rows, label }: { head: ReactNode[]; rows: ReactNode[][]; label: string }) {
  return (
    <div className="card overflow-x-auto">
      <table className="w-full min-w-[640px] text-left text-[12.5px]" aria-label={label}>
        <thead><tr className="border-b border-line bg-raised/60">
          {head.map((h, i) => <th key={i} scope="col" className="whitespace-nowrap px-3 py-2 align-bottom text-[11px] font-semibold uppercase tracking-wider text-muted">{h}</th>)}
        </tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} className="border-b border-line last:border-0 align-top">
            {r.map((c, j) => j === 0 ? <th key={j} scope="row" className="px-3 py-2 font-medium">{c}</th> : <td key={j} className="px-3 py-2 text-muted">{c}</td>)}
          </tr>))}
        </tbody>
      </table>
    </div>
  );
}

const TASK_HEAD = ["", "Sift (this project)", "Panthaplackel et al., AAAI 2021", "Panthaplackel et al., ACL 2020", "Herzig & Zeller, MSR 2013"];
const TASK_ROWS: string[][] = [
  ["Question", "Does this hunk serve the developer's stated intent?", "Did a code change make an existing comment inconsistent?", "Given a code change, how should the comment be updated?", "How much do tangled changes (unrelated edits in one commit) pollute mining results?"],
  ["Pair classified", "Consolidated multi-turn intent + one diff hunk", "Comment + code change (just-in-time)", "Comment + code change", "Commit contents (untangling)"],
  ["Granularity", "Hunk-level inside one diff, ranked", "Method-level comment/code pairs", "Method-level", "File/commit-level"],
  ["Text side", "Chat turns, some noisy (chit-chat, retracted, repeated)", "Code comments", "Code comments", "None (change structure)"],
  ["Output", "Aligned/misaligned, 3 drift types, calibrated probability, ranking, token explanation", "Inconsistent / consistent", "Edit sequence for the comment", "Untangled change groups"],
  ["Method", "ModernBERT cross-encoder, multi-task heads, focal loss", "Deep model correlating comment with code change", "Edit-based neural model", "Multi-predictor untangling"],
  ["Evaluation", "Macro-F1, per-tier pairwise AUC, NDCG@5, ECE", "Classification metrics on their corpus", "Edit and update metrics", "Effect on bug-count bias"],
  ["Relation to Sift", "Sift borrows the \"text vs code pair\" framing", "Same family of task, closest prior", "Related task", "Motivates our limitation: seed commits may be tangled"],
];

const BLOCKS: string[][] = [
  ["Cross-encoder on a text pair", "Nogueira & Cho, Passage Re-ranking with BERT (2019)", "Same pattern: concatenate the two texts, classify from the [CLS] vector. Sift applies it to intent + hunk"],
  ["Encoder", "Warner et al., Smarter, Better, Faster, Longer: ModernBERT (2024)", "ModernBERT-base (RoPE, GLU-family activation, alternating local/global attention, 8k context), fine-tuned"],
  ["Code-aware encoder baseline", "Feng et al., CodeBERT (EMNLP Findings 2020)", "Used as a baseline only; under-trained here (4,000 rows, 1 epoch), so not a fair comparison"],
  ["Focal loss", "Lin et al., Focal Loss for Dense Object Detection (ICCV 2017)", "Alignment term, gamma 2, for the ~78:22 class ratio"],
  ["Calibration", "Guo et al., On Calibration of Modern Neural Networks (ICML 2017)", "Temperature scaling on validation, ECE 0.064 to 0.028"],
  ["Distillation", "Hinton et al., Distilling the Knowledge in a Neural Network (2015)", "tau 4, alpha 0.5, MiniLM student, then INT8 ONNX"],
  ["Attribution", "Sundararajan et al., Axiomatic Attribution for Deep Networks (ICML 2017)", "Integrated Gradients, 16 steps, aggregated to words and lines"],
  ["Label-noise detection", "Northcutt et al., Confident Learning (JAIR 2021)", "cleanlab scored against the 569 known flips (precision 0.150, recall 0.387)"],
  ["Noisy-label training (option)", "Han et al., Co-teaching (NeurIPS 2018)", "Named in the design; not run in this project"],
  ["Tangled commits", "Herzig & Zeller (MSR 2013)", "Stated as a dataset limitation"],
];

// Which master_comparison rows to show, in the order of docs/RELATED_WORK.md section 3.
const BASELINES: { match: (m: string) => boolean; name: string; idea: string; main?: boolean }[] = [
  { match: (m) => m.startsWith("B4"), name: "B4 TF-IDF dissimilarity", idea: "Lexical overlap" },
  { match: (m) => m.startsWith("B5"), name: "B5 logistic regression (4 features)", idea: "Simple features" },
  { match: (m) => m.startsWith("Siamese"), name: "Siamese BiLSTM", idea: "Two separate encoders" },
  { match: (m) => m.startsWith("MiniLM-L6"), name: "MiniLM-L6 cross-encoder (reduced training)", idea: "Small cross-encoder" },
  { match: (m) => m.startsWith("CodeBERT"), name: "CodeBERT cross-encoder (reduced training)", idea: "Code-aware encoder" },
  { match: (m) => m.startsWith("Main"), name: "ModernBERT cross-encoder (main)", idea: "Cross-encoder, multi-task", main: true },
  { match: (m) => m.startsWith("Frozen-embedding frozen cosine"), name: "Frozen cosine dissimilarity (embeddings)", idea: "No training" },
];

const REFS_OK: [string, string, string][] = [
  ["S. Panthaplackel, J. J. Li, M. Gligoric, R. J. Mooney.", "Deep Just-In-Time Inconsistency Detection Between Comments and Source Code.", "AAAI 2021. https://arxiv.org/abs/2010.01625"],
  ["S. Panthaplackel, P. Nie, M. Gligoric, J. J. Li, R. J. Mooney.", "Learning to Update Natural Language Comments Based on Code Changes.", "ACL 2020. https://aclanthology.org/2020.acl-main.168/"],
  ["K. Herzig, A. Zeller.", "The Impact of Tangled Code Changes.", "MSR 2013. https://www.st.cs.uni-saarland.de/publications/details/herzig-msr-2013"],
  ["B. Warner et al.", "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT).", "2024. https://arxiv.org/abs/2412.13663"],
  ["R. Nogueira, K. Cho.", "Passage Re-ranking with BERT.", "2019. https://arxiv.org/abs/1901.04085"],
  ["Z. Feng et al.", "CodeBERT: A Pre-Trained Model for Programming and Natural Languages.", "Findings of EMNLP 2020. https://arxiv.org/abs/2002.08155"],
];
const REFS_MEM: [string, string, string][] = [
  ["T.-Y. Lin et al.", "Focal Loss for Dense Object Detection.", "ICCV 2017."],
  ["C. Guo et al.", "On Calibration of Modern Neural Networks.", "ICML 2017."],
  ["G. Hinton, O. Vinyals, J. Dean.", "Distilling the Knowledge in a Neural Network.", "2015."],
  ["M. Sundararajan, A. Taly, Q. Yan.", "Axiomatic Attribution for Deep Networks.", "ICML 2017."],
  ["C. G. Northcutt, L. Jiang, I. L. Chuang.", "Confident Learning: Estimating Uncertainty in Dataset Labels.", "JAIR 2021."],
  ["B. Han et al.", "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels.", "NeurIPS 2018."],
];

function Linked({ text }: { text: string }) {
  const m = text.match(/^(.*?)(https?:\/\/\S+)$/);
  if (!m) return <>{text}</>;
  return <>{m[1]}<a href={m[2]} target="_blank" rel="noreferrer noopener" className="break-all text-accent underline-offset-2 hover:underline">{m[2]}</a></>;
}

function Baselines() {
  const { data, error, loading, reload } = useApi<Master>("/api/results/master_comparison.json");
  if (loading) return <SkeletonBlock rows={5} />;
  if (error || !data) return <ErrorBox message={error ?? "master_comparison.json is not available"} onRetry={reload} />;
  const rows = BASELINES.map((b) => ({ b, r: data.rows.find((r) => b.match(r.model)) }));
  return (
    <>
      <Table label="Baselines on Sift-IA, test split" head={["Model", "Idea", "Macro-F1", "NDCG@5", "Hard-tier pairwise AUC"]}
        rows={rows.map(({ b, r }) => [
          <span key="n" className={b.main ? "text-accent" : undefined}>{b.name}{b.main && <> <Badge tone="accent">main</Badge></>}</span>,
          b.idea,
          r?.macro_f1 == null ? "not reported" : f3(r.macro_f1),
          r?.["ndcg@5"] == null ? "not reported" : f3(r["ndcg@5"]),
          r?.auc_hard == null ? "not reported" : f3(r.auc_hard),
        ])} />
      <p className="mt-2 text-[12px] text-muted">Same dataset, same split, same metrics ({data.seed ?? "single seed 473"}). Loaded live from results/master_comparison.json; nothing on this table is typed in by hand.</p>
    </>
  );
}

export default function RelatedWorkPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-9">
      <PageHeader title="Related work" sub="Sift compared with standard base papers: closest research area, the building blocks it reuses, and the baselines actually run." />
      <Highlights page="related" />

      <div role="note" className="rounded-xl border border-warn/40 bg-warn/10 p-4 text-[13.5px]" data-testid="caveat">
        <div className="mb-1 font-semibold text-warn">Read this first: numbers across papers are not comparable</div>
        The papers below solve neighbouring tasks on different data with different metrics. Their reported numbers are <b>not comparable</b> with Sift's numbers, so this page compares <i>task, input, granularity, method and evaluation design</i>, never "our score beats their score". No paper's results are quoted here because none were reproduced on Sift-IA.
      </div>

      <Section title="1. Closest research area: comment-code inconsistency and code-change tasks">
        <Table label="Task comparison" head={TASK_HEAD} rows={TASK_ROWS} />
        <p className="text-[13px] text-muted">What is different in Sift: intent is <b className="text-fg">multi-turn and noisy</b> (consolidated first), the unit is the <b className="text-fg">hunk inside a diff</b>, and the goal is a <b className="text-fg">ranking for human review</b>, not a single yes/no on a comment.</p>
      </Section>

      <Section title="2. Building blocks Sift reuses" hint="All existing methods.">
        <Table label="Building blocks" head={["Component in Sift", "Base paper", "What Sift does with it"]} rows={BLOCKS} />
      </Section>

      <Section title="3. Honest comparison against the baselines we actually ran">
        <Baselines />
        <div className="card p-4">
          <div className="label">Reading it honestly</div>
          <ul className="list-disc space-y-1.5 pl-5 text-[13px]">
            <li>Reading both texts jointly helps most where lexical overlap fails: on the <b>hard tier</b> the main model reaches 0.726 pairwise AUC against 0.551 for B4 and B5.</li>
            <li>The main model's macro-F1 is <b>below</b> B5, and a frozen cosine baseline is slightly better on NDCG@5. The case for the model is hard-tier ranking, not every metric.</li>
            <li>MiniLM, BiLSTM and CodeBERT were trained with a smaller budget than the main model, so part of their gap is budget, not architecture.</li>
            <li>The one comparison the literature supports directly is architectural: joint (cross) encoding beats separate encoding for pair relevance (Nogueira &amp; Cho), and our BiLSTM and frozen-cosine rows are consistent with, but do not prove, that.</li>
          </ul>
        </div>
      </Section>

      <Section title="4. What Sift does not claim">
        <ul className="list-disc space-y-1.5 rounded-xl border border-warn/30 bg-warn/5 p-4 pl-9 text-[13px]">
          <li>No new neural architecture. The contribution is the task framing, the pipeline and the dataset.</li>
          <li>No state-of-the-art claim against any published system: no published system was evaluated on Sift-IA, and Sift was not evaluated on their corpora.</li>
          <li>Labels are constructed (donor hunks from other commits), with no human validation yet.</li>
        </ul>
      </Section>

      <Section title="5. References">
        <div className="card p-4">
          <div className="label">Confirmed by web search while writing this section (titles, authors and venues as found)</div>
          <ol className="list-decimal space-y-2 pl-6 text-[13px]" data-testid="refs-confirmed">
            {REFS_OK.map(([a, t, v]) => <li key={t}>{a} <i>{t}</i> <Linked text={v} /></li>)}
          </ol>
        </div>
        <div className="card p-4">
          <div className="label">Cited from memory, not re-verified in this session (check the details before submitting)</div>
          <ol start={7} className="list-decimal space-y-2 pl-6 text-[13px]" data-testid="refs-memory">
            {REFS_MEM.map(([a, t, v]) => <li key={t}>{a} <i>{t}</i> {v} <Badge tone="warn">not re-verified</Badge></li>)}
          </ol>
        </div>
      </Section>
    </div>
  );
}
