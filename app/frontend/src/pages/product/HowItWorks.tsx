import { Link } from "react-router-dom";
import { Architecture } from "../../components/Architecture";
import { Icon } from "../../components/ui";

const STEPS = [
  ["Capture the session", "Sift takes the conversation you had with the agent, as one or many turns, plus the diff it produced."],
  ["Consolidate the intent", "Chit-chat, repeated turns and requests you later retracted are dropped. What remains is the intent: what you actually wanted."],
  ["Split the diff into hunks", "Every change is cut into hunks, the small blocks you would normally review one at a time."],
  ["Read intent and hunk together", "A ModernBERT cross-encoder reads the intent and a single hunk jointly. It outputs a probability that the hunk is misaligned, a drift type and a confidence."],
  ["Calibrate and rank", "Probabilities are temperature-scaled. Hunks are sorted so the likeliest misaligned ones come first."],
  ["Explain", "On request, Integrated Gradients highlight which intent words and diff lines pushed the verdict."],
];

export default function HowItWorks() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16 sm:px-6 sm:py-20">
      <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">How Sift works</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">Six small steps, all on your machine. Nothing is sent anywhere.</p>
      <ol className="mt-12 space-y-8">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="flex gap-5">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-line bg-surface font-mono text-[13px] text-accent">{i + 1}</span>
            <div><h2 className="text-lg font-semibold tracking-tight">{t}</h2><p className="mt-1 leading-relaxed text-muted">{d}</p></div>
          </li>
        ))}
      </ol>
      <h2 className="mb-4 mt-16 text-2xl font-semibold tracking-tight">The pipeline</h2>
      <Architecture />
      <div className="mt-12 flex flex-wrap gap-3">
        <Link to="/app" className="btn btn-primary btn-lg">Try Sift<Icon name="arrow" className="h-4 w-4" /></Link>
        <Link to="/about" className="btn btn-lg">About Sift</Link>
      </div>
    </div>
  );
}
