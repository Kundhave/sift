import { Link } from "react-router-dom";
import { ReactNode } from "react";
import { Architecture } from "../../components/Architecture";

function H({ children }: { children: ReactNode }) { return <h2 className="mb-3 mt-14 text-2xl font-semibold tracking-tight">{children}</h2>; }
const P = ({ children }: { children: ReactNode }) => <p className="leading-relaxed text-muted">{children}</p>;

export default function About() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16 sm:px-6 sm:py-20">
      <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">About Sift</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">A local verification layer for AI coding agents.</p>

      <H>The problem</H>
      <P>An AI coding agent hands you a diff. You have to decide, hunk by hunk, whether each change is what you asked for. Most are. A few are not, and they hide in the noise. Sift reads your intent, scores every hunk, and puts the likeliest problems first.</P>

      <H>Who it is for</H>
      <P>Developers who work with Claude Code, Cursor, Copilot and similar agents, and reviewers who inherit the result.</P>

      <H>Three kinds of drift</H>
      <div className="grid gap-4 sm:grid-cols-3">
        {[["Collateral", "An unrequested change next to the requested one, like reformatting a module while fixing one function."],
          ["Scope-Expansion", "An extra feature or behaviour nobody asked for."],
          ["Purpose-Drift", "A change that serves a different purpose than the one you described."]].map(([t, d]) => (
          <div key={t} className="card p-4"><div className="font-semibold text-accent">{t}</div><p className="mt-1.5 text-[14px] leading-relaxed text-muted">{d}</p></div>
        ))}
      </div>

      <H>Architecture</H>
      <Architecture />

      <H>Private by design</H>
      <P>Models run on your own CPU. Your code and conversation are never uploaded, and there is no account.</P>

      <H>Why it matters</H>
      <P><b className="text-fg">SDG 9, Industry, Innovation and Infrastructure:</b> more trustworthy AI-assisted software, on local hardware. <b className="text-fg">SDG 8, Decent Work and Economic Growth:</b> reviewer attention goes to the riskiest changes first.</P>

      <H>What is and is not new</H>
      <P>Sift is built on existing models. The contribution is the task framing (hunk-level checking against multi-turn intent), the pipeline, and the Sift-IA dataset. No new neural architecture is claimed.</P>

      <H>Honest limitations</H>
      <ul className="list-disc space-y-2 pl-5 leading-relaxed text-muted">
        <li>Training labels come from how the dataset was built, not from human reviewers. Misaligned hunks are borrowed from other commits.</li>
        <li>Only Python, JavaScript and TypeScript, and only open-source repositories, were tested.</li>
        <li>A model can pick up quirks of how the data was built. A harder test tier reduces this but does not remove it.</li>
        <li>Sift ranks hunks for a human. It does not replace review, and explanations are evidence, not proof.</li>
        <li>Speed and size are from one machine.</li>
      </ul>
      <p className="mt-12 text-[14px] text-muted">Metrics and experiments live in the <Link className="text-accent hover:underline" to="/dev">developer console</Link>.</p>
    </div>
  );
}
