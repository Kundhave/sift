const BOXES = [
  { x: 10, label: "Agent output", sub: "diff + chat turns" },
  { x: 150, label: "Session capture", sub: "multi-turn intent" },
  { x: 290, label: "Intent consolidation", sub: "drop chit-chat, retractions" },
  { x: 430, label: "Diff parser", sub: "hunk segmentation" },
  { x: 570, label: "Sift model", sub: "ModernBERT cross-encoder" },
  { x: 710, label: "Calibrate + rank", sub: "temperature, threshold" },
  { x: 850, label: "Triage UI", sub: "this app" },
];

export function Architecture() {
  return (
    <div className="card overflow-x-auto p-4">
      <svg viewBox="0 0 990 250" className="min-w-[760px]" role="img" aria-label="Sift pipeline: agent output, session capture, intent consolidation, diff parser, Sift model, calibration and ranking, triage UI">
        <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="rgb(var(--muted))" /></marker></defs>
        {BOXES.map((b, i) => (
          <g key={b.label}>
            <rect x={b.x} y={30} width={120} height={72} rx={10} fill="rgb(var(--raised))" stroke={i === 4 ? "rgb(var(--accent))" : "rgb(var(--line))"} strokeWidth={i === 4 ? 2 : 1} />
            <text x={b.x + 60} y={62} textAnchor="middle" fontSize="12.5" fontWeight="600" fill="rgb(var(--fg))">{b.label}</text>
            <text x={b.x + 60} y={82} textAnchor="middle" fontSize="10" fill="rgb(var(--muted))">{b.sub}</text>
            {i < BOXES.length - 1 && <line x1={b.x + 121} y1={66} x2={b.x + 139} y2={66} stroke="rgb(var(--muted))" strokeWidth="1.5" markerEnd="url(#ar)" />}
          </g>
        ))}
        <rect x={430} y={150} width={400} height={78} rx={10} fill="none" stroke="rgb(var(--line))" strokeDasharray="5 4" />
        <text x={630} y={170} textAnchor="middle" fontSize="11" fontWeight="600" fill="rgb(var(--fg))">Heads on the pooled [CLS] representation</text>
        {["Alignment (binary, focal)", "Drift type (3-way)", "Confidence"].map((t, i) => (
          <g key={t}><rect x={444 + i * 130} y={184} width={122} height={32} rx={7} fill="rgb(var(--accent) / 0.12)" stroke="rgb(var(--accent) / 0.4)" />
            <text x={505 + i * 130} y={204} textAnchor="middle" fontSize="10" fill="rgb(var(--fg))">{t}</text></g>
        ))}
        <line x1={630} y1={102} x2={630} y2={148} stroke="rgb(var(--muted))" strokeDasharray="3 3" />
        <text x={10} y={150} fontSize="11" fill="rgb(var(--muted))">Input to the model: [CLS] consolidated_intent [SEP] hunk_text</text>
        <text x={10} y={170} fontSize="11" fill="rgb(var(--muted))">Everything runs locally; no code leaves the machine.</text>
      </svg>
    </div>
  );
}
