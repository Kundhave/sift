import { useState } from "react";
import type { ResultFile } from "../types";

export function Figure({ file }: { file: ResultFile }) {
  const [bad, setBad] = useState(false);
  const name = file.path.split("/").pop()!.replace(/\.[a-z]+$/i, "").replace(/[_-]+/g, " ");
  return (
    <figure className="card overflow-hidden">
      {bad ? <div className="p-6 text-center text-[13px] text-muted">Image failed to load.</div>
        : <img src={file.url} alt={name} loading="lazy" onError={() => setBad(true)} className="w-full bg-white object-contain" />}
      <figcaption className="border-t border-line px-3 py-2 text-[12px] capitalize text-muted">{name}</figcaption>
    </figure>
  );
}

export function Gallery({ files }: { files: ResultFile[] }) {
  return <div className="grid gap-4 md:grid-cols-2">{files.map((f) => <Figure key={f.path} file={f} />)}</div>;
}
