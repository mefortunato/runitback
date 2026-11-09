import { pct } from "../lib/format";
import type { Team } from "../lib/types";
import s from "./ProbabilityCompare.module.css";

// Win probability bars for the model and Vegas (no-vig), away on the left.
export default function ProbabilityCompare({ home, away, model, market = null }: {
  home: Team; away: Team; model: number; market?: number | null;
}) {
  const rows = [{ label: "Model", p: model }, { label: "Vegas", p: market }]
    .filter((r) => r.p !== null && r.p !== undefined) as { label: string; p: number }[];
  return (
    <div className={s.compare} aria-label="Win probability, model and Vegas">
      {rows.map((row) => (
        <div key={row.label} className={`${s.row} small`}>
          <span className={`${s.label} muted`}>{row.label}</span>
          <span className="tabular">{away.team} {pct(1 - row.p)}</span>
          <div className={s.bar}>
            <div style={{ width: `${(1 - row.p) * 100}%`, background: away.color ?? "var(--muted)" }} />
            <div style={{ width: `${row.p * 100}%`, background: home.color ?? "var(--accent)" }} />
          </div>
          <span className="tabular">{pct(row.p)} {home.team}</span>
        </div>
      ))}
    </div>
  );
}
