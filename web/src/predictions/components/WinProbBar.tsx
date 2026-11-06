import { pct } from "../lib/format";
import type { Team } from "../lib/types";
import s from "./WinProbBar.module.css";

/** `probability`: P(home team wins). Away on the left, home on the right. */
export default function WinProbBar({ home, away, probability }: { home: Team; away: Team; probability: number }) {
  return (
    <div className={s.wrap} aria-label="Win probability">
      <div className={`${s.labels} small`}>
        <span><strong>{away.team}</strong> {pct(1 - probability)}</span>
        <span className="muted">win probability</span>
        <span>{pct(probability)} <strong>{home.team}</strong></span>
      </div>
      <div className={s.bar}>
        <div style={{ width: `${(1 - probability) * 100}%`, background: away.color ?? "var(--muted)" }} />
        <div style={{ width: `${probability * 100}%`, background: home.color ?? "var(--accent)" }} />
      </div>
    </div>
  );
}
