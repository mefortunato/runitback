import MarginChart from "./MarginChart";
import { pct, type SimResult } from "../format";

// Result tiles plus the margin histogram for an N-game matchup forecast.
export default function Forecast({ r, ticks, showTie = false }: { r: SimResult; ticks?: number[]; showTie?: boolean }) {
  return (
    <section className="card forecast" aria-label="Matchup forecast">
      <div className="tiles">
        <div className="tile"><div className="muted">{r.away} win</div><div className="big">{pct(r.away_win)}</div></div>
        <div className="tile"><div className="muted">{r.home} win</div><div className="big">{pct(r.home_win)}</div></div>
        <div className="tile"><div className="muted">Avg score</div><div className="big">{r.away_score_mean.toFixed(1)}–{r.home_score_mean.toFixed(1)}</div></div>
        <div className="tile"><div className="muted">Avg total{showTie && r.tie > 0 ? ` · tie ${pct(r.tie)}` : ""}</div><div className="big">{r.total_mean.toFixed(1)}</div></div>
      </div>
      <MarginChart r={r} ticks={ticks} />
    </section>
  );
}
