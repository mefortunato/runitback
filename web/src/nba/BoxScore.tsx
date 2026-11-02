import { qtrName } from "../format";
import type { Play, State } from "./api";
import { isMade } from "./api";

interface Line { fgm: number; fga: number; tpm: number; tpa: number; ftm: number; fta: number; oreb: number; tov: number }

// Line score by period and team totals, computed from the event log.
export default function BoxScore({ s, plays }: { s: State; plays: Play[] }) {
  const periods = Math.max(4, s.qtr);
  const pts: Record<string, number[]> = { [s.away]: Array(periods).fill(0), [s.home]: Array(periods).fill(0) };
  const blank = (): Line => ({ fgm: 0, fga: 0, tpm: 0, tpa: 0, ftm: 0, fta: 0, oreb: 0, tov: 0 });
  const st: Record<string, Line> = { [s.away]: blank(), [s.home]: blank() };
  for (const p of plays) {
    const b = p.before, a = p.after, q = b.qtr - 1;
    pts[b.home][q] += a.home_score - b.home_score;
    pts[b.away][q] += a.away_score - b.away_score;
    const t = st[b.pos], { play_type: type, result } = p.outcome;
    if (type === "shot2" || type === "shot3") {
      t.fga++; if (isMade(result)) t.fgm++;
      if (type === "shot3") { t.tpa++; if (isMade(result)) t.tpm++; }
    }
    if (type === "free_throw") { t.fta++; if (result === "made") t.ftm++; }
    if (result === "missed_off") t.oreb++;
    if (type === "turnover") t.tov++;
  }
  const teams = [s.away, s.home];
  const total = (team: string) => pts[team].reduce((x, y) => x + y, 0);
  const frac = (m: number, a: number) => `${m}-${a}`;
  const rate = (m: number, a: number) => (a ? `${Math.round((m / a) * 100)}%` : "–");
  return (
    <div className="boxscore">
      <table className="results-table line-score">
        <thead><tr><th scope="col">Team</th>
          {Array.from({ length: periods }, (_, i) => <th key={i} scope="col">{qtrName(i + 1)}</th>)}
          <th scope="col">T</th></tr></thead>
        <tbody>{teams.map((team) => (
          <tr key={team}><th scope="row">{team}</th>
            {pts[team].map((v, i) => <td key={i} className={i + 1 > s.qtr ? "muted" : ""}>{i + 1 > s.qtr ? "" : v}</td>)}
            <td className="game-score">{total(team)}</td></tr>
        ))}</tbody>
      </table>
      <table className="results-table team-stats">
        <thead><tr><th scope="col">Team</th><th scope="col">FG</th><th scope="col">FG%</th><th scope="col">3PT</th>
          <th scope="col">3P%</th><th scope="col">FT</th><th scope="col">OREB</th><th scope="col">TOV</th></tr></thead>
        <tbody>{teams.map((team) => {
          const t = st[team];
          return (
            <tr key={team}><th scope="row">{team}</th><td>{frac(t.fgm, t.fga)}</td><td>{rate(t.fgm, t.fga)}</td>
              <td>{frac(t.tpm, t.tpa)}</td><td>{rate(t.tpm, t.tpa)}</td><td>{frac(t.ftm, t.fta)}</td>
              <td>{t.oreb}</td><td>{t.tov}</td></tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}
