import type { PlayerStats as Stats } from "./api";

const COLUMNS = [
  ["carries", "CAR", "Carries"],
  ["rushing_yards", "YDS", "Rushing yards"],
  ["rushing_tds", "TD", "Rushing touchdowns"],
  ["targets", "TGT", "Targets"],
  ["receptions", "REC", "Receptions"],
  ["receiving_yards", "YDS", "Receiving yards"],
  ["receiving_tds", "TD", "Receiving touchdowns"],
] as const;

export default function PlayerStats({ stats, home, away }: { stats: Stats; home: string; away: string }) {
  return (
    <section className="card player-stats" aria-labelledby="player-stats-title">
      <h2 id="player-stats-title">Player stats</h2>
      <p className="muted small">Simulated rushing and receiving production. Updates after each play.</p>
      {([["away", away], ["home", home]] as const).map(([side, team]) => (
        <div key={side} className="player-stats-team">
          <h3>{team} <span className="muted small">{side === "away" ? "Away" : "Home"}</span></h3>
          {stats[side].length === 0 ? <p className="muted small">No carries or targets yet.</p> : (
            <div className="player-stats-scroll" role="region" aria-label={`${team} player stats`} tabIndex={0}>
              <table className="results-table">
                <caption className="sr-only">{team} player rushing and receiving stats</caption>
                <thead>
                  <tr><th scope="col" rowSpan={2}>Player</th>
                    <th scope="colgroup" colSpan={3}>Rushing</th><th scope="colgroup" colSpan={4}>Receiving</th></tr>
                  <tr>{COLUMNS.map(([key, label, title]) => <th key={key} scope="col"><abbr title={title}>{label}</abbr></th>)}</tr>
                </thead>
                <tbody>{stats[side].map((p) => (
                  <tr key={p.slot}>
                    <th scope="row">{p.name} <span className="muted small">{p.role}</span></th>
                    {COLUMNS.map(([key]) => <td key={key}>{p[key]}</td>)}
                  </tr>
                ))}</tbody>
                <tfoot><tr><th scope="row">Total</th>
                  {COLUMNS.map(([key]) => <td key={key}>{stats[side].reduce((sum, p) => sum + p[key], 0)}</td>)}
                </tr></tfoot>
              </table>
            </div>
          )}
        </div>
      ))}
      <p className="muted small">Players without a name are labeled by position. Yardage on plays ending in a lost fumble is unavailable.</p>
    </section>
  );
}
