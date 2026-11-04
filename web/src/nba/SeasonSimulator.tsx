import { useEffect, useRef, useState } from "react";
import { errorText } from "../http";
import { api, seasonLabel, type SeasonGame, type SeasonJob, type SeasonRequest, type Series, type Standing } from "./api";

type StrengthMode = SeasonRequest["strength_mode"];

// The 30-team, six-division alignment (2004-05 onward); seasons are named by the year they end.
const FIRST_SEASON = 2005;
const LAST_SEASON = 2040;
const FIRST_PLAY_IN_SEASON = 2021;
const REGULAR_GAMES = 1230;
const totalSlots = (season: number) => REGULAR_GAMES + (season >= FIRST_PLAY_IN_SEASON ? 6 : 0) + 15 * 7;

const pct = (p: number) => p.toFixed(3).replace(/^0/, "");
const signed = (n: number) => (n > 0 ? `+${n}` : String(n));

function StandingsTable({ rows, playIn }: { rows: Standing[]; playIn: boolean }) {
  const cutoffs = new Set(playIn ? [6, 10] : [8]);
  return (
    <table className="standings-table">
      <thead><tr><th>Team</th><th>W-L</th><th>PCT</th><th>GB</th><th className="wide-only">Home</th><th className="wide-only">Away</th>
        <th>Conf</th><th>Diff</th><th>L10</th><th>Strk</th></tr></thead>
      <tbody>{rows.map((t) => (
        <tr key={t.team} className={`${t.status === "playoffs" ? "playoff-qualifier" : ""} ${t.status === "play_in" ? "play-in-team" : ""} ${cutoffs.has(t.seed) ? "seed-cutoff" : ""}`}>
          <th scope="row"><small className="seed-num">{t.seed}</small>{t.team}{t.division_winner ? <sup title="Division winner">*</sup> : null}</th>
          <td>{t.wins}-{t.losses}</td><td>{pct(t.win_pct)}</td>
          <td>{t.games_behind === 0 ? "–" : t.games_behind.toFixed(1)}</td>
          <td className="wide-only">{t.home_wins}-{t.home_losses}</td><td className="wide-only">{t.away_wins}-{t.away_losses}</td>
          <td>{t.conference_wins}-{t.conference_losses}</td>
          <td>{signed(Math.round(t.point_diff / (t.wins + t.losses || 1) * 10) / 10)}</td>
          <td>{t.last_ten}</td><td>{t.streak}</td>
        </tr>
      ))}</tbody>
    </table>
  );
}

function GameTile({ game }: { game: SeasonGame }) {
  return (
    <article className="bracket-game">
      {[{ team: game.away, score: game.away_score, seed: game.away_seed },
        { team: game.home, score: game.home_score, seed: game.home_seed }].map((side) => (
        <div key={side.team} className={`bracket-team ${game.winner === side.team ? "winner" : ""}`}>
          <span className="bracket-name">{side.seed ? <small>{side.seed}</small> : null}{side.team}</span>
          <strong>{side.score}</strong>
        </div>
      ))}
      <div className="bracket-meta">{game.round}{game.overtimes ? ` · ${game.overtimes > 1 ? game.overtimes : ""}OT` : ""}</div>
    </article>
  );
}

function SeriesTile({ series }: { series: Series }) {
  const sides = [series.high, series.low];
  return (
    <details className="bracket-game series">
      <summary>
        {sides.map((side) => (
          <div key={side.team} className={`bracket-team ${series.winner === side.team ? "winner" : ""}`}>
            <span className="bracket-name"><small>{side.seed}</small>{side.team}</span>
            <strong>{series.wins[side.team]}</strong>
          </div>
        ))}
        <div className="bracket-meta">{series.summary} · games</div>
      </summary>
      <ol className="series-games">
        {series.games.map((g) => (
          <li key={g.game_id} className={g.winner === g.home ? "home-win" : "away-win"}>
            <span>G{g.game_in_series}</span>
            <span>{g.away} {g.away_score} @ {g.home} {g.home_score}{g.overtimes ? " (OT)" : ""}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

export default function SeasonSimulator({ seasons }: { seasons: number[] }) {
  const latestSeason = Math.min(LAST_SEASON - 1, Math.max(2026, ...seasons));
  const [season, setSeason] = useState(latestSeason + 1);
  const seasonEdited = useRef(false);
  const [seed, setSeed] = useState(0);
  const [temperature, setTemperature] = useState(1);
  const [strengthMode, setStrengthMode] = useState<StrengthMode>("checkpoint");
  const [job, setJob] = useState<SeasonJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [teamFilter, setTeamFilter] = useState("");

  // The season list arrives after the first render: follow it until the user picks a season.
  useEffect(() => {
    if (!seasonEdited.current) setSeason(latestSeason + 1);
  }, [latestSeason]);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = window.setInterval(() => {
      api.seasonJob(job.id).then(setJob).catch((e) => {
        // e.g. the server restarted and lost the job: stop polling and re-enable the form.
        window.clearInterval(timer);
        setJob((current) => current && current.id === job.id
          ? { ...current, status: "failed", error: errorText(e) } : current);
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  const inputError =
    !Number.isInteger(season) || season < FIRST_SEASON || season > LAST_SEASON
      ? `Season must be a whole number from ${FIRST_SEASON} to ${LAST_SEASON}.`
      : !Number.isInteger(seed) || seed < 0 ? "Seed must be a whole number of 0 or more."
      : !Number.isFinite(temperature) || temperature <= 0 || temperature > 3
        ? "Temperature must be greater than 0 and at most 3." : null;

  const runSeason = async () => {
    setError(null);
    setJob(null);
    try { setJob(await api.startSeason({ season, seed, temperature, strength_mode: strengthMode })); }
    catch (e) { setError(errorText(e)); }
  };

  const result = job?.result;
  const progress = job?.progress;
  const total = progress?.total || totalSlots(season);
  const playIn = season >= FIRST_PLAY_IN_SEASON;
  const filter = teamFilter.trim().toUpperCase();
  const regular = (result?.games ?? []).filter((g) => !filter || g.home === filter || g.away === filter);
  return (
    <main className="season-view">
      <section className="season-controls">
        <div className="season-title">
          <div><p className="eyebrow">NBA · SEASON LAB</p><h2>Full season simulation</h2></div>
          <span className="muted">82 games{playIn ? " + play-in" : ""} + best-of-seven playoffs</span>
        </div>
        <div className="season-form">
          <label>Season ({Number.isInteger(season) ? seasonLabel(season) : "—"})
            <input type="number" min={FIRST_SEASON} max={LAST_SEASON}
              value={Number.isNaN(season) ? "" : season}
              onChange={(e) => { seasonEdited.current = true; setSeason(e.target.valueAsNumber); }} /></label>
          <label>Seed<input type="number" min="0" step="1" value={Number.isNaN(seed) ? "" : seed}
            onChange={(e) => setSeed(e.target.valueAsNumber)} /></label>
          <label>Temperature<input type="number" min="0.1" max="3" step="0.1"
            value={Number.isNaN(temperature) ? "" : temperature}
            onChange={(e) => setTemperature(e.target.valueAsNumber)} /></label>
          <label>Team strength<select value={strengthMode}
            onChange={(e) => setStrengthMode(e.target.value as StrengthMode)}>
            <option value="checkpoint">Preseason ratings</option>
            <option value="randomized">Randomized profiles</option>
            <option value="neutral">League average</option>
          </select></label>
          <button className="primary run-season" disabled={job?.status === "running" || inputError !== null}
            onClick={runSeason}>Simulate season</button>
        </div>
        {inputError && <p className="season-schedule-note" role="status">{inputError}</p>}
        <p className="season-schedule-note">The 82-game schedule follows the NBA formula (4 games vs division rivals,
          3–4 vs the rest of the conference, 2 vs the other conference). Playoff series sample all seven games
          and stop at four wins.</p>
        {error && <div className="error" role="alert">{error}</div>}
        {job && (
          <div className={`season-progress ${job.status}`} aria-live="polite">
            <div className="progress-line"><strong>{job.status === "complete" ? "Season complete" :
              job.status === "failed" ? "Simulation failed" : progress?.stage}</strong>
              <span>{progress?.completed ?? 0} / {total} games</span></div>
            <progress max={total} value={progress?.completed ?? 0} />
            {job.error && <p className="error" role="alert">{job.error}</p>}
          </div>
        )}
      </section>

      {result && (
        <>
          <section className="season-summary" aria-label="Season summary">
            <div><span className="muted">Champion</span><strong className="champion-name">{result.champion}</strong></div>
            <div><span className="muted">Finals</span><strong>{result.finals.summary} vs {result.finals.loser}</strong></div>
            <div><span className="muted">Playoffs</span><strong>{result.playoff_game_count} games</strong></div>
            <div><span className="muted">Model strength</span><strong>{result.strength_mode}</strong></div>
          </section>

          <section className="season-standings">
            <div className="section-heading"><h2>Final standings</h2><span className="muted">
              {result.play_in ? "Seeds 1–6 qualify · 7–10 go to the play-in" : "Top 8 per conference qualify"}
              {" "}· * division winner · NBA tiebreakers</span></div>
            <div className="conference-grid">
              {(["East", "West"] as const).map((conference) => (
                <section key={conference} className="conference">
                  <h3>{conference}ern Conference</h3>
                  <StandingsTable rows={result.standings[conference].seeds} playIn={result.play_in} />
                </section>
              ))}
            </div>
          </section>

          <section className="season-playoffs">
            <div className="section-heading"><h2>Playoffs</h2><span className="muted">
              Higher seed hosts games 1, 2, 5 and 7 · click a series for game scores</span></div>
            <div className="bracket-scroll">
              <div className="bracket-rounds" style={{ gridTemplateColumns: `repeat(${result.playoffs.length + result.play_in_rounds.length}, minmax(190px, 1fr))` }}>
                {result.play_in_rounds.map((round) => (
                  <section key={round.round} className="bracket-round">
                    <h3>{round.round}</h3>
                    <div className="bracket-games">{round.games.map((g) => <GameTile key={g.game_id} game={g} />)}</div>
                  </section>
                ))}
                {result.playoffs.map((round) => (
                  <section key={round.round} className="bracket-round">
                    <h3>{round.round}</h3>
                    <div className="bracket-games">{round.series.map((s) => <SeriesTile key={s.high.team} series={s} />)}</div>
                  </section>
                ))}
              </div>
            </div>
          </section>

          <details className="season-games">
            <summary>Regular-season results <span>{result.regular_game_count} games</span></summary>
            <label className="inline results-filter">Team
              <select value={filter} onChange={(e) => setTeamFilter(e.target.value)}>
                <option value="">All teams</option>
                {[...result.standings.East.seeds, ...result.standings.West.seeds].map((t) => t.team).sort()
                  .map((t) => <option key={t} value={t}>{t}</option>)}
              </select></label>
            <div className="season-table-scroll">
              <table className="results-table"><thead><tr><th>Date</th><th>Away</th><th>Score</th><th>Home</th></tr></thead>
                <tbody>{regular.map((g) => (
                  <tr key={g.game_id}>
                    <td>{g.date}</td><th scope="row">{g.away}</th>
                    <td className="game-score">{g.away_score}–{g.home_score}{g.overtimes ? " OT" : ""}</td>
                    <th scope="row">{g.home}</th>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </main>
  );
}
