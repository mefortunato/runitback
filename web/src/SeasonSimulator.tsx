import { useEffect, useRef, useState } from "react";
import { api, type SeasonGame, type SeasonJob, type SeasonRequest, type Standing } from "./api";

type StrengthMode = SeasonRequest["strength_mode"];

// Seasons the simulator supports: the 32-team, eight-division alignment (2002 onward).
const FIRST_SEASON = 2002;
const LAST_SEASON = 2040;

function record(standing: Standing) {
  return `${standing.wins}-${standing.losses}${standing.ties ? `-${standing.ties}` : ""}`;
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
      <div className="bracket-meta">{game.site === "neutral" ? "Neutral site" : `Week ${game.week}`}</div>
    </article>
  );
}

function StandingsTable({ rows }: { rows: Standing[] }) {
  return (
    <table className="standings-table">
      <thead><tr><th>Team</th><th>W-L</th><th>PCT</th><th>DIFF</th><th>Bye</th><th>Seed</th></tr></thead>
      <tbody>{rows.map((team) => (
        <tr key={team.team} className={team.playoff_team ? "playoff-qualifier" : ""}>
          <th scope="row">{team.team}</th><td>{record(team)}</td>
          <td>{team.win_pct.toFixed(3).replace(/^0/, "")}</td>
          <td>{team.point_diff > 0 ? `+${team.point_diff}` : team.point_diff}</td>
          <td>{team.bye_week ?? ""}</td>
          <td>{team.seed ?? ""}</td>
        </tr>
      ))}</tbody>
    </table>
  );
}

export default function SeasonSimulator({ seasons }: { seasons: number[] }) {
  const latestSeason = Math.min(LAST_SEASON - 1, Math.max(2025, ...seasons));
  const [season, setSeason] = useState(latestSeason + 1);
  const seasonEdited = useRef(false);
  const [seed, setSeed] = useState(23);
  const [temperature, setTemperature] = useState(1);
  const [strengthMode, setStrengthMode] = useState<StrengthMode>("randomized");
  const [job, setJob] = useState<SeasonJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The season list arrives after the first render: follow it until the user picks a season.
  useEffect(() => {
    if (!seasonEdited.current) setSeason(latestSeason + 1);
  }, [latestSeason]);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = window.setInterval(() => {
      api.seasonJob(job.id).then(setJob).catch((e: Error) => {
        // e.g. the server restarted and lost the job: stop polling and re-enable the form.
        window.clearInterval(timer);
        setJob((current) => current && current.id === job.id
          ? { ...current, status: "failed", error: e.message } : current);
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
    const request: SeasonRequest = {
      season, seed, temperature, strength_mode: strengthMode,
    };
    try { setJob(await api.startSeason(request)); }
    catch (e) { setError((e as Error).message); }
  };

  const result = job?.result;
  const expectedGames = season <= 2020 ? 256 : 272;
  const expectedPlayoffGames = season < 2020 ? 11 : 13;
  const progress = job?.progress;
  return (
    <main className="season-view">
      <section className="season-controls">
        <div className="season-title">
          <div><p className="eyebrow">NFL · SEASON LAB</p><h2>Full season simulation</h2></div>
          <span className="muted">Regular season + {season < 2020 ? 12 : 14}-team playoffs</span>
        </div>
        <div className="season-form">
          <label>Season<input type="number" min={FIRST_SEASON} max={LAST_SEASON}
            value={Number.isNaN(season) ? "" : season}
            onChange={(event) => { seasonEdited.current = true; setSeason(event.target.valueAsNumber); }} /></label>
          <label>Seed<input type="number" min="0" step="1" value={Number.isNaN(seed) ? "" : seed}
            onChange={(event) => setSeed(event.target.valueAsNumber)} /></label>
          <label>Temperature<input type="number" min="0.1" max="3" step="0.1"
            value={Number.isNaN(temperature) ? "" : temperature}
            onChange={(event) => setTemperature(event.target.valueAsNumber)} /></label>
          <label>Team strength<select value={strengthMode}
            onChange={(event) => setStrengthMode(event.target.value as StrengthMode)}>
            <option value="randomized">Randomized profiles</option>
            <option value="neutral">League average</option>
            <option value="checkpoint">Checkpoint ratings</option>
          </select></label>
          <button className="primary run-season" disabled={job?.status === "running" || inputError !== null}
            onClick={runSeason}>Simulate season</button>
        </div>
        {inputError && <p className="season-schedule-note" role="status">{inputError}</p>}
        <p className="season-schedule-note">Opponent rotations and weekly matchups are generated automatically.</p>
        {error && <div className="error" role="alert">{error}</div>}
        {job && (
          <div className={`season-progress ${job.status}`} aria-live="polite">
            <div className="progress-line"><strong>{job.status === "complete" ? "Season complete" :
              job.status === "failed" ? "Simulation failed" : progress?.stage}</strong>
              <span>{progress?.completed ?? 0} / {progress?.total ?? expectedGames + expectedPlayoffGames} games</span></div>
            <progress max={progress?.total || expectedGames + expectedPlayoffGames} value={progress?.completed ?? 0} />
            {job.error && <p className="error" role="alert">{job.error}</p>}
          </div>
        )}
      </section>

      {result && (
        <>
          <section className="season-summary" aria-label="Season summary">
            <div><span className="muted">Champion</span><strong className="champion-name">{result.champion}</strong></div>
            <div><span className="muted">Regular season</span><strong>{result.regular_game_count} games</strong></div>
            <div><span className="muted">Playoffs</span><strong>{result.playoff_team_count} teams · {result.playoff_game_count} games</strong></div>
            <div><span className="muted">Model strength</span><strong>{result.strength_mode}</strong></div>
          </section>
          {result.position_source && <p className="schedule-provenance">Schedule position matchups: {result.position_source}</p>}

          <section className="season-standings">
            <div className="section-heading"><h2>Final standings</h2><span className="muted">
              Top {result.playoff_team_count / 2} per conference qualify · ties broken by NFL tiebreakers</span></div>
            <div className="conference-grid">
              {(["AFC", "NFC"] as const).map((conference) => (
                <section key={conference} className="conference">
                  <h3>{conference}</h3>
                  <div className="division-grid">
                    {Object.entries(result.standings[conference].divisions).map(([division, teams]) => (
                      <div key={division} className="division-table">
                        <h4>{conference} {division}</h4><StandingsTable rows={teams} />
                      </div>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          </section>

          <section className="season-playoffs">
            <div className="section-heading"><h2>Playoff bracket</h2><span className="muted">
              {result.playoff_team_count} teams · conference champions meet at a neutral site</span></div>
            <div className="bracket-scroll">
              <div className="bracket-rounds">
                {result.playoffs.map((round) => (
                  <section key={round.round} className="bracket-round">
                    <h3>{round.round}</h3>
                    <div className="bracket-games">{round.games.map((game) => <GameTile key={game.game_id} game={game} />)}</div>
                  </section>
                ))}
              </div>
            </div>
          </section>

          <details className="season-games">
            <summary>Regular-season results <span>{result.regular_game_count} games</span></summary>
            <div className="season-table-scroll">
              <table className="results-table"><thead><tr><th>Week</th><th>Away</th><th>Score</th><th>Home</th></tr></thead>
                <tbody>{result.games.filter((game) => game.stage === "Regular Season").map((game) => (
                  <tr key={game.game_id}>
                    <td>{game.week}</td><th scope="row">{game.away}</th>
                    <td className="game-score">{game.away_score}–{game.home_score}</td>
                    <th scope="row">{game.home}</th>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </details>
          {result.training_data_path && <p className="dataset-path">Training data saved to <code>{result.training_data_path}</code></p>}
        </>
      )}
    </main>
  );
}