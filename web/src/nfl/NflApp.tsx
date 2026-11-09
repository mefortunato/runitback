import { useCallback, useEffect, useRef, useState } from "react";
import Forecast from "../components/Forecast";
import { useBatchedForecast } from "../components/useBatchedForecast";
import WinProbChart from "../components/WinProbChart";
import {
  api, clock, qtrName, situation, type CoachInfo, type Game, type LineupEdit, type LineupRequest,
  type TeamLineupView,
} from "./api";
import Field from "./Field";
import LineupEditor, { lineupFromView } from "./LineupEditor";
import PlayerStats from "./PlayerStats";
import SeasonSimulator from "./SeasonSimulator";

const CALLABLE: Record<string, string> = {
  run: "Run", pass: "Pass", punt: "Punt", field_goal: "Field goal", qb_kneel: "Kneel", qb_spike: "Spike",
  extra_point: "Extra point", two_point: "Go for two", kickoff: "Kickoff", timeout: "Timeout",
};

export default function NflApp() {
  const [view, setView] = useState<"game" | "season">("game");
  const [teams, setTeams] = useState<string[]>([]);
  const [seasons, setSeasons] = useState<number[]>([]);
  // a predictions page links here with ?home=SEA&away=ARI to generate that matchup
  const query = new URLSearchParams(window.location.search);
  const [home, setHome] = useState(query.get("home") || "KC");
  const [away, setAway] = useState(query.get("away") || "BUF");
  const [season, setSeason] = useState(2026);
  const [gameType, setGameType] = useState("REG");
  const [game, setGame] = useState<Game | null>(null);
  const fc = useBatchedForecast();
  const [busy, setBusy] = useState(false);
  const [auto, setAuto] = useState(false);
  const [speed, setSpeed] = useState(700);
  const [liveWp, setLiveWp] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wpBusy = useRef(false);
  const sinceWp = useRef(0);
  const [hasLineups, setHasLineups] = useState(false);
  const [showLineups, setShowLineups] = useState(false);
  const [coaches, setCoaches] = useState<CoachInfo[]>([]);
  const [views, setViews] = useState<Record<string, TeamLineupView>>({});
  const [edits, setEdits] = useState<Record<string, LineupEdit>>({});

  useEffect(() => {
    api.teams().then((t) => {
      setTeams(t.teams);
      setSeasons(t.seasons);
      setSeason(t.seasons[t.seasons.length - 1]);
      setHasLineups(!!t.lineups);
    }).catch((e) => setError(String(e.message || e)));
  }, []);

  // Each team's default lineup is the one from its most recent game; edits are kept per team.
  useEffect(() => {
    if (!hasLineups) return;
    api.coaches(season).then((c) => setCoaches(c.coaches)).catch(() => {});
    for (const t of [home, away]) {
      if (views[t]) continue;
      api.teamLineup(t, season).then((v) => {
        setViews((cur) => ({ ...cur, [t]: v }));
        setEdits((cur) => (cur[t] ? cur : { ...cur, [t]: lineupFromView(v) }));
      }).catch((e) => setError(String(e.message || e)));
    }
  }, [hasLineups, home, away, season]);

  const lineupRequest = (): LineupRequest | null => {
    if (!hasLineups) return null;
    const out: LineupRequest = {};
    for (const t of [home, away]) {
      const e = edits[t];
      if (!e) continue;
      out[t] = { coach: e.coach.trim() || null,
        depth_chart: Object.fromEntries(Object.entries(e.depth_chart).map(([g, ps]) => [g, ps.map((p) => p.id)])) };
    }
    return out;
  };
  const edited = [home, away].filter((t) => edits[t]?.edited);

  const refreshWp = useCallback(async (g: Game) => {
    if (wpBusy.current) return;
    wpBusy.current = true;
    try {
      const wp = await api.winprob(g.id);
      setGame((cur) => cur && cur.id === g.id
        ? { ...cur, winprob: [...cur.winprob.filter((w) => w.n !== wp.n), wp] } : cur);
    } catch { /* non-fatal */ } finally { wpBusy.current = false; }
  }, []);

  const run = async (fn: () => Promise<Game>, wpEvery = 1) => {
    setBusy(true);
    setError(null);
    try {
      const g = await fn();
      setGame(g);
      sinceWp.current += g.new_plays?.length ?? 1;
      if (liveWp && (sinceWp.current >= wpEvery || g.state.over)) {
        sinceWp.current = 0;
        refreshWp(g);
      }
      return g;
    } catch (e) {
      setError(String((e as Error).message || e));
      setAuto(false);
    } finally {
      setBusy(false);
    }
  };

  const start = () => run(() => api.newGame(home, away, season, gameType, lineupRequest()));
  const step = (play_type?: string) => game && run(() => api.step(game.id, { play_type }));

  useEffect(() => {
    if (!auto || !game || busy) return;
    if (game.state.over) { setAuto(false); return; }
    const t = setTimeout(() => run(() => api.step(game.id, {}), 8), speed);
    return () => clearTimeout(t);
  }, [auto, game, busy, speed]);

  const doForecast = () => fc.start(home, away, (n, signal) =>
    api.simulate(home, away, season, gameType, n, { include_scores: true, signal, lineups: lineupRequest() }));

  const s = game?.state;
  const lastPlay = game?.plays[game.plays.length - 1];
  return (
    <div className="sport-page">
      <div className="page-intro">
        <h1>NFL</h1>
        <p className="muted">Play one game, forecast a matchup, or simulate a full season through the playoffs.</p>
      </div>
      <nav className="view-tabs" aria-label="Simulation type">
        <button className={view === "game" ? "selected" : ""} aria-pressed={view === "game"}
          onClick={() => setView("game")}>Game</button>
        <button className={view === "season" ? "selected" : ""} aria-pressed={view === "season"}
          onClick={() => setView("season")}>Season</button>
      </nav>

      {view === "season" ? <SeasonSimulator seasons={seasons} /> : <>
      <section className="card setup">
        <label>Away<select value={away} onChange={(e) => setAway(e.target.value)}>
          {teams.map((t) => <option key={t}>{t}</option>)}</select></label>
        <span className="at">@</span>
        <label>Home<select value={home} onChange={(e) => setHome(e.target.value)}>
          {teams.map((t) => <option key={t}>{t}</option>)}</select></label>
        <label>Season<select value={season} onChange={(e) => setSeason(Number(e.target.value))}>
          {seasons.map((y) => <option key={y}>{y}</option>)}</select></label>
        <label>Type<select value={gameType} onChange={(e) => setGameType(e.target.value)}>
          <option value="REG">Regular</option><option value="POST">Playoff</option></select></label>
        <button className="primary" disabled={busy || home === away} onClick={start}>New game</button>
        {fc.running
          ? <button onClick={fc.stop} aria-busy="true">
              Stop · {(fc.forecast?.n ?? 0).toLocaleString()} / {fc.total.toLocaleString()} games</button>
          : <button disabled={home === away} onClick={doForecast}>Forecast 1,000 games</button>}
        {hasLineups && (
          <button aria-expanded={showLineups} onClick={() => setShowLineups(!showLineups)}>
            Depth charts &amp; coaches{edited.length > 0 && ` (edited: ${edited.join(", ")})`}
          </button>
        )}
      </section>
      {hasLineups && showLineups && (
        <section className="card lineups">
          <p className="muted small">
            Lineups start from each team's most recent game. Reorder by how much a player plays: the model picks who
            is on the field for each play from these depth charts and each player's recent snap share. Players in their
            first three games, or with no NFL history, use a generic <em>new player</em> rating for their position.
            Changes apply to the next new game or forecast.
          </p>
          <div className="lineup-grid">
            {([["away", away], ["home", home]] as const).map(([side, t]) => (
              <LineupEditor key={t} side={side} season={season} view={views[t] ?? null} value={edits[t] ?? null}
                coaches={coaches} onChange={(v) => setEdits((cur) => ({ ...cur, [t]: v }))}
                onReset={() => views[t] && setEdits((cur) => ({ ...cur, [t]: lineupFromView(views[t]) }))} />
            ))}
          </div>
        </section>
      )}
      {(error || fc.error) && <div className="error" role="alert">{error || fc.error}</div>}

      {fc.forecast && <Forecast r={fc.forecast} showTie />}

      {game && s && (
        <div className="game">
          <div className="main">
            <section className="card scoreboard">
              {[["away", s.away, s.away_score, s.away_timeouts], ["home", s.home, s.home_score, s.home_timeouts]].map(
                ([side, team, score, to]) => (
                  <div key={side as string} className={`team ${s.pos === team && !s.over ? "has-ball" : ""}`}>
                    <div className="abbr">{team as string}{s.pos === team && !s.over && <span className="poss" aria-label="possession"> ●</span>}</div>
                    <div className="score">{score as number}</div>
                    <div className="tos" aria-label={`${to} timeouts`}>
                      {[0, 1, 2].map((i) => <span key={i} className={i < (to as number) ? "on" : ""} />)}
                    </div>
                  </div>
                ))}
              <div className="clock">
                <div className="q">{s.over ? "FINAL" : qtrName(s.qtr)}</div>
                {!s.over && <div className="t">{clock(s.qsec)}</div>}
                <div className="sit">{situation(s)}</div>
              </div>
            </section>
            <section className="card">
              <Field s={s} prev={lastPlay?.before} />
              {lastPlay && <p className="last">{lastPlay.text}</p>}
            </section>
            <section className="card controls">
              <button className="primary" disabled={busy || s.over || auto} onClick={() => step()}>Next play</button>
              <div className="call">
                <span className="muted">Call:</span>
                {game.legal_play_types.filter((t) => t !== "penalty").map((t) => (
                  <button key={t} disabled={busy || auto} onClick={() => step(t)}>{CALLABLE[t] || t}</button>
                ))}
              </div>
              <div className="row">
                <button disabled={s.over} onClick={() => setAuto(!auto)}>{auto ? "Pause" : "Auto-play"}</button>
                <label className="inline">Speed
                  <input type="range" min="50" max="1500" step="50" value={1550 - speed}
                    onChange={(e) => setSpeed(1550 - Number(e.target.value))} /></label>
                <button disabled={busy || s.over || auto} onClick={() => run(() => api.step(game.id, { count: 400 }))}>Sim to end</button>
                <button disabled={busy || auto || game.plays.length === 0} onClick={() => run(() => api.undo(game.id), 1e9)}>Undo</button>
                <label className="inline"><input type="checkbox" checked={liveWp} onChange={(e) => setLiveWp(e.target.checked)} /> Live win prob</label>
              </div>
            </section>
            {game.player_stats && <PlayerStats stats={game.player_stats} home={s.home} away={s.away} />}
            <section className="card">
              <WinProbChart data={game.winprob} home={s.home} away={s.away} total={game.plays.length} />
            </section>
          </div>
          <aside className="card log">
            <h2>Play-by-play <span className="muted">({game.plays.length})</span></h2>
            <ol reversed>
              {[...game.plays].reverse().map((p) => {
                const scored = p.after.home_score + p.after.away_score > p.before.home_score + p.before.away_score;
                return (
                  <li key={p.n} className={scored ? "scoring" : ""}>
                    <div className="meta">{qtrName(p.before.qtr)} {clock(p.before.qsec)} · {situation(p.before)}</div>
                    <div>{p.text}</div>
                    {scored && <div className="meta">{p.after.away} {p.after.away_score} – {p.after.home} {p.after.home_score}</div>}
                  </li>
                );
              })}
            </ol>
          </aside>
        </div>
      )}
      </>}
    </div>
  );
}
