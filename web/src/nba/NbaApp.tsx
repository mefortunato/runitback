import { useCallback, useEffect, useRef, useState } from "react";
import Forecast from "../components/Forecast";
import { useBatchedForecast } from "../components/useBatchedForecast";
import WinProbChart from "../components/WinProbChart";
import { clock, qtrName } from "../format";
import { errorText } from "../http";
import { api, seasonLabel, situation, type Game, type StepBody, type Teams } from "./api";
import BoxScore from "./BoxScore";
import Court from "./Court";
import SeasonSimulator from "./SeasonSimulator";

const CALLABLE: Record<string, string> = {
  shot2: "2-pointer", shot3: "3-pointer", timeout: "Timeout", jump_ball: "Jump ball",
};
const MARGIN_TICKS = [-30, -20, -10, 0, 10, 20, 30];

export default function NbaApp() {
  const [meta, setMeta] = useState<Teams | null>(null);
  const [home, setHome] = useState("BOS");
  const [away, setAway] = useState("NYK");
  const [season, setSeason] = useState(2027);
  const [gameType, setGameType] = useState("REG");
  const [game, setGame] = useState<Game | null>(null);
  const fc = useBatchedForecast();
  const [busy, setBusy] = useState(false);
  const [auto, setAuto] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [liveWp, setLiveWp] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"game" | "season">("game");
  const wpBusy = useRef(false);
  const wpPending = useRef<Game | null>(null);
  const sinceWp = useRef(0);

  useEffect(() => {
    api.teams().then((t) => {
      setMeta(t);
      if (t.seasons.length) setSeason(t.seasons[t.seasons.length - 1]);
    }).catch((e) => setError(errorText(e)));
  }, []);

  // One win-probability rollout at a time; a request made meanwhile runs next, for the latest state.
  const refreshWp = useCallback(async (g: Game) => {
    if (wpBusy.current) { wpPending.current = g; return; }
    wpBusy.current = true;
    try {
      const wp = await api.winprob(g.id, 100);
      setGame((cur) => cur && cur.id === g.id
        ? { ...cur, winprob: [...cur.winprob.filter((w) => w.n !== wp.n), wp] } : cur);
    } catch { /* non-fatal */ } finally {
      wpBusy.current = false;
      const next = wpPending.current;
      wpPending.current = null;
      if (next) refreshWp(next);
    }
  }, []);

  // Win probability is refreshed every `wpEvery` events (rollouts are the expensive part).
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
      setError(errorText(e));
      setAuto(false);
    } finally {
      setBusy(false);
    }
  };

  const start = () => { setAuto(false); wpPending.current = null; run(() => api.newGame(home, away, season, gameType)); };
  const step = (body: StepBody, wpEvery = 1) => game && run(() => api.step(game.id, body), wpEvery);

  useEffect(() => {
    if (!auto || !game || busy) return;
    if (game.state.over) { setAuto(false); return; }
    const t = setTimeout(() => run(() => api.step(game.id, { until: "possession", count: 12 }), 6), speed);
    return () => clearTimeout(t);
  }, [auto, game, busy, speed]);

  const doForecast = () => fc.start(home, away, (n, signal) =>
    api.simulate(home, away, season, gameType, n, { include_scores: true, signal }));

  const ready = meta?.model_loaded ?? false;
  const teams = meta?.teams ?? [];
  const name = (t: string) => meta?.names[t] ?? t;
  const s = game?.state;
  const lastPlay = game?.plays[game.plays.length - 1];
  return (
    <div className="sport-page nba">
      <div className="page-intro">
        <h1>NBA</h1>
        <p className="muted">Play a game possession by possession, forecast a matchup over 1,000 simulated games, or simulate a full season through the Finals.</p>
      </div>

      {meta && !ready && (
        <section className="card notice" role="status">
          <h2>No NBA model loaded yet</h2>
          <p>{meta.error}</p>
          <pre><code>{`runitback nba download --start 2015 --end 2027
runitback nba preprocess --start 2015 --end 2027
runitback nba train --out runs/nba/prod
runitback serve`}</code></pre>
        </section>
      )}

      <nav className="view-tabs" aria-label="Simulation type">
        <button className={view === "game" ? "selected" : ""} aria-pressed={view === "game"}
          onClick={() => setView("game")}>Game</button>
        <button className={view === "season" ? "selected" : ""} aria-pressed={view === "season"}
          onClick={() => setView("season")}>Season</button>
      </nav>

      {view === "season" ? (ready && <SeasonSimulator seasons={meta?.seasons ?? []} />) : <>
      <section className="card setup">
        <label>Away<select value={away} onChange={(e) => setAway(e.target.value)}>
          {teams.map((t) => <option key={t} value={t}>{t} · {name(t)}</option>)}</select></label>
        <span className="at">@</span>
        <label>Home<select value={home} onChange={(e) => setHome(e.target.value)}>
          {teams.map((t) => <option key={t} value={t}>{t} · {name(t)}</option>)}</select></label>
        <label>Season<select value={season} onChange={(e) => setSeason(Number(e.target.value))} disabled={!ready}>
          {(meta?.seasons ?? []).map((y) => <option key={y} value={y}>{seasonLabel(y)}</option>)}</select></label>
        <label>Type<select value={gameType} onChange={(e) => setGameType(e.target.value)}>
          <option value="REG">Regular season</option><option value="POST">Playoffs</option></select></label>
        <button className="primary" disabled={!ready || busy || home === away} onClick={start}>New game</button>
        {fc.running
          ? <button onClick={fc.stop} aria-busy="true">
              Stop · {(fc.forecast?.n ?? 0).toLocaleString()} / {fc.total.toLocaleString()} games</button>
          : <button disabled={!ready || home === away} onClick={doForecast}>Forecast 1,000 games</button>}
      </section>
      {(error || fc.error) && <div className="error" role="alert">{error || fc.error}</div>}

      {fc.forecast && <Forecast r={fc.forecast} ticks={MARGIN_TICKS} />}

      {game && s && (
        <div className="game">
          <div className="main">
            <section className="card scoreboard" aria-live="polite">
              {([["away", s.away, s.away_score, s.away_timeouts], ["home", s.home, s.home_score, s.home_timeouts]] as const).map(
                ([side, team, score, to]) => (
                  <div key={side} className={`team ${s.pos === team && !s.over && s.phase !== "tip" ? "has-ball" : ""}`}>
                    <div className="abbr">{team}{s.pos === team && !s.over && s.phase !== "tip" && <span className="poss" aria-label="possession"> ●</span>}</div>
                    <div className="score">{score}</div>
                    <div className="tos" aria-label={`${to} timeouts left`}>
                      {Array.from({ length: Math.max(to, s.qtr >= 5 ? 2 : s.qtr === 4 ? 4 : 7) }, (_, i) =>
                        <span key={i} className={i < to ? "on" : ""} />)}
                    </div>
                  </div>
                ))}
              <div className="clock">
                <div className="q">{s.over ? "FINAL" : qtrName(s.qtr)}{s.over && s.qtr > 4 ? ` (${qtrName(s.qtr)})` : ""}</div>
                {!s.over && <div className="t">{clock(s.qsec)}</div>}
                <div className="sit">{situation(s)}</div>
              </div>
            </section>
            <section className="card">
              <Court s={s} plays={game.plays} />
              {lastPlay && <p className="last">{lastPlay.text}</p>}
            </section>
            <section className="card controls">
              <div className="row">
                <button className="primary" disabled={busy || s.over || auto} onClick={() => step({})}>Next event</button>
                <button disabled={busy || s.over || auto} onClick={() => step({ until: "possession", count: 30 })}>Next possession</button>
                <button disabled={busy || s.over || auto} onClick={() => step({ until: "period", count: 800 }, 1)}>
                  Sim to end of {s.qtr <= 4 ? qtrName(s.qtr) : "period"}</button>
                <button disabled={busy || s.over || auto} onClick={() => step({ count: 800 })}>Sim to end</button>
              </div>
              <div className="call">
                <span className="muted">Call:</span>
                {game.legal_play_types.filter((t) => t in CALLABLE).map((t) => (
                  <button key={t} disabled={busy || auto} onClick={() => step({ play_type: t })}>{CALLABLE[t]}</button>
                ))}
              </div>
              <div className="row">
                <button disabled={s.over} onClick={() => setAuto(!auto)}>{auto ? "Pause" : "Auto-play"}</button>
                <label className="inline">Speed
                  <input type="range" min="50" max="1500" step="50" value={1550 - speed}
                    onChange={(e) => setSpeed(1550 - Number(e.target.value))} /></label>
                <button disabled={busy || auto || game.plays.length === 0} onClick={() => run(() => api.undo(game.id), 1e9)}>Undo</button>
                <label className="inline"><input type="checkbox" checked={liveWp} onChange={(e) => setLiveWp(e.target.checked)} /> Live win prob</label>
              </div>
            </section>
            <section className="card">
              <BoxScore s={s} plays={game.plays} />
            </section>
            <section className="card">
              <WinProbChart data={game.winprob} home={s.home} away={s.away} total={game.plays.length} minN={300} unit="event" />
            </section>
          </div>
          <aside className="card log">
            <h2>Play-by-play <span className="muted">({game.plays.length})</span></h2>
            <ol reversed>
              {[...game.plays].reverse().map((p) => {
                const scored = p.after.home_score + p.after.away_score > p.before.home_score + p.before.away_score;
                return (
                  <li key={p.n} className={scored ? "scoring" : ""}>
                    <div className="meta">{qtrName(p.before.qtr)} {clock(p.after.qtr === p.before.qtr ? p.after.qsec : 0)}</div>
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
