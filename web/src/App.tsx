import { useCallback, useEffect, useRef, useState } from "react";
import { api, clock, qtrName, situation, type Game, type SimResult } from "./api";
import Field from "./Field";
import MarginChart from "./MarginChart";
import WinProbChart from "./WinProbChart";

const CALLABLE: Record<string, string> = {
  run: "Run", pass: "Pass", punt: "Punt", field_goal: "Field goal", qb_kneel: "Kneel", qb_spike: "Spike",
  extra_point: "Extra point", two_point: "Go for two", kickoff: "Kickoff", timeout: "Timeout",
};

export default function App() {
  const [teams, setTeams] = useState<string[]>([]);
  const [seasons, setSeasons] = useState<number[]>([]);
  const [home, setHome] = useState("KC");
  const [away, setAway] = useState("BUF");
  const [season, setSeason] = useState(2026);
  const [gameType, setGameType] = useState("REG");
  const [game, setGame] = useState<Game | null>(null);
  const [forecast, setForecast] = useState<SimResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [auto, setAuto] = useState(false);
  const [speed, setSpeed] = useState(700);
  const [liveWp, setLiveWp] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wpBusy = useRef(false);
  const sinceWp = useRef(0);

  useEffect(() => {
    api.teams().then((t) => {
      setTeams(t.teams);
      setSeasons(t.seasons);
      setSeason(t.seasons[t.seasons.length - 1]);
    }).catch((e) => setError(String(e.message || e)));
  }, []);

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

  const start = () => run(() => api.newGame(home, away, season, gameType));
  const step = (play_type?: string) => game && run(() => api.step(game.id, { play_type }));

  useEffect(() => {
    if (!auto || !game || busy) return;
    if (game.state.over) { setAuto(false); return; }
    const t = setTimeout(() => run(() => api.step(game.id, {}), 8), speed);
    return () => clearTimeout(t);
  }, [auto, game, busy, speed]);

  const doForecast = async () => {
    setBusy(true);
    setError(null);
    try { setForecast(await api.simulate(home, away, season, gameType, 1000)); }
    catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };

  const s = game?.state;
  const lastPlay = game?.plays[game.plays.length - 1];
  return (
    <div className="app">
      <header>
        <h1>Run It Back <span className="sport">NFL</span></h1>
        <p className="muted">Simulate a game one play at a time, or replay a matchup a thousand times.</p>
      </header>

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
        <button disabled={busy || home === away} onClick={doForecast}>Forecast 1,000 games</button>
      </section>
      {error && <div className="error" role="alert">{error}</div>}

      {forecast && (
        <section className="card forecast">
          <div className="tiles">
            <div className="tile"><div className="muted">{forecast.away} win</div><div className="big">{(forecast.away_win * 100).toFixed(1)}%</div></div>
            <div className="tile"><div className="muted">{forecast.home} win</div><div className="big">{(forecast.home_win * 100).toFixed(1)}%</div></div>
            <div className="tile"><div className="muted">Avg score</div><div className="big">{forecast.away_score_mean.toFixed(1)}–{forecast.home_score_mean.toFixed(1)}</div></div>
            <div className="tile"><div className="muted">Avg total</div><div className="big">{forecast.total_mean.toFixed(1)}</div></div>
          </div>
          <MarginChart r={forecast} />
        </section>
      )}

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
    </div>
  );
}
