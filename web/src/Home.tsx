import { useEffect, useState } from "react";
import { call } from "./http";
import { Link } from "./router";
import ReplayAnimation from "./components/ReplayAnimation";
import BacktestRecord from "./BacktestRecord";

interface Sport { id: string; name: string; model_available: boolean }

const SPORTS = [
  {
    id: "nfl", name: "NFL", to: "/nfl", unit: "play",
    blurb: "Spread, moneyline and over/under predictions for every game, checked against Vegas. And every snap, kickoff, punt, kick and timeout: call plays yourself, watch win probability move, forecast a matchup, or simulate a full season with schedule, tiebreakers and playoffs.",
    outcome: ["play type", "result", "yards", "clock"],
    facts: ["nflverse play-by-play, 1999 to today", "Engine replays 99.55% of real plays exactly", "Predictions, rankings and a walk-forward backtest", "Game, forecast and season simulators"],
  },
  {
    id: "nba", name: "NBA", to: "/nba", unit: "event",
    blurb: "Every shot, free throw, turnover, foul, timeout and jump ball. Step event by event or a possession at a time, call 2s and 3s, follow the shot chart and box score, or forecast a matchup.",
    outcome: ["event type", "result", "clock"],
    facts: ["ESPN play-by-play via hoopR, 2002 to today", "Engine replays 99.5% of real events exactly", "Game and forecast simulators"],
  },
];

const STEPS = [
  { h: "The game so far", p: "Score, clock, possession, down and distance or free throws left, timeouts, and every previous play." },
  { h: "Transformer", p: "A causal transformer reads that history and predicts a distribution over what happens next." },
  { h: "Sample an outcome", p: "Draw the next play field by field: its type, then its result, then yards (football) and clock, each conditioned on the fields before it." },
  { h: "Rules engine", p: "Deterministic rules apply the outcome: scoring, possession, periods, overtime. The game stays legal." },
];

export default function Home() {
  const [status, setStatus] = useState<Record<string, boolean>>({});
  useEffect(() => {
    call<{ sports: Sport[] }>("/api/sports")
      .then((r) => setStatus(Object.fromEntries(r.sports.map((s) => [s.id, s.model_available]))))
      .catch(() => { /* status badges are optional */ });
  }, []);
  return (
    <div className="home">
      <section className="hero">
        <p className="eyebrow">Sports, simulated one play at a time</p>
        <h1>Replay any matchup a thousand different ways.</h1>
        <p className="lede">
          Run It Back is a generative model of sports games. It has read decades of real play-by-play,
          and it writes new games the same way they happen: one play after another, each one depending
          on everything before it. Play a game and make the calls yourself, or simulate a matchup a
          thousand times to see how often each team wins and by how much.
        </p>
        <div className="hero-links">
          {SPORTS.map((s) => <Link key={s.id} to={s.to} className="button primary">Open {s.name}</Link>)}
        </div>
      </section>

      <ReplayAnimation />

      <BacktestRecord />

      <section className="sport-cards" aria-label="Sports">
        {SPORTS.map((s) => (
          <Link key={s.id} to={s.to} className="card sport-card">
            <div className="sport-card-head">
              <h2>{s.name}</h2>
              {s.id in status && (
                <span className={`badge ${status[s.id] ? "ok" : ""}`}>{status[s.id] ? "Model loaded" : "No model yet"}</span>
              )}
            </div>
            <p>{s.blurb}</p>
            <p className="outcome" aria-label={`Each ${s.unit} is sampled as`}>
              {s.outcome.map((o, i) => <span key={o}>{i > 0 && <span className="arrow" aria-hidden="true">→</span>}<code>{o}</code></span>)}
            </p>
            <ul>{s.facts.map((f) => <li key={f}>{f}</li>)}</ul>
            <span className="go">Open {s.name} →</span>
          </Link>
        ))}
      </section>

      <section className="how">
        <h2>How it works</h2>
        <ol className="loop" aria-label="One step of the simulation loop">
          {STEPS.map((s, i) => (
            <li key={s.h}><span className="num">{i + 1}</span><h3>{s.h}</h3><p>{s.p}</p></li>
          ))}
        </ol>
        <p className="loop-note muted">…and repeat until the final whistle or buzzer. Batched, a CPU plays a thousand games in seconds.</p>
        <div className="how-grid">
          <div>
            <h3>The model never breaks the rules</h3>
            <p>
              The network only predicts <em>what happened</em> on a play. A hand-written rules engine for
              each sport owns the game state and applies that outcome, and illegal outcomes are masked
              out before sampling. Replaying real games through the engines reproduces the next real
              state on more than 99% of plays.
            </p>
          </div>
          <div>
            <h3>Team strength</h3>
            <p>
              Each play sees both teams' pre-game ratings: offensive and defensive efficiency as
              recency-weighted averages, plus an Elo rating, all computed only from games played
              before it. Forecasts use each team's latest ratings for the season you pick.
            </p>
          </div>
          <div>
            <h3>Win probability</h3>
            <p>
              During a game, the live chart re-simulates the rest of it a few hundred times from the
              current state. A forecast plays the whole game a thousand times and shows the spread of
              final margins.
            </p>
          </div>
          <div>
            <h3>What it doesn't know</h3>
            <p>
              Teams, not players: no rosters, injuries or rest. A missing star is the biggest thing it
              cannot see, and ratings take several games to catch up. It is a fun, honest simulator,
              not a betting model.
            </p>
          </div>
        </div>
        <p className="muted data-credit">
          Data from <a href="https://www.sportsdataverse.org/">SportsDataverse</a> (nflverse and hoopR).
          Source on <a href="https://github.com/mefortunato/runitback">GitHub</a>.
        </p>
      </section>
    </div>
  );
}
