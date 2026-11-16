import { useEffect, useState } from "react";
import { call } from "./http";
import { Link } from "./router";
import ReplayAnimation from "./components/ReplayAnimation";
import BacktestRecord from "./BacktestRecord";

interface Sport { id: string; name: string; model_available: boolean }

// the sports with a play-by-play generator page (/<sport>/generator) next to their predictions
const SPORTS = [
  {
    id: "nfl", name: "NFL", unit: "play",
    blurb: "Every snap, kickoff, punt, kick and timeout: call plays yourself, watch win probability move, forecast a matchup, or simulate a full season with schedule, tiebreakers and playoffs. Plus spread, moneyline and over/under predictions for every game, checked against Vegas.",
    outcome: ["play type", "result", "yards", "clock"],
    facts: ["nflverse play-by-play, 1999 to today", "Engine replays 99.55% of real plays exactly", "Game, forecast and season simulators", "Predictions, rankings, player values and a walk-forward backtest"],
  },
  {
    id: "nba", name: "NBA", unit: "event",
    blurb: "Every shot, free throw, turnover, foul, timeout and jump ball. Step event by event or a possession at a time, call 2s and 3s, follow the shot chart and box score, or forecast a matchup. Plus game predictions built on player values.",
    outcome: ["event type", "result", "clock"],
    facts: ["ESPN play-by-play via hoopR, 2002 to today", "Engine replays 99.5% of real events exactly", "Game, forecast and season simulators", "Predictions with a lineup editor"],
  },
];

const STEPS = [
  { h: "The game so far", p: "Score, clock, possession, down and distance or free throws left, timeouts, and every previous play." },
  { h: "Transformer", p: "A causal transformer reads that history and predicts a distribution over what happens next." },
  { h: "Sample an outcome", p: "Draw the next play field by field: its type, then its result, then yards (football) and clock, each conditioned on the fields before it." },
  { h: "Rules engine", p: "Deterministic rules apply the outcome: scoring, possession, periods, overtime. The game stays legal." },
];

const WAGER_STEPS = [
  { h: "Read every past game", p: "The generator re-reads each real game play by play, with the real outcomes fed in, instead of writing a new one." },
  { h: "Game embedding", p: "Its internal state is averaged over each team's offensive and defensive plays and compressed to 32 numbers: a fingerprint of how that team played." },
  { h: "Wager transformer", p: "A second transformer reads each team's recent games (box score plus fingerprint) and the projected lineup's player values." },
  { h: "Price the game", p: "It outputs a probability for every final margin and total, which become the spread, moneyline and over/under picks set next to Vegas." },
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
        <p className="eyebrow">Sports, simulated one play at a time, and predicted before kickoff</p>
        <h1>Replay any matchup a thousand different ways. Then price it.</h1>
        <p className="lede">
          Run It Back is a generative model of sports games. It has read decades of real play-by-play,
          and it writes new games the same way they happen: one play after another, each one depending
          on everything before it. Play a game and make the calls yourself, or simulate a matchup a
          thousand times to see how often each team wins and by how much.
        </p>
        <p className="lede">
          On top of that generator sits a <strong>wager model</strong>: it uses what the generator learned
          about every past game, plus who is expected to play, to predict the spread, moneyline and
          over/under of upcoming games, and scores itself against the betting market.
        </p>
        <div className="hero-links">
          {SPORTS.map((s) => <Link key={s.id} to={`/${s.id}`} className="button primary">{s.name} predictions</Link>)}
          {SPORTS.map((s) => <Link key={`${s.id}-gen`} to={`/${s.id}/generator`} className="button">{s.name} generator</Link>)}
        </div>
      </section>

      <section className="pillars" aria-label="What Run It Back does">
        <article className="card pillar">
          <p className="eyebrow">1 · The generator</p>
          <h2>Replay games</h2>
          <p>
            A play-by-play transformer and a rules engine for each sport. Re-run a real matchup, call the
            plays yourself, watch live win probability, forecast a game over a thousand simulations, or
            play out a whole season.
          </p>
          <div className="pillar-links">
            {SPORTS.map((s) => <Link key={s.id} to={`/${s.id}/generator`} className="button">{s.name} generator</Link>)}
          </div>
        </article>
        <div className="pillar-arrow" aria-hidden="true">→</div>
        <article className="card pillar">
          <p className="eyebrow">2 · The wager model, built on the generator</p>
          <h2>Predict wagers</h2>
          <p>
            A game-level model that treats the generator as a foundation: it reads the generator's
            summary of each team's recent games, adds player values for the projected lineup, and predicts
            the margin and total of the next game. Those become spread, moneyline and over/under picks,
            next to Vegas.
          </p>
          <div className="pillar-links">
            {SPORTS.map((s) => <Link key={s.id} to={`/${s.id}`} className="button">{s.name} predictions</Link>)}
          </div>
        </article>
      </section>

      <h2 className="home-section-title">Replaying games</h2>
      <ReplayAnimation />

      <section className="how">
        <h2>How the generator works</h2>
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
            <h3>Who plays</h3>
            <p>
              The NFL generator also sees each team's depth chart, the players on the field and the head
              coach, and you can edit the lineup before a forecast. The NBA generator works at the team
              level: no rosters, injuries or rest.
            </p>
          </div>
        </div>
      </section>

      <section className="how wager-how">
        <h2>From replays to wager predictions</h2>
        <p className="muted section-lede">
          A generator that can write a believable game has to understand how each team plays. The wager
          model borrows that understanding instead of learning it from box scores alone.
        </p>
        <ol className="loop" aria-label="How the wager model uses the generator">
          {WAGER_STEPS.map((s, i) => (
            <li key={s.h}><span className="num">{i + 1}</span><h3>{s.h}</h3><p>{s.p}</p></li>
          ))}
        </ol>
        <p className="loop-note muted">Refit every season on earlier games only, and refreshed daily with the latest lines and lineups.</p>
        <div className="how-grid">
          <div>
            <h3>A frozen foundation</h3>
            <p>
              The generator is not retrained for betting: its embeddings are computed once and fed in
              as-is. Fine-tuning it on the wager loss made predictions worse, while the frozen
              32-number fingerprint matched or beat the box-score-only model on every market.
            </p>
          </div>
          <div>
            <h3>Player values</h3>
            <p>
              Each player gets a plus-minus value, refit weekly on earlier games. The projected lineup's
              value, compared with the team's recent lineups, moves the margin and total, so a missing
              starter shows up in the price. Swap players in the lineup editor to see it move.
            </p>
          </div>
          <div>
            <h3>No peeking</h3>
            <p>
              Every season is predicted by a generator, embedding and wager model trained only on
              earlier seasons, then replayed game by game. That walk-forward backtest is the record
              shown below and on each sport's Performance page.
            </p>
          </div>
          <div>
            <h3>Which sports use it</h3>
            <p>
              The NFL and college football models read the generator's embeddings. In the NBA the
              embedding carried little beyond Elo, so its model uses player values alone. The other
              sports ported from sportsml use box-score history for now.
            </p>
          </div>
        </div>
      </section>

      <BacktestRecord />

      <section className="sport-cards" aria-label="Sports">
        {SPORTS.map((s) => (
          <article key={s.id} className="card sport-card">
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
            <div className="pillar-links">
              <Link to={`/${s.id}`} className="button primary">Predictions</Link>
              <Link to={`/${s.id}/generator`} className="button">Generator</Link>
            </div>
          </article>
        ))}
      </section>

      <section className="how">
        <div className="how-grid">
          <div>
            <h3>What it doesn't know</h3>
            <p>
              Projected lineups are a guess until kickoff, and late scratches, weather and motivation are
              outside the data. Ratings and values take several games to catch up with a changed team.
            </p>
          </div>
          <div>
            <h3>For fun</h3>
            <p>
              The record above is an honest backtest, not a promise. Predictions are for entertainment,
              not betting advice.
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
