import { Link } from "./router";
import ReplayAnimation from "./components/ReplayAnimation";
import { api } from "./predictions/lib/api";
import { pct, seasonsText } from "./predictions/lib/format";
import { links } from "./predictions/lib/links";
import { breakEven } from "./predictions/lib/markets";
import type { SportInfo } from "./predictions/lib/types";
import { useData } from "./predictions/useData";
import { SPORTS, navSports } from "./sports";

const STEPS = [
  { h: "The game so far", p: "Score, clock, possession, timeouts, and every play that has happened." },
  { h: "Transformer", p: "A causal transformer reads that history and predicts a distribution over what happens next." },
  { h: "Sample an outcome", p: "Draw the next play field by field (its type, its result, then the clock), each conditioned on the fields before it." },
  { h: "Rules engine", p: "A rules engine for the sport applies the outcome: scoring, possession, periods, overtime. The game stays legal." },
];

const WAGER_STEPS = [
  { h: "Read every past game", p: "The generator re-reads each real game play by play, with the real outcomes fed in." },
  { h: "Game fingerprint", p: "Its internal state, averaged over each team's plays and compressed to 32 numbers, describes how that team played." },
  { h: "Wager model", p: "A second transformer reads each team's recent games and the value of the players expected to play." },
  { h: "Price the game", p: "It predicts the margin and total, which become spread, moneyline and over/under picks next to Vegas." },
];

export default function Home() {
  // sports.json (the predictions export): which sports have predictions, and their backtests
  const { data: infos } = useData(() => api.sports().catch(() => [] as SportInfo[]), []);
  const sports = navSports(infos?.map((i) => i.sport));
  const usesGenerator = SPORTS.filter((s) => s.wagerUsesGenerator);
  return (
    <div className="home">
      <section className="hero">
        <p className="eyebrow">Sports, simulated one play at a time, and predicted before kickoff</p>
        <h1>Replay any matchup a thousand different ways. Then price it.</h1>
        <p className="lede">
          Run It Back is a generative model of sports games. It has read decades of real play-by-play
          and writes new games the way they happen: one play after another, each depending on everything
          before it. On top of it sits a <strong>wager model</strong> that predicts the spread, moneyline
          and over/under of upcoming games and scores itself against the betting market.
        </p>
        <div className="hero-links">
          <a href="#sports" className="button primary">Pick a sport</a>
          <a href="#how" className="button">How it works</a>
        </div>
      </section>

      <section className="pillars" aria-label="What Run It Back does">
        <article className="card pillar">
          <p className="eyebrow">1 · The generator</p>
          <h2>Replay games</h2>
          <p>
            Re-run a matchup, call the plays yourself, watch live win probability, forecast a game over a
            thousand simulations, or play out a whole season.
          </p>
        </article>
        <div className="pillar-arrow" aria-hidden="true">→</div>
        <article className="card pillar">
          <p className="eyebrow">2 · The wager model, built on the generator</p>
          <h2>Predict wagers</h2>
          <p>
            Uses what the generator learned about each team's recent games, plus who is expected to play,
            to predict every upcoming game's margin and total, priced against Vegas.
          </p>
        </article>
      </section>

      <ReplayAnimation />

      <section className="how" id="how">
        <h2>How the generator works</h2>
        <ol className="loop" aria-label="One step of the simulation loop">
          {STEPS.map((s, i) => <li key={s.h}><span className="num">{i + 1}</span><h3>{s.h}</h3><p>{s.p}</p></li>)}
        </ol>
        <p className="loop-note muted">
          …and repeat until the final whistle. The network only predicts <em>what happened</em> on a play,
          and illegal outcomes are masked out, so simulated games never break the rules. Batched, a CPU
          plays a thousand games in seconds.
        </p>
      </section>

      <section className="how wager-how">
        <h2>From replays to wager predictions</h2>
        <ol className="loop" aria-label="How the wager model uses the generator">
          {WAGER_STEPS.map((s, i) => (
            <li key={s.h}>
              <span className="num">{i + 1}</span>
              <h3>{s.h}{i === 0 && <a href="#generator-note" className="footnote-ref" aria-label="Which sports use the generator">*</a>}</h3>
              <p>{s.p}</p>
            </li>
          ))}
        </ol>
        <p className="loop-note muted">
          No peeking: each season is predicted by models trained only on earlier seasons, and player values are refit weekly on earlier games. That walk-forward backtest is
          the record below.
        </p>
      </section>

      <section className="sports-summary" id="sports" aria-labelledby="sports-title">
        <h2 id="sports-title">Sports and performance</h2>
        <p className="muted section-lede">
          Each sport's walk-forward backtest against the Vegas lines. Beating the spread or the
          total takes winning more than {pct(breakEven(), 1)} of bets at standard −110 prices.
        </p>
        <div className="table-scroll">
          <table className="perf-table">
            <thead>
              <tr>
                <th scope="col">Sport</th>
                <th scope="col">Tested on</th>
                <th scope="col">Picks the winner<span className="muted"> (Vegas)</span></th>
                <th scope="col">Against the spread</th>
                <th scope="col">Over/under</th>
                <th scope="col">Games</th>
                <th scope="col"><span className="sr-only">Pages</span></th>
              </tr>
            </thead>
            <tbody>
              {sports.map((s) => {
                const info = infos?.find((i) => i.sport === s.id);
                return <SportRow key={s.id} id={s.id} label={s.label} generator={s.generator} info={info} usesGenerator={!!s.wagerUsesGenerator} />;
              })}
            </tbody>
          </table>
        </div>
        <p className="muted footnote" id="generator-note">
          <span aria-hidden="true">* </span>
          The generator's game fingerprints feed the production wager model only
          for {usesGenerator.map((s) => s.label).join(" and ")}. In the other sports the wager model reads
          box-score history, plus player values where the sport has them; where a sport's fingerprints were tested they did not
          improve the predictions enough to keep.
        </p>
      </section>

      <section className="how">
        <p className="muted closing">
          Projected lineups are a guess until kickoff, and late scratches, weather and motivation are
          outside the data. The record is an honest backtest, not a promise: for entertainment, not
          betting advice. Data from <a href="https://www.sportsdataverse.org/">SportsDataverse</a> and
          other public sources. Source on <a href="https://github.com/mefortunato/runitback">GitHub</a>.
        </p>
      </section>
    </div>
  );
}

function SportRow({ id, label, generator, info, usesGenerator }: {
  id: string; label: string; generator: boolean; info?: SportInfo; usesGenerator: boolean;
}) {
  const bt = info?.backtest?.overall?.model ? info.backtest : null;
  const dash = <span className="muted">–</span>;
  const rate = (v?: number | null) => (v == null ? dash : pct(v, 1));
  return (
    <tr>
      <th scope="row">
        <Link to={info ? links.upcoming(id) : links.generator(id)}>{label}</Link>
        {usesGenerator && <a href="#generator-note" className="footnote-ref" aria-label="Uses the generator">*</a>}
      </th>
      <td>{bt?.seasons?.length ? seasonsText(bt.seasons, info?.split_season) : dash}</td>
      <td>
        {bt ? <>{pct(bt.overall.model.accuracy)}{bt.overall.market && <span className="muted"> ({pct(bt.overall.market.accuracy)})</span>}</> : dash}
      </td>
      <td>{rate(bt?.ats?.win_pct)}</td>
      <td>{rate(bt?.totals?.record?.win_pct)}</td>
      <td>{bt ? bt.overall.model.n.toLocaleString("en-US") : dash}</td>
      <td>
        <span className="row-links">
          {info && <Link to={links.upcoming(id)}>Predictions</Link>}
          {info && info.challenge_games > 0 && <Link to={links.performance(id)}>Performance</Link>}
          {generator && <Link to={links.generator(id)}>Generator</Link>}
        </span>
      </td>
    </tr>
  );
}
