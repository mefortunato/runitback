import { Link } from "./router";
import { api } from "./predictions/lib/api";
import { pct, seasonsText } from "./predictions/lib/format";
import { links } from "./predictions/lib/links";
import { breakEven } from "./predictions/lib/markets";
import type { SportInfo } from "./predictions/lib/types";
import { useData } from "./predictions/useData";

// where each sport's results, box scores and lines come from
const SOURCES: Record<string, string> = {
  nfl: "nflverse",
  nba: "ESPN (schedules and box scores via sportsdataverse)",
};

/** Each sport's walk-forward backtest record against Vegas (nothing when the
 * predictions export is missing or has no backtest). */
export default function BacktestRecord() {
  const { data } = useData(() => api.sports(), []);
  const sports = (data ?? []).filter((s) => s.backtest?.overall?.model);
  if (!sports.length) return null;
  const even = pct(breakEven(), 1);
  return (
    <section className="record-section" aria-labelledby="record-title">
      <h2 id="record-title">Predictions, scored against Vegas</h2>
      <p className="muted record-lede">
        Every upcoming game gets a predicted margin, win probability and total, next to the betting
        market's spread, moneyline and over/under. These records come from a walk-forward backtest: each
        season is predicted by a model trained only on earlier seasons, replayed game by game.
      </p>
      <div className="record-cards">
        {sports.map((info) => <SportRecord key={info.sport} info={info} even={even} />)}
      </div>
      <p className="muted record-note">
        Beating the spread takes winning more
        than {even} of bets at standard −110 prices, and each sport's Performance page breaks the record
        down by market, season and size of edge. For entertainment, not betting advice.
      </p>
    </section>
  );
}

function SportRecord({ info, even }: { info: SportInfo; even: string }) {
  const bt = info.backtest!;
  const { model, market } = bt.overall;
  const seasons = bt.seasons?.length ? seasonsText(bt.seasons, info.split_season) : null;
  return (
    <article className="card record-card">
      <header>
        <h3><Link to={links.upcoming(info.sport)}>{info.label}</Link></h3>
        {seasons && <span className="badge">Tested on {seasons}</span>}
      </header>
      <dl className="record">
        <div>
          <dt>Picks the winner</dt>
          <dd>{pct(model.accuracy)}{market && <span className="muted">Vegas {pct(market.accuracy)}</span>}</dd>
        </div>
        {bt.ats?.win_pct != null && (
          <div>
            <dt>Against the spread</dt>
            <dd>{pct(bt.ats.win_pct, 1)}<span className="muted">break-even {even}</span></dd>
          </div>
        )}
        {bt.totals?.record?.win_pct != null && (
          <div>
            <dt>Over/under picks</dt>
            <dd>{pct(bt.totals.record.win_pct, 1)}<span className="muted">break-even {even}</span></dd>
          </div>
        )}
        <div>
          <dt>Games tested</dt>
          <dd>{model.n.toLocaleString("en-US")}</dd>
        </div>
      </dl>
      <nav className="record-links" aria-label={`${info.label} pages`}>
        <Link className="button primary" to={links.upcoming(info.sport)}>Upcoming games</Link>
        <Link className="button" to={links.rankings(info.sport)}>Power rankings</Link>
        {info.challenge_games > 0 && <Link className="button" to={links.performance(info.sport)}>Performance</Link>}
        {info.challenge_games > 0 && (info.has_spread || !!info.markets?.length) && (
          <Link className="button" to={links.challenge(info.sport)}>Beat the model</Link>
        )}
      </nav>
      {SOURCES[info.sport] && <p className="muted record-source">Results, box scores and lines from {SOURCES[info.sport]}.</p>}
    </article>
  );
}
