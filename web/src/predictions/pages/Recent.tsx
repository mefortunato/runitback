import { useState } from "react";
import { Link } from "../../router";
import { api } from "../lib/api";
import { dateText, pct, seasonText } from "../lib/format";
import { links } from "../lib/links";
import { unitsLabel } from "../lib/markets";
import { RECENT_SORT_LABELS, sortPlayed, summarize, type MarketRecord, type RecentSort, type RecentSummary } from "../lib/recent";
import type { SportInfo } from "../lib/types";
import PlayedGameCard from "../components/PlayedGameCard";
import { useData } from "../useData";
import { cx } from "../cx";
import s from "./Recent.module.css";

interface Line { label: string; model: string; vegas: string; better: boolean | null; note?: string }

const record = (r: MarketRecord) => `${r.wins}-${r.losses}${r.pushes ? `-${r.pushes}` : ""}`;
const rate = (r: MarketRecord) => (r.wins + r.losses ? pct(r.wins / (r.wins + r.losses)) : "—");
const SORTS: RecentSort[] = ["kickoff", "miss", "edge"];

export default function Recent({ info, season, date }: { info: SportInfo; season?: number | null; date?: number | null }) {
  const { data: recent, error } = useData(() => api.recent(info.sport, season, date), [info.sport, season, date]);
  const [sort, setSort] = useState<RecentSort>("kickoff");
  if (error) return <p className="error">{error}</p>;
  if (!recent) return <p className="muted">Loading…</p>;

  const lines = (sum: RecentSummary): Line[] => {
    const out: Line[] = [];
    const w = sum.winners;
    out.push({ label: "Picked the winner", model: `${record(w.model)} (${rate(w.model)})`,
      vegas: w.market ? `${record(w.market)} (${rate(w.market)})` : "—", better: w.market ? w.model.wins > w.market.wins : null });
    const bets: [string, MarketRecord | null][] = [
      ["Against the spread", info.has_spread ? sum.spread : null], ["Moneyline", sum.moneyline], ["Over/under", sum.total]];
    for (const [label, r] of bets) {
      if (r) out.push({ label, model: `${record(r)} (${rate(r)})`, vegas: "", better: null, note: unitsLabel(r.units) });
    }
    if (sum.margin) out.push({ label: "Avg margin miss", model: `${sum.margin.model.toFixed(1)} pts`, vegas: `${sum.margin.market.toFixed(1)} pts`, better: sum.margin.model < sum.margin.market });
    if (sum.points) out.push({ label: "Avg total miss", model: `${sum.points.model.toFixed(1)} pts`, vegas: `${sum.points.market.toFixed(1)} pts`, better: sum.points.model < sum.points.market });
    if (sum.brier) out.push({ label: "Brier score", model: sum.brier.model.toFixed(3), vegas: sum.brier.market.toFixed(3), better: sum.brier.model < sum.brier.market });
    return out;
  };

  // oldest to newest, like the upcoming page's dates
  const dates = [...recent.dates].reverse();
  const dayText = dateText(info.date_label, recent.date, recent.name);
  const windowText = `Last ${recent.dates.length} ${info.date_label.toLowerCase()}${recent.dates.length === 1 ? "" : "s"}`;
  const summaries = [{ title: dayText, summary: summarize(recent.games) }, { title: windowText, summary: summarize(recent.all) }];
  const replayed = recent.all.some((g) => g.prediction_source === "replay");
  const backtested = recent.all.some((g) => g.prediction_source === "backtest");
  const options = SORTS.filter((o) => o !== "edge" || recent.games.some((g) => g.prediction.edge != null));
  const active = options.includes(sort) ? sort : "kickoff";
  const games = sortPlayed(recent.games, active);

  return (
    <div className={s.root}>
      <div className={s.heading}>
        <div>
          <h1>
            {info.label} · {recent.date === null ? "Recent games" : dayText}{" "}
            {recent.season ? <span className="muted">{seasonText(recent.season, info.split_season)}</span> : null}
          </h1>
          <p className={cx("muted small", s.legend)}>
            {info.has_spread && <><span className={cx(s.key, s.vegas)} /> Vegas line</>}
            <span className={cx(s.key, s.model)} /> model's pre-game margin
            <span className={cx(s.key, s.final)} /> final margin · right of center favors the home team
          </p>
        </div>
        {dates.length > 1 && (
          <nav className={s.dates}>
            {dates.map((d) => (
              <Link key={`${d.season}-${d.date}`} className={cx("pill", d.season === recent.season && d.date === recent.date && "accent")}
                to={links.recent(info.sport, d.season, d.date)}>{dateText(info.date_label, d.date, d.name)}</Link>
            ))}
          </nav>
        )}
      </div>
      {recent.all.length === 0 ? (
        <div className="card">
          <h2>No recent games</h2>
          <p className="muted">
            No games have been played lately. See the <Link to={links.upcoming(info.sport)}>upcoming games</Link> or how the
            model did over whole seasons on the <Link to={links.performance(info.sport)}>performance page</Link>.
          </p>
        </div>
      ) : (
        <>
          <div className={s.summaries}>
            {summaries.map(({ title, summary }) => (
              <section key={title} className={cx("card", s.summary)}>
                <h2>{title} <span className="muted small">{summary.games} games</span></h2>
                <table className="small tabular">
                  <thead><tr><th></th><th>Model</th><th>Vegas</th></tr></thead>
                  <tbody>
                    {lines(summary).map((l) => (
                      <tr key={l.label}>
                        <th scope="row">{l.label}</th>
                        <td className={cx(l.better === true && s.better)}>{l.model}</td>
                        <td>{l.note
                          ? <span className={cx(l.note.startsWith("+") && s.win, !l.note.startsWith("+") && l.note !== "0.0 u" && s.loss)}>{l.note}</span>
                          : l.vegas}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            ))}
          </div>
          <p className={cx("muted small", s.note)}>
            Records are the model's picks at the lines the games closed at (spread and over/under at −110, moneyline at the
            posted odds); the units column is the profit from 1 unit on each.
            {replayed && " Predictions are the model's before each game: replayed with each team's state built from earlier games only, by a model not trained on this season."}
            {backtested && " Games from backtested seasons use the walk-forward backtest's predictions."}
          </p>
          {recent.games.length > 0 && (
            <>
              {options.length > 1 && (
                <div className={s.sort} role="group" aria-label="Order games">
                  <span className="muted small">Sort by</span>
                  {options.map((o) => (
                    <button key={o} className={cx("pill", active === o && "accent")} aria-pressed={active === o} onClick={() => setSort(o)}>
                      {RECENT_SORT_LABELS[o]}
                    </button>
                  ))}
                </div>
              )}
              <div className={s.games}>
                {games.map((g) => <PlayedGameCard key={g.game_id} game={g} sport={info.sport} hasSpread={info.has_spread} />)}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
