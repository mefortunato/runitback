import { handicap, num, spreadLabel } from "../lib/format";
import { americanOdds } from "../lib/markets";
import { age } from "../lib/odds";
import type { BookOffer, GameOdds, Team } from "../lib/types";
import { cx } from "../cx";
import s from "./Sportsbooks.module.css";

const W = 220, H = 44;
const isBest = (offer: BookOffer | null, book: string, odds: number | null, points?: number | null) =>
  !!offer && offer.book === book && offer.odds === odds && (points == null || offer.points === points);

// Every book's line, the best price on each side highlighted, and how the consensus moved since the
// line was first seen.
export default function Sportsbooks({ home, away, odds }: { home: Team; away: Team; odds: GameOdds }) {
  // the consensus spread over time, as a small line chart (home margin up)
  const points = odds.history.filter((p) => p.spread !== null && p.spread !== undefined) as { t: string; spread: number }[];
  let path = "";
  if (points.length >= 2) {
    const times = points.map((p) => new Date(p.t).getTime());
    const lo = Math.min(...points.map((p) => p.spread)) - 0.5;
    const hi = Math.max(...points.map((p) => p.spread)) + 0.5;
    const t0 = times[0], t1 = times[times.length - 1] || t0 + 1;
    path = points.map((p, i) => {
      const x = 4 + ((times[i] - t0) / Math.max(1, t1 - t0)) * (W - 8);
      const y = 4 + ((hi - p.spread) / (hi - lo)) * (H - 8);
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join("");
  }
  const c = odds.consensus;
  const best = (on: boolean) => cx(s.num, "tabular", on && s.best);
  return (
    <div className={cx("card", s.root, s.books)}>
      <div className={s.head}>
        <h2>Sportsbooks</h2>
        <span className="muted small">updated {age(odds.updated)}</span>
      </div>
      <div className={s.scroll}>
        <table className="small">
          <thead>
            <tr>
              <th>Book</th><th className={s.num}>{away.team}</th><th className={s.num}>{home.team}</th>
              <th className={s.num}>{away.team} ML</th><th className={s.num}>{home.team} ML</th>
              <th className={s.num}>Over</th><th className={s.num}>Under</th>
            </tr>
          </thead>
          <tbody>
            {odds.books.map((b) => (
              <tr key={b.book}>
                <td>{b.book}</td>
                <td className={best(isBest(odds.best.spread.away, b.book, b.spread_away_odds, b.spread))}>
                  {b.spread === null ? "—" : `${handicap(b.spread, "away")} ${b.spread_away_odds === null ? "" : americanOdds(b.spread_away_odds)}`}
                </td>
                <td className={best(isBest(odds.best.spread.home, b.book, b.spread_home_odds, b.spread))}>
                  {b.spread === null ? "—" : `${handicap(b.spread, "home")} ${b.spread_home_odds === null ? "" : americanOdds(b.spread_home_odds)}`}
                </td>
                <td className={best(isBest(odds.best.moneyline.away, b.book, b.ml_away))}>{americanOdds(b.ml_away)}</td>
                <td className={best(isBest(odds.best.moneyline.home, b.book, b.ml_home))}>{americanOdds(b.ml_home)}</td>
                <td className={best(isBest(odds.best.total.over, b.book, b.over_odds, b.total))}>
                  {b.total === null ? "—" : `${num(b.total)} ${b.over_odds === null ? "" : americanOdds(b.over_odds)}`}
                </td>
                <td className={best(isBest(odds.best.total.under, b.book, b.under_odds, b.total))}>
                  {b.total === null ? "—" : `${num(b.total)} ${b.under_odds === null ? "" : americanOdds(b.under_odds)}`}
                </td>
              </tr>
            ))}
            <tr className={s.consensus}>
              <td>Consensus</td>
              <td className={cx(s.num, "tabular")}>{c.spread === null ? "—" : handicap(c.spread, "away")}</td>
              <td className={cx(s.num, "tabular")}>{c.spread === null ? "—" : handicap(c.spread, "home")}</td>
              <td className={cx(s.num, "tabular")}>{americanOdds(c.ml_away)}</td>
              <td className={cx(s.num, "tabular")}>{americanOdds(c.ml_home)}</td>
              <td className={cx(s.num, "tabular")} colSpan={2}>{c.total === null ? "—" : num(c.total)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className={cx("muted small", s.note)}>Highlighted: the best price on each side. Spreads without a price are −110.</p>
      <div className={cx(s.movement, "small")}>
        <div><span className="muted">Spread</span> {spreadLabel(home, away, odds.open.spread)} → <strong>{spreadLabel(home, away, c.spread)}</strong></div>
        <div><span className="muted">Total</span> {odds.open.total === null ? "—" : num(odds.open.total)} → <strong>{c.total === null ? "—" : num(c.total)}</strong></div>
        {path && (
          <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Consensus spread over time">
            <path d={path} className={s.trend} />
          </svg>
        )}
      </div>
    </div>
  );
}
