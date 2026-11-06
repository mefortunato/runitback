import { useState } from "react";
import { pct } from "../lib/format";
import { MARKET_LABELS, pickLabel } from "../lib/markets";
import { KELLY_FRACTION, noVig, priceMoneyline, priceSpread, priceTotal, validOdds, type Quote } from "../lib/pricing";
import type { BookLine, GameHeader, Market, Prediction, Pricing } from "../lib/types";
import { cx } from "../cx";
import s from "./LineCalculator.module.css";

interface Line {
  handicap: number; spreadHome: number; spreadAway: number; mlHome: number; mlAway: number;
  total: number; over: number; under: number;
}
interface Row { market: Market; quotes: [Quote, Quote] | null; describe: (q: Quote) => string }

const fmtOdds = (v: number) => (v > 0 ? `+${v}` : `${v}`);

// Price the line your sportsbook offers: starts at the Vegas consensus; change any number and every
// pick, probability and stake updates. Sports without a point spread price no spread.
export default function LineCalculator({ game, prediction, pricing, books = [], hasSpread = true }: {
  game: GameHeader; prediction: Prediction; pricing: Pricing;
  /** Sportsbooks' current lines, to load one into the calculator. */
  books?: (BookLine & { book: string })[]; hasSpread?: boolean;
}) {
  // books quote the home team's handicap: minus the expected home margin
  const consensus = (): Line => ({
    handicap: game.spread == null ? 0 : -game.spread,
    spreadHome: -110, spreadAway: -110,
    mlHome: game.moneyline?.home ?? -110, mlAway: game.moneyline?.away ?? -110,
    total: game.total_line ?? Math.round((prediction.total ?? 44) * 2) / 2,
    over: game.total_odds?.over ?? -110, under: game.total_odds?.under ?? -110,
  });
  const [line, setLine] = useState<Line>(consensus);
  const [source, setSource] = useState("");
  const reset = () => { setLine(consensus()); setSource(""); };
  // a book's line, its missing markets and prices from the consensus
  const load = (name: string) => {
    setSource(name);
    const b = books.find((x) => x.book === name);
    if (!b) return reset();
    const base = consensus();
    setLine({
      handicap: b.spread == null ? base.handicap : -b.spread,
      spreadHome: b.spread_home_odds ?? -110, spreadAway: b.spread_away_odds ?? -110,
      mlHome: b.ml_home ?? base.mlHome, mlAway: b.ml_away ?? base.mlAway,
      total: b.total ?? base.total, over: b.over_odds ?? -110, under: b.under_odds ?? -110,
    });
  };
  const field = (key: keyof Line, step: number, label: string, ariaLabel: string) => (
    <label>
      <span>{label}</span>
      <input type="number" step={step} aria-label={ariaLabel} value={Number.isFinite(line[key]) ? line[key] : ""}
        onChange={(e) => setLine({ ...line, [key]: e.target.value === "" ? NaN : Number(e.target.value) })} />
    </label>
  );

  // every market is moved toward the consensus (its spread, total and moneyline), not toward the
  // line you enter: a better number is what you are looking for
  const marketProb = game.moneyline ? noVig(game.moneyline.home, game.moneyline.away) : undefined;
  const rows: Row[] = [];
  const spreadOk = validOdds(line.spreadHome) && validOdds(line.spreadAway) && Number.isFinite(line.handicap);
  if (hasSpread) {
    rows.push({
      market: "spread",
      quotes: spreadOk ? priceSpread(prediction, -line.handicap, { home: line.spreadHome, away: line.spreadAway }, pricing, game.spread ?? undefined) : null,
      describe: (q) => {
        const h = q.pick === "home" ? line.handicap : -line.handicap;
        return `${pickLabel(q.pick, game.home, game.away)} ${h > 0 ? "+" : ""}${h === 0 ? "PK" : h}`;
      },
    });
  }
  const mlOk = validOdds(line.mlHome) && validOdds(line.mlAway);
  rows.push({
    market: "moneyline",
    quotes: mlOk ? priceMoneyline(prediction, { home: line.mlHome, away: line.mlAway }, pricing, marketProb) : null,
    describe: (q) => pickLabel(q.pick, game.home, game.away),
  });
  const totalOk = validOdds(line.over) && validOdds(line.under) && Number.isFinite(line.total);
  const total = totalOk ? priceTotal(prediction, line.total, { over: line.over, under: line.under }, pricing, game.total_line ?? undefined) : null;
  if (total || prediction.total != null) {
    rows.push({ market: "total", quotes: total, describe: (q) => `${pickLabel(q.pick, game.home, game.away)} ${line.total}` });
  }
  const trustText = (["spread", "moneyline", "total"] as Market[])
    .filter((m) => pricing.trust[m] !== undefined)
    .map((m) => `${MARKET_LABELS[m].toLowerCase()} ${pct(pricing.trust[m] ?? 0)}`)
    .join(", ");

  return (
    <div className={cx("card", s.root, s.calc)}>
      <div className={s.head}>
        <h2>Price your line</h2>
        <div className={s.actions}>
          {books.length > 0 && (
            <select value={source} onChange={(e) => load(e.target.value)} aria-label="Load a sportsbook's line">
              <option value="">Consensus</option>
              {books.map((b) => <option key={b.book} value={b.book}>{b.book}</option>)}
            </select>
          )}
          <button className="small" onClick={reset}>Reset to Vegas</button>
        </div>
      </div>
      <p className="muted small">
        Enter the line and odds your sportsbook offers. The model's prediction doesn't change with the odds, so every
        pick and stake below updates as you type.
      </p>
      <div className={s.inputs}>
        {hasSpread && (
          <fieldset>
            <legend>Spread</legend>
            {field("handicap", 0.5, `${game.home.team} handicap`, `${game.home.team} spread handicap`)}
            {field("spreadHome", 5, `${game.home.team} odds`, `${game.home.team} spread odds`)}
            {field("spreadAway", 5, `${game.away.team} odds`, `${game.away.team} spread odds`)}
          </fieldset>
        )}
        <fieldset>
          <legend>Moneyline</legend>
          {field("mlHome", 5, game.home.team, `${game.home.team} moneyline`)}
          {field("mlAway", 5, game.away.team, `${game.away.team} moneyline`)}
        </fieldset>
        {prediction.total != null && (
          <fieldset>
            <legend>Over/under</legend>
            {field("total", 0.5, "Total", "Total line")}
            {field("over", 5, "Over odds", "Over odds")}
            {field("under", 5, "Under odds", "Under odds")}
          </fieldset>
        )}
      </div>
      <div className={s.scroll}>
        <table className="small">
          <thead>
            <tr>
              <th>Bet</th>
              <th className={s.num}>Odds</th>
              <th className={s.num} title="The model's probability, moved toward the market by how much its disagreements have paid; below it the win rate the odds need to break even">Win chance</th>
              <th className={s.num}>EV / $100</th>
              <th className={s.num}>Stake</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => [
              <tr key={row.market} className={s.group}><th colSpan={5}>{MARKET_LABELS[row.market]}</th></tr>,
              ...(row.quotes ? row.quotes.map((q) => (
                <tr key={`${row.market}-${q.pick}`} className={cx(q.ev > 0 && s.value)}>
                  <td>{row.describe(q)}</td>
                  <td className={cx(s.num, "tabular")}>{fmtOdds(q.odds)}</td>
                  <td className={cx(s.num, "tabular")} title={`model alone: ${pct(q.raw, 1)}`}>
                    {pct(q.prob, 1)} <span className={cx(s.be, "muted")}>need {pct(q.breakEven, 1)}</span>
                  </td>
                  <td className={cx(s.num, "tabular", s.ev)}>{q.ev >= 0 ? "+" : "−"}${Math.abs(q.ev * 100).toFixed(2)}</td>
                  <td className={cx(s.num, "tabular")}>{q.stake > 0 ? pct(q.stake, 1) : "no bet"}</td>
                </tr>
              )) : [<tr key={`${row.market}-none`}><td colSpan={5} className="muted">Enter a line and odds (at or beyond ±100).</td></tr>]),
            ])}
          </tbody>
        </table>
      </div>
      <p className={cx("muted small", s.note)}>
        Win chances are the model's, moved toward the market by how much of its disagreement has been right in the
        backtest ({trustText}; 0% means its disagreements haven't paid, so only a better price than the market's can
        make a bet worthwhile). Stakes are shares of your bankroll at {pct(KELLY_FRACTION)} Kelly. Hover a win chance
        for the model's own number. For entertainment, not betting advice.
      </p>
    </div>
  );
}
