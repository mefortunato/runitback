import { marginLabel, pct, signed, spreadLabel } from "../lib/format";
import { americanOdds, pickLabel, totalLabel } from "../lib/markets";
import type { GameHeader, GameResult, Prediction, Side } from "../lib/types";
import { cx } from "../cx";
import s from "./MarketTable.module.css";

interface Row {
  market: string; vegas: string; model: string; pick: string; result: string | null;
  /** Whether the model's pick won, once played (null: no pick, or a push). */
  hit: boolean | null;
}

// One row per market: the line, the model's number, its pick and, once played, how the market
// finished. Sports without a point spread (hockey, baseball) have no spread row.
export default function MarketTable({ game, prediction, result = null, hasSpread = true }: {
  game: GameHeader; prediction: Prediction | null; result?: GameResult | null; hasSpread?: boolean;
}) {
  const graded = (pick: string | null | undefined, outcome: string | null | undefined) =>
    !result || !pick || !outcome || outcome === "push" ? null : pick === outcome;
  const team = (side: Side) => (side === "home" ? game.home : game.away);
  const p = prediction;
  // the moneyline favorite by the market (home on a pick'em)
  const fav: Side = p?.market_prob != null && p.market_prob < 0.5 ? "away" : "home";
  const prob = (q: number | null | undefined, side: Side) => (q === null || q === undefined ? null : side === "home" ? q : 1 - q);
  const ml = game.moneyline;
  const vegasFav = prob(p?.market_prob, fav);
  const modelFav = prob(p?.win_prob, fav);
  const rows: Row[] = [];
  if (hasSpread) {
    rows.push({
      market: "Spread",
      vegas: spreadLabel(game.home, game.away, game.spread),
      model: p ? marginLabel(game.home, game.away, p.margin) : "—",
      pick: p?.ats_pick && p.edge !== null ? `${pickLabel(p.ats_pick, game.home, game.away)} (${signed(Math.abs(p.edge))})` : "—",
      result: result ? (result.covered === "push" ? "Push" : result.covered ? `${team(result.covered).team} covered` : "—") : null,
      hit: graded(p?.ats_pick, result?.covered),
    });
  }
  if (ml) {
    rows.push({
      market: "Moneyline",
      vegas: `${team(fav).team} ${americanOdds(ml[fav])}${vegasFav === null ? "" : ` (${pct(vegasFav)})`}`,
      model: modelFav === null ? "—" : `${team(fav).team} ${pct(modelFav)}`,
      pick: p?.ml_pick && p.ml_edge != null
        ? `${pickLabel(p.ml_pick, game.home, game.away)} ${americanOdds(ml[p.ml_pick])} (+${pct(Math.abs(p.ml_edge), 1)})` : "—",
      result: result ? (result.winner === "push" ? "Tie" : result.winner ? `${team(result.winner).team} won` : "—") : null,
      hit: graded(p?.ml_pick, result?.winner),
    });
  }
  if (game.total_line != null || p?.total != null) {
    rows.push({
      market: "Over/under",
      vegas: totalLabel(game.total_line),
      model: p?.total == null ? "—" : p.total.toFixed(1),
      pick: p?.ou_pick && p.total_edge != null ? `${pickLabel(p.ou_pick, game.home, game.away)} (${signed(p.total_edge)})` : "—",
      result: result
        ? result.total == null ? "—" : `${result.total}${result.went ? ` · ${result.went === "push" ? "push" : result.went}` : ""}`
        : null,
      hit: graded(p?.ou_pick, game.total_line == null ? null : result?.went),
    });
  }
  const played = result !== null;
  // Cells wrap at spaces on narrow cards; the pick's last word stays glued to its ✓/✗.
  const lead = (t: string) => t.slice(0, t.lastIndexOf(" ") + 1);
  const last = (t: string) => t.slice(t.lastIndexOf(" ") + 1);
  return (
    <div className={cx(s.root, s.scroll)}>
      <table className={cx(s.markets, "small", played && s.played)}>
        <thead>
          <tr><th></th><th>Vegas</th><th>Model</th><th>Model pick</th>{played && <th>Result</th>}</tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.market}>
              <th scope="row">{row.market}</th>
              <td className="tabular" data-label="Vegas">{row.vegas}</td>
              <td className="tabular" data-label="Model">{row.model}</td>
              <td className={cx("tabular", s.pick)} data-label="Model pick">
                {lead(row.pick)}
                <span className={s.nowrap}>
                  {last(row.pick)}
                  {row.hit !== null && (
                    <span className={cx(s.mark, row.hit ? s.hit : s.miss)}>
                      {row.hit ? "✓" : "✗"}<span className={s.sr}>{row.hit ? " won" : " lost"}</span>
                    </span>
                  )}
                </span>
              </td>
              {played && <td className="tabular" data-label="Result">{row.result}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
