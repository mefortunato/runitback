import { useState } from "react";
import { Link } from "../../router";
import { api } from "../lib/api";
import { num, pct, seasonText } from "../lib/format";
import { links } from "../lib/links";
import { MARKET_LABELS, unitsLabel } from "../lib/markets";
import type { BetRow, Market, Metrics, Performance as PerformanceData, SportInfo, StakingPolicy } from "../lib/types";
import CalibrationChart from "../components/CalibrationChart";
import { useData } from "../useData";
import { cx } from "../cx";
import s from "./Performance.module.css";

// win rate needed to break even betting at -110
const BREAKEVEN = 110 / 210;
const fixed = (v: number | null | undefined, digits = 3) => (v === null || v === undefined ? "—" : v.toFixed(digits));
const record = (row: BetRow) => `${row.wins}-${row.losses}${row.pushes ? `-${row.pushes}` : ""}`;
const signedNum = (v: number, digits = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
const better = (a: number | undefined, b: number | undefined, lowerIsBetter = false) =>
  a !== undefined && b !== undefined && (lowerIsBetter ? a < b : a > b);

export default function Performance({ info }: { info: SportInfo }) {
  const { data, error } = useData(() => api.performance(info.sport), [info.sport]);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  return <PerformanceView info={info} perf={data} />;
}

function PerformanceView({ info, perf }: { info: SportInfo; perf: PerformanceData }) {
  const [chosen, setChosen] = useState<Market>("spread");
  const market = perf.overall.market;
  const model = perf.overall.model;
  const ats = perf.ats?.[0];
  const first = seasonText(perf.seasons[0], info.split_season);
  const last = seasonText(perf.seasons[perf.seasons.length - 1], info.split_season);
  const lower = info.date_label.toLowerCase();
  // the betting section's markets: those the backtest scored
  const markets: Market[] = [...(perf.ats ? ["spread" as const] : []), ...(perf.moneyline ? ["moneyline" as const] : []),
    ...(perf.totals ? ["total" as const] : [])];
  const m_ = markets.includes(chosen) ? chosen : (markets[0] ?? "spread");
  const betRows: BetRow[] = m_ === "moneyline" ? (perf.moneyline ?? []) : m_ === "total" ? (perf.totals?.record ?? []) : (perf.ats ?? []);
  const edgeLabel = (row: BetRow) => (m_ === "moneyline" ? `${Math.round(row.min_edge * 100)} pct pts` : `${row.min_edge} pts`);
  const staking = perf.staking?.[m_];
  const policyLabel = (p: StakingPolicy) =>
    p.policy === "flat" ? "Flat: 1 unit on every pick"
      : p.policy === "proportional" ? "Proportional to the edge"
      : p.policy === "learned" ? "Learned from past edges (Kelly)"
      : p.policy === "model_kelly" ? "Model's own probability (Kelly)"
      : `1 unit only if edge over ${p.policy.replace("edge>", "")}`;
  const edgeUnit = (e: number) => (m_ === "moneyline" ? `${(e * 100).toFixed(1)} pct pts` : `${e.toFixed(1)} pts`);
  const overCalibration = perf.over_calibration
    ? [{ label: "Model", slot: 1, bins: perf.over_calibration.model },
       ...(perf.over_calibration.market ? [{ label: "Vegas", slot: 2, bins: perf.over_calibration.market }] : [])]
    : [];
  const calibration = [{ label: "Model", slot: 1, bins: perf.calibration.model },
    ...(perf.calibration.market ? [{ label: "Vegas", slot: 2, bins: perf.calibration.market }] : [])];
  const seasons = Object.entries(perf.by_season).sort(([a], [b]) => Number(b) - Number(a));
  const phases = Object.entries(perf.by_phase);
  const tiles: { label: string; key: keyof Metrics; lower?: boolean; format: (v?: number) => string }[] = [
    { label: "Picks the winner", key: "accuracy", format: (v) => (v === undefined ? "—" : pct(v, 1)) },
    { label: "Brier score (lower is better)", key: "brier", lower: true, format: (v) => fixed(v) },
    { label: "Margin error, RMSE (points)", key: "rmse", lower: true, format: (v) => (v === undefined ? "—" : num(v, 1)) },
  ];
  const legend = (series: { label: string; slot: number }[]) => (
    <div className={cx(s.legend, "small")}>
      {series.map((x) => <span key={x.label}><span className={s.swatch} style={{ background: `var(--series-${x.slot})` }} /> {x.label}</span>)}
    </div>
  );

  return (
    <div className={s.root}>
      <h1>Model performance</h1>
      <p className={cx("muted", s.intro)}>
        Every number here is from games the model had not seen. For each season from {first} to {last}, a model trained
        only on earlier seasons predicted it one {lower} at a time, with each {lower}'s results added as they happened.
        These are the same predictions you play against in <Link to={links.challenge(info.sport)}>Beat the model</Link>.
        {market && ` Vegas is the closing line on the same ${model.n} games.`}
      </p>
      <div className={s.tiles}>
        {tiles.map((t) => {
          const m = model[t.key] as number | undefined;
          const v = market?.[t.key] as number | undefined;
          return (
            <div key={t.key} className={cx("card", s.tile)}>
              <div className="muted small">{t.label}</div>
              <div className={s.pair}>
                <div><div className={cx(s.big, better(m, v, t.lower) && s.win)}>{t.format(m)}</div><div className="muted small">Model</div></div>
                {market && <div><div className={cx(s.big, better(v, m, t.lower) && s.win)}>{t.format(v)}</div><div className="muted small">Vegas</div></div>}
              </div>
            </div>
          );
        })}
        {perf.totals && (
          <div className={cx("card", s.tile)}>
            <div className="muted small">Total points error, RMSE</div>
            <div className={s.pair}>
              <div><div className={cx(s.big, perf.totals.rmse.model < perf.totals.rmse.market && s.win)}>{num(perf.totals.rmse.model, 1)}</div><div className="muted small">Model</div></div>
              <div><div className={cx(s.big, perf.totals.rmse.market < perf.totals.rmse.model && s.win)}>{num(perf.totals.rmse.market, 1)}</div><div className="muted small">Vegas line</div></div>
            </div>
          </div>
        )}
        {ats && (
          <div className={cx("card", s.tile)}>
            <div className="muted small">Against the spread (model's side every game)</div>
            <div className={s.pair}>
              <div><div className={cx(s.big, (ats.win_pct ?? 0) > BREAKEVEN && s.win)}>{ats.win_pct === null ? "—" : pct(ats.win_pct, 1)}</div><div className="muted small">{record(ats)}</div></div>
              <div><div className={cx(s.big, "muted")}>{pct(BREAKEVEN, 1)}</div><div className="muted small">break-even at −110</div></div>
            </div>
          </div>
        )}
      </div>

      <div className="card section">
        <h2>By season</h2>
        <div className={s.scroll}>
          <table>
            <thead>
              <tr>
                <th>Season</th><th className={s.num}>Games</th><th className={s.num}>Accuracy</th>{market && <th className={s.num}>Vegas</th>}
                <th className={s.num}>Brier</th>{market && <th className={s.num}>Vegas</th>}
                <th className={s.num}>RMSE</th>{market && <th className={s.num}>Vegas</th>}
                {perf.ats_by_season && <th className={s.num}>ATS</th>}
                {perf.moneyline_by_season && <th className={s.num}>Moneyline</th>}
                {perf.totals_by_season && <th className={s.num}>Over/under</th>}
              </tr>
            </thead>
            <tbody>
              {seasons.map(([season, x]) => {
                const row = perf.ats_by_season?.[season];
                const ml = perf.moneyline_by_season?.[season];
                const ou = perf.totals_by_season?.[season];
                const n = cx(s.num, "tabular");
                return (
                  <tr key={season}>
                    <td>{seasonText(Number(season), info.split_season)}</td>
                    <td className={n}>{x.model.n}</td>
                    <td className={cx(n, better(x.model.accuracy, x.market?.accuracy) && s.win)}>{pct(x.model.accuracy, 1)}</td>
                    {x.market && <td className={cx(n, "muted")}>{pct(x.market.accuracy, 1)}</td>}
                    <td className={cx(n, better(x.model.brier, x.market?.brier, true) && s.win)}>{fixed(x.model.brier)}</td>
                    {x.market && <td className={cx(n, "muted")}>{fixed(x.market.brier)}</td>}
                    <td className={cx(n, better(x.model.rmse, x.market?.rmse, true) && s.win)}>{num(x.model.rmse, 1)}</td>
                    {x.market && <td className={cx(n, "muted")}>{num(x.market.rmse, 1)}</td>}
                    {row && <td className={n}>{record(row)} ({row.win_pct === null ? "—" : pct(row.win_pct, 1)})</td>}
                    {perf.moneyline_by_season && <td className={cx(n, ml && ml.roi > 0 && s.win)}>{ml ? `${record(ml)} (${unitsLabel(ml.units ?? 0)})` : "—"}</td>}
                    {perf.totals_by_season && <td className={cx(n, ou && ou.roi > 0 && s.win)}>{ou ? `${record(ou)} (${ou.win_pct === null ? "—" : pct(ou.win_pct, 1)})` : "—"}</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted small">Green marks where the model beat Vegas.</p>
      </div>

      <div className={cx("grid-2 section", s["grid-2"])}>
        <div className="card">
          <h2>Calibration</h2>
          <p className="muted small">
            When the model gives a team a 70% chance, does that team win about 70% of the time? Points on the diagonal are
            perfectly calibrated.
          </p>
          {legend(calibration)}
          <CalibrationChart series={calibration} />
          <details>
            <summary className="small">Table</summary>
            <table className="small">
              <thead><tr><th>Source</th><th className={s.num}>Predicted</th><th className={s.num}>Won</th><th className={s.num}>Games</th></tr></thead>
              <tbody>
                {calibration.map((x) => x.bins.map((b) => (
                  <tr key={`${x.label}-${b.predicted}`}>
                    <td>{x.label}</td><td className={cx(s.num, "tabular")}>{pct(b.predicted)}</td>
                    <td className={cx(s.num, "tabular")}>{pct(b.actual)}</td><td className={cx(s.num, "tabular")}>{b.n}</td>
                  </tr>
                )))}
              </tbody>
            </table>
          </details>
        </div>

        <div className={s.stack}>
          {markets.length > 0 && (
            <div className="card">
              <h2>Betting the model's edge</h2>
              {markets.length > 1 && (
                <div className={s.tabs} role="tablist" aria-label="Market">
                  {markets.map((m) => (
                    <button key={m} role="tab" aria-selected={m_ === m} className={cx(m_ === m && s.active)} onClick={() => setChosen(m)}>{MARKET_LABELS[m]}</button>
                  ))}
                </div>
              )}
              <p className="muted small">
                {m_ === "moneyline"
                  ? "Bet the team the model rates more likely to win than the market's odds imply (vig removed), when the gap is more than a few percentage points. Profit is at the posted moneyline: favorites pay less than a unit, underdogs more."
                  : m_ === "total"
                  ? "Bet the over when the model's total is above the line by more than a few points, the under when it is below. Profit is at the posted over/under odds."
                  : `Take the model's side against the spread only when it disagrees with Vegas by more than a few points. Above ${pct(BREAKEVEN, 1)} would make money at −110.`}
              </p>
              <div className={s.scroll}>
                <table>
                  <thead>
                    <tr><th>Edge over</th><th className={s.num}>Bets</th><th className={s.num}>W-L-P</th><th className={s.num}>Win %</th>
                      <th className={s.num}>95% range</th><th className={s.num}>Units (ROI)</th></tr>
                  </thead>
                  <tbody>
                    {betRows.map((row) => (
                      <tr key={row.min_edge}>
                        <td className="tabular">{edgeLabel(row)}</td>
                        <td className={cx(s.num, "tabular")}>{row.bets}</td>
                        <td className={cx(s.num, "tabular")}>{record(row)}</td>
                        <td className={cx(s.num, "tabular", m_ === "spread" && (row.win_pct ?? 0) > BREAKEVEN && s.win)}>{row.win_pct === null ? "—" : pct(row.win_pct, 1)}</td>
                        <td className={cx(s.num, "tabular", "muted")}>{row.ci95_low === null || row.ci95_high === null ? "—" : `${pct(row.ci95_low)}–${pct(row.ci95_high)}`}</td>
                        <td className={cx(s.num, "tabular", row.roi > 0 && s.win)}>{row.units === undefined ? "" : `${unitsLabel(row.units)} `}({pct(row.roi, 1)})</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {m_ === "spread" && perf.edge_correlation ? (
                <p className="muted small">
                  Correlation between the model's edge and how the game finished against the spread: {perf.edge_correlation.r.toFixed(3)} (95%
                  range {perf.edge_correlation.ci95_low.toFixed(3)} to {perf.edge_correlation.ci95_high.toFixed(3)}).{" "}
                  {perf.edge_correlation.ci95_low > 0 ? "Above zero: the model knows something the line does not." : "A range that includes zero means no reliable edge over the line."}
                </p>
              ) : m_ === "total" && perf.totals ? (
                <p className="muted small">
                  On {perf.totals.n} games the model's total missed by {num(perf.totals.rmse.model, 1)} points (RMSE), the line by {num(perf.totals.rmse.market, 1)}.
                  {perf.totals.brier.model !== undefined && ` Over/under Brier score: model ${fixed(perf.totals.brier.model)}${perf.totals.brier.market !== undefined ? `, Vegas ${fixed(perf.totals.brier.market)}` : ""}.`}
                  {` Correlation between the model's edge and how the total finished against the line: ${perf.totals.edge_correlation.r.toFixed(3)} (95% range ${perf.totals.edge_correlation.ci95_low.toFixed(3)} to ${perf.totals.edge_correlation.ci95_high.toFixed(3)}).`}
                </p>
              ) : m_ === "moneyline" && market ? (
                <p className="muted small">
                  The moneyline's win probabilities are the Vegas column of the Brier score and calibration above: the model beats
                  the market's odds only where its probabilities are better.
                </p>
              ) : null}
            </div>
          )}

          {staking && (
            <div className="card">
              <h2>Weighting bets by edge</h2>
              <p className="muted small">
                Flat betting risks one unit however small the edge. Here the stake grows with the edge, learned only from
                earlier seasons: each season is staked using a fit on the seasons before it, so nothing is peeked at. Stakes are
                relative units (average stake 1), so ROI is profit per unit staked and comparable with flat.
              </p>
              <div className={s.scroll}>
                <table>
                  <thead>
                    <tr><th>Staking</th><th className={s.num}>Bets</th><th className={s.num}>Staked</th><th className={s.num}>Profit</th>
                      <th className={s.num}>ROI</th><th className={s.num}>vs flat (95% range)</th></tr>
                  </thead>
                  <tbody>
                    {staking.policies.map((p) => (
                      <tr key={p.policy}>
                        <td>{policyLabel(p)}</td>
                        <td className={cx(s.num, "tabular")}>{p.bets}</td>
                        <td className={cx(s.num, "tabular")}>{num(p.staked, 0)}</td>
                        <td className={cx(s.num, "tabular", p.profit > 0 && s.win)}>{signedNum(p.profit)}</td>
                        <td className={cx(s.num, "tabular", p.roi > 0 && s.win)}>{pct(p.roi, 1)}</td>
                        <td className={cx(s.num, "tabular", "muted")}>
                          {p.policy === "flat" ? "—" : `${signedNum(p.roi_vs_flat * 100)} pts (${signedNum(p.roi_vs_flat_ci95_low * 100, 0)} to ${signedNum(p.roi_vs_flat_ci95_high * 100, 0)})`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="muted small">
                Scored on {staking.n} picks from {seasonText(staking.seasons[0], info.split_season)} to{" "}
                {seasonText(staking.seasons[staking.seasons.length - 1], info.split_season)}; the earlier seasons only train the stakes.
              </p>
              <details>
                <summary className="small">How the stake depends on the edge</summary>
                <p className="muted small">
                  The learned stake fits how often a pick has won at each edge (a logistic regression on the edge and the odds),
                  then bets the Kelly fraction <code>p − (1 − p) / b</code> at the pick's odds, where <code>p</code> is that win
                  chance and <code>b</code> the payout per unit. It is {staking.fraction === 0.5 ? "half" : `${staking.fraction}×`} Kelly
                  because the fit is noisy, scaled so the average stake is 1 unit, and capped at {staking.max_units} units. Below
                  break-even the stake is 0: that edge has not beaten the odds.
                </p>
                <table className="small">
                  <thead><tr><th>Edge</th><th className={s.num}>Units</th></tr></thead>
                  <tbody>
                    {staking.curve.map((c) => (
                      <tr key={c.edge}><td>{edgeUnit(c.edge)}</td><td className={cx(s.num, "tabular")}>{c.units === 0 ? "no bet" : num(c.units, 2)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </details>
            </div>
          )}

          {overCalibration.length > 0 && (
            <div className="card">
              <h2>Over/under calibration</h2>
              <p className="muted small">When the model gives the over a 60% chance, does the game go over about 60% of the time?</p>
              {legend(overCalibration)}
              <CalibrationChart series={overCalibration} event="over" />
            </div>
          )}

          <div className="card">
            <h2>Early season vs. the rest</h2>
            <div className={s.scroll}>
              <table>
                <thead>
                  <tr><th></th><th className={s.num}>Games</th><th className={s.num}>Accuracy</th><th className={s.num}>Brier</th>{market && <th className={s.num}>Vegas Brier</th>}</tr>
                </thead>
                <tbody>
                  {phases.map(([phase, p]) => (
                    <tr key={phase}>
                      <td>{phase.replace("dates", `${lower}s`)}</td>
                      <td className={cx(s.num, "tabular")}>{p.model.n}</td>
                      <td className={cx(s.num, "tabular")}>{pct(p.model.accuracy, 1)}</td>
                      <td className={cx(s.num, "tabular")}>{fixed(p.model.brier)}</td>
                      {p.market && <td className={cx(s.num, "tabular", "muted")}>{fixed(p.market.brier)}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
