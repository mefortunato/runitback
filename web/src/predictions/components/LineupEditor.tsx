import { useState } from "react";
import { marginLabel, pct, signed } from "../lib/format";
import {
  columnPoints, initialLineup, playerIndex, playerPoints, ruleOut, sameLineup, setStarter,
  type LineupState,
} from "../lib/lineups";
import type { GameHeader, GameLineups, Prediction, Side } from "../lib/types";
import TeamBadge from "./TeamBadge";
import { cx } from "../cx";
import s from "./LineupEditor.module.css";

// Both teams' projected lineups (published depth chart minus players ruled out), each player's
// value in points per game over a replacement-level player at his slot, and controls to change
// starters or rule players out. The page re-prices the game for the edited lineups
// (lib/lineups.ts: the same arithmetic as the model's lineup terms).
/** Points summed over the value columns whose weight on the total passes `pick`: offense scores
 * (positive), defense prevents (negative), special teams moves neither. */
const sumWhere = (points: number[], totals: number[], pick: (w: number) => boolean) =>
  points.reduce((a, p, i) => (pick(totals[i] ?? 0) ? a + p : a), 0);

export default function LineupEditor({ game, lineups, state, onChange, base, edited }: {
  game: GameHeader; lineups: GameLineups; state: LineupState; onChange: (next: LineupState) => void;
  /** The published prediction and the edited lineups' (the same when nothing changed). */
  base: Prediction; edited: Prediction;
}) {
  const [side, setSide] = useState<Side>("home");
  const team = side === "home" ? game.home : game.away;
  const lineup = state[side];
  const points = columnPoints(lineups, side, lineup);
  const changed = (["home", "away"] as Side[]).filter((x) => !sameLineup(state[x], initialLineup(lineups, x)));
  const update = (next: LineupState[Side]) => onChange({ ...state, [side]: next });
  const reset = () => onChange({ home: initialLineup(lineups, "home"), away: initialLineup(lineups, "away") });
  const groupTotal = (group: string) => {
    const players = playerIndex(lineups, side, group);
    return lineup[group].active.reduce((sum, id, rank) => sum + playerPoints(lineups, group, rank, players.get(id)), 0);
  };
  const perPoint = lineups.weights.margin[0] ?? 0;
  return (
    <div className="card">
      <div className={s.head}>
        <h2 className={s.title}>Lineups</h2>
        <div className={s.tabs} role="tablist">
          {(["away", "home"] as Side[]).map((x) => (
            <button key={x} role="tab" aria-selected={side === x} className={cx(s.tab, side === x && s.active)} onClick={() => setSide(x)}>
              <TeamBadge team={x === "home" ? game.home : game.away} size={20} label="abbr" />
              {changed.includes(x) && <span className={s.dot} aria-label="edited" />}
            </button>
          ))}
        </div>
      </div>
      <p className="muted small">
        Projected from the latest depth charts and injury reports. Values are points per game over a
        replacement-level player at that slot; the model moves its margin {perPoint.toFixed(2)} points
        per point of value changed. Change a starter or rule a player out to re-price the game.
      </p>
      <div className={cx(s.effect, changed.length ? s.on : null)} aria-live="polite">
        {changed.length ? (
          <>
            <div>
              <span className="muted">Spread </span>
              <span className="tabular">{marginLabel(game.home, game.away, base.margin)} → <b>{marginLabel(game.home, game.away, edited.margin)}</b></span>
            </div>
            <div>
              <span className="muted">{game.home.team} win </span>
              <span className="tabular">{pct(base.win_prob)} → <b>{pct(edited.win_prob)}</b></span>
            </div>
            {base.total != null && edited.total != null && (
              <div>
                <span className="muted">Total </span>
                <span className="tabular">{base.total.toFixed(1)} → <b>{edited.total.toFixed(1)}</b></span>
              </div>
            )}
            <button onClick={reset}>Reset lineups</button>
          </>
        ) : <span className="muted">The model's prediction for these lineups is the one above.</span>}
      </div>
      <div className={s.groups}>
        {lineups.groups.map((g) => {
          const players = playerIndex(lineups, side, g.key);
          const slots = lineups.slots[g.key]?.length ?? 0;
          const { active, out } = lineup[g.key];
          const starters = active.slice(0, slots);
          if (!players.size) return null;
          const option = (id: string) => {
            const p = players.get(id)!;
            return <option key={id} value={id}>{p.name}{p.pos ? ` (${p.pos})` : ""}</option>;
          };
          return (
            <section key={g.key} className={s.group}>
              <div className={s.groupHead}>
                <span>{g.label}</span>
                <span className={cx("tabular", s.pts)}>{signed(Math.abs(groupTotal(g.key)) < 0.05 ? 0 : groupTotal(g.key))}</span>
              </div>
              {starters.map((id, rank) => {
                const p = players.get(id)!;
                const raw = playerPoints(lineups, g.key, rank, p);
                const pts = Math.abs(raw) < 0.05 ? 0 : raw;   // no "−0.0"
                return (
                  <div key={rank} className={s.row}>
                    <span className={cx("muted small", s.slot)}>{g.key}{slots > 1 ? rank + 1 : ""}</span>
                    <select
                      className={s.select}
                      aria-label={`${team.team} ${g.label} ${rank + 1}`}
                      value={id}
                      onChange={(e) => update(setStarter(lineup, g.key, rank, e.target.value))}
                    >
                      <optgroup label="Starting">{starters.map(option)}</optgroup>
                      {active.length > slots && <optgroup label="Backups">{active.slice(slots).map(option)}</optgroup>}
                      {out.length > 0 && <optgroup label="Ruled out">{out.map(option)}</optgroup>}
                    </select>
                    <span className={cx("tabular small", s.pts, pts > 0 ? s.plus : pts < 0 ? s.minus : null)}>{signed(pts)}</span>
                    <button className={s.out} onClick={() => update(ruleOut(lineup, g.key, id))} title={`Rule ${p.name} out`}>Out</button>
                  </div>
                );
              })}
              {out.length > 0 && (
                <div className={cx("muted small", s.outList)}>
                  Out: {out.map((id) => players.get(id)?.name ?? id).join(", ")}
                </div>
              )}
            </section>
          );
        })}
      </div>
      <p className="muted small">
        {team.team} lineup value: offense {signed(sumWhere(points, lineups.weights.total, (w) => w > 0))}, defense{" "}
        {signed(sumWhere(points, lineups.weights.total, (w) => w < 0))}
        {lineups.weights.total.some((w) => w === 0) && <>, kicking {signed(sumWhere(points, lineups.weights.total, (w) => w === 0))}</>}{" "}
        points per game.
      </p>
    </div>
  );
}
