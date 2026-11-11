import { useState } from "react";
import { api } from "../lib/api";
import type { PlayerGroup, PlayerRanks, SportInfo } from "../lib/types";
import { remember, remembered, useData } from "../useData";
import { cx } from "../cx";
import s from "./Players.module.css";

const SHOWN = 25;
const KEY = "players-position";
const signed = (v: number, digits: number) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;

export default function Players({ info }: { info: SportInfo }) {
  const { data, error } = useData(() => api.players(info.sport), [info.sport]);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  return <PlayersView info={info} ranks={data} />;
}

function PlayersView({ info, ranks }: { info: SportInfo; ranks: PlayerRanks }) {
  const [chosen, setChosen] = useState(() => remembered<string>(KEY, ""));
  const [all, setAll] = useState(false);
  const group = ranks.positions.find((p) => p.key === chosen) ?? ranks.positions[0];
  if (!group) return <div><h1>Player rankings</h1><div className="card">No players yet.</div></div>;
  const pick = (key: string) => { setChosen(key); setAll(false); remember(KEY, key); };

  return (
    <div>
      <h1>Player rankings</h1>
      <p className={cx("muted", s.intro)}>
        Every {info.label} player's value this season, ranked within a position: {ranks.unit}, from the
        model's player values after week {ranks.through}. Players with very little playing time are left out.
      </p>
      <nav className={s.filters} aria-label="Position">
        {ranks.positions.map((p) => (
          <button key={p.key} className={cx("pill", p.key === group.key && "accent")} aria-pressed={p.key === group.key}
            onClick={() => pick(p.key)}>{p.key}</button>
        ))}
      </nav>
      <PositionChart group={group} ranks={ranks} all={all} onAll={() => setAll(true)} />
    </div>
  );
}

function PositionChart({ group, ranks, all, onAll }: { group: PlayerGroup; ranks: PlayerRanks; all: boolean; onAll: () => void }) {
  const rows = all ? group.players : group.players.slice(0, SHOWN);
  const values = group.players.map((p) => p.value);
  const hi = Math.max(0, ...values);
  const lo = Math.min(0, ...values);
  const span = hi - lo || 1;
  const zero = (-lo / span) * 100;   // where zero sits along the track, in %
  return (
    <section className="card">
      <h2 className={s.title}>{group.label} <span className="muted">· {ranks.season} season, through week {ranks.through}</span></h2>
      <p className={cx("muted", s.note)}>Bars: {ranks.unit}. Rate column: {ranks.detail}.</p>
      <table className={s.table}>
        <thead>
          <tr><th className={s.rank}>#</th><th>Player</th><th className={s.barhead}>Value</th>
            <th className={s.num}>Rate</th><th className={s.num}>Games</th></tr>
        </thead>
        <tbody>
          {rows.map((p) => {
            const left = p.value >= 0 ? zero : zero - (-p.value / span) * 100;
            const width = (Math.abs(p.value) / span) * 100;
            return (
              <tr key={p.id}>
                <td className={cx(s.rank, "muted")}>{p.rank}</td>
                <td className={s.player}><span className={s.name}>{p.name}</span> <span className="muted">{p.team}</span></td>
                <td><div className={s.barcell}>
                  <div className={s.track} role="img" aria-label={`${p.name}: ${signed(p.value, 1)}`}>
                    <div className={cx(s.bar, p.value < 0 && s.neg)} style={{ left: `${left}%`, width: `${width}%` }} />
                    {lo < 0 && <div className={s.zero} style={{ left: `${zero}%` }} />}
                  </div>
                  <span className={s.value}>{signed(p.value, 1)}</span>
                </div></td>
                <td className={cx(s.num, "muted")}>{signed(p.rate, 3)}</td>
                <td className={cx(s.num, "muted")}>{p.games}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!all && group.players.length > SHOWN && (
        <button className={cx("pill", s.more)} onClick={onAll}>Show all {group.players.length}</button>
      )}
    </section>
  );
}
