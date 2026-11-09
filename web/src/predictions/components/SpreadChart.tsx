import { marginLabel, spreadLabel } from "../lib/format";
import type { Team } from "../lib/types";
import s from "./SpreadChart.module.css";

const WIDTH = 440, GUTTER = 56, PAD = 14, ROW = 26, TOP = 24;

// All values are the home team's margin: right of center favors the home team.
export default function SpreadChart({ home, away, spread = null, model = null, result = null }: {
  home: Team; away: Team; spread?: number | null; model?: number | null; result?: number | null;
}) {
  const rows = [
    { key: "vegas", label: "Vegas", value: spread, text: spreadLabel(home, away, spread) },
    { key: "model", label: "Model", value: model, text: model === null ? "" : marginLabel(home, away, model) },
    { key: "final", label: "Final", value: result, text: result === null ? "" : marginLabel(home, away, result) },
  ].filter((r) => r.value !== null && r.value !== undefined) as { key: string; label: string; value: number; text: string }[];
  const range = Math.max(10, Math.ceil(Math.max(0, ...rows.map((r) => Math.abs(r.value))) + 3));
  const height = TOP + rows.length * ROW + 18;
  const x = (v: number) => GUTTER + PAD + ((v + range) / (2 * range)) * (WIDTH - GUTTER - 2 * PAD);
  const ticks = [-14, -7, 0, 7, 14].filter((t) => Math.abs(t) < range).concat(range > 21 ? [-21, 21] : []);
  const edge = spread !== null && model !== null ? [Math.min(spread, model), Math.max(spread, model)] : null;
  const anchor = (v: number) => (x(v) > WIDTH - 110 ? "end" : "start");
  const textX = (v: number) => (anchor(v) === "end" ? x(v) - 10 : x(v) + 10);
  return (
    <svg viewBox={`0 0 ${WIDTH} ${height}`} role="img" aria-label="Spread versus model margin" className={s.chart}>
      {/* which side favors whom */}
      <rect x={GUTTER} y={TOP - 6} width={x(0) - GUTTER} height={rows.length * ROW + 4} fill={away.color ?? "var(--muted)"} opacity="0.07" />
      <rect x={x(0)} y={TOP - 6} width={WIDTH - x(0)} height={rows.length * ROW + 4} fill={home.color ?? "var(--muted)"} opacity="0.07" />
      <text x={GUTTER + 4} y="12" className={s.side}>◀ {away.team} favored</text>
      <text x={WIDTH - 4} y="12" className={s.side} textAnchor="end">{home.team} favored ▶</text>
      {edge && rows.length >= 2 && (
        <rect x={x(edge[0])} y={TOP - 4} width={Math.max(2, x(edge[1]) - x(edge[0]))} height={2 * ROW} className={s.edge} rx="3" />
      )}
      <line x1={x(0)} x2={x(0)} y1={TOP - 6} y2={TOP + rows.length * ROW - 2} className={s.zero} />
      {rows.map((row, i) => {
        const y = TOP + i * ROW + ROW / 2 - 4;
        return (
          <g key={row.key}>
            <text x={GUTTER - 6} y={y + 4} textAnchor="end" className={s["row-label"]}>{row.label}</text>
            <line x1={GUTTER + PAD} x2={WIDTH - PAD} y1={y} y2={y} className={s.track} />
            {row.key === "vegas" ? (
              <rect x={x(row.value) - 6} y={y - 6} width="12" height="12" transform={`rotate(45 ${x(row.value)} ${y})`} className={`${s.marker} ${s.vegas}`} />
            ) : row.key === "model" ? (
              <circle cx={x(row.value)} cy={y} r="7" className={`${s.marker} ${s.model}`} />
            ) : (
              <circle cx={x(row.value)} cy={y} r="6" className={`${s.marker} ${s.final}`} />
            )}
            <text x={textX(row.value)} y={y + 4} textAnchor={anchor(row.value)} className={s.value}>{row.text}</text>
          </g>
        );
      })}
      {ticks.map((t) => (
        <text key={t} x={x(t)} y={height - 4} textAnchor="middle" className={s.tick}>{t === 0 ? "even" : Math.abs(t)}</text>
      ))}
    </svg>
  );
}
