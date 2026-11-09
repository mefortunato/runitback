import s from "./TotalChart.module.css";

const WIDTH = 440, GUTTER = 56, PAD = 14, ROW = 26, TOP = 24;

// Combined points on one axis: the Vegas line, the model's total and the final, with the gap
// between line and model shaded.
export default function TotalChart({ line = null, model = null, result = null }: {
  line?: number | null; model?: number | null; result?: number | null;
}) {
  const rows = [
    { key: "vegas", label: "Vegas", value: line, text: line == null ? "" : `O/U ${line}` },
    { key: "model", label: "Model", value: model, text: model == null ? "" : model.toFixed(1) },
    { key: "final", label: "Final", value: result, text: result == null ? "" : String(result) },
  ].filter((r) => r.value !== null && r.value !== undefined) as { key: string; label: string; value: number; text: string }[];
  const values = rows.map((r) => r.value);
  // at least ~30 points of axis around the values, and 6 beyond the outermost
  const pad = Math.max(6, (30 - (Math.max(...values) - Math.min(...values))) / 2);
  const low = Math.floor((Math.min(...values) - pad) / 5) * 5;
  const high = Math.ceil((Math.max(...values) + pad) / 5) * 5;
  const height = TOP + rows.length * ROW + 18;
  const x = (v: number) => GUTTER + PAD + ((v - low) / (high - low)) * (WIDTH - GUTTER - 2 * PAD);
  const step = high - low > 40 ? 10 : 5;
  const ticks: number[] = [];
  for (let t = Math.ceil(low / step) * step; t <= high; t += step) ticks.push(t);
  const gap = line != null && model != null ? [Math.min(line, model), Math.max(line, model)] : null;
  const anchor = (v: number) => (x(v) > WIDTH - 90 ? "end" : "start");
  const textX = (v: number) => (anchor(v) === "end" ? x(v) - 10 : x(v) + 10);
  return (
    <svg viewBox={`0 0 ${WIDTH} ${height}`} role="img" aria-label="Total points: line, model and final" className={s.chart}>
      <text x={GUTTER + 4} y="12" className={s.side}>◀ under</text>
      <text x={WIDTH - 4} y="12" className={s.side} textAnchor="end">over ▶</text>
      {gap && rows.length >= 2 && (
        <rect x={x(gap[0])} y={TOP - 4} width={Math.max(2, x(gap[1]) - x(gap[0]))} height={2 * ROW} className={s.edge} rx="3" />
      )}
      {line != null && <line x1={x(line)} x2={x(line)} y1={TOP - 6} y2={TOP + rows.length * ROW - 2} className={s.zero} />}
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
      {ticks.map((t) => <text key={t} x={x(t)} y={height - 4} textAnchor="middle" className={s.tick}>{t}</text>)}
    </svg>
  );
}
