import { useLayoutEffect, useRef, useState } from "react";
import { pct } from "../lib/format";
import type { CalibrationBin } from "../lib/types";
import { cx } from "../cx";
import s from "./CalibrationChart.module.css";

const M = { top: 12, right: 16, bottom: 40, left: 46 };
const ticks = [0, 0.25, 0.5, 0.75, 1];

// Predicted win probability (x) vs how often those teams won (y); the diagonal is perfect calibration.
export default function CalibrationChart({ series, event = "win" }: {
  series: { label: string; slot: number; bins: CalibrationBin[] }[];
  /** What the probability is of: a team winning, or the game going over. */
  event?: "win" | "over";
}) {
  const words = event === "over"
    ? { predicted: "Predicted over probability", actual: "Actual over rate", did: "over" }
    : { predicted: "Predicted win probability", actual: "Actual win rate", did: "won" };
  // drawn at the container's own width (square plot), so text stays 12px
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(420);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  const [hover, setHover] = useState<{ label: string; slot: number; bin: CalibrationBin } | null>(null);
  const W = Math.max(260, Math.round(width));
  const size = W - M.left - M.right;
  const H = size + M.top + M.bottom;
  const x = (p: number) => M.left + p * size;
  const y = (p: number) => M.top + (1 - p) * size;
  return (
    <div ref={box} className={cx(s.root, s.chart)}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Calibration of the ${words.predicted.toLowerCase()}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line className={s.grid} x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} />
            <line className={s.grid} x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} />
            <text className={s.tick} x={x(0) - 8} y={y(t)} textAnchor="end" dominantBaseline="middle">{pct(t)}</text>
            <text className={s.tick} x={x(t)} y={y(0) + 18} textAnchor="middle">{pct(t)}</text>
          </g>
        ))}
        <text className={s.axis} x={x(0.5)} y={H - 4} textAnchor="middle">{words.predicted}</text>
        <text className={s.axis} transform={`translate(12 ${y(0.5)}) rotate(-90)`} textAnchor="middle">{words.actual}</text>
        <line className={s.diagonal} x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
        {series.map((sr) => (
          <path key={sr.label} className={s.line} style={{ stroke: `var(--series-${sr.slot})` }}
            d={sr.bins.map((b, i) => `${i ? "L" : "M"}${x(b.predicted).toFixed(1)},${y(b.actual).toFixed(1)}`).join("")} />
        ))}
        {series.map((sr) => sr.bins.map((bin) => (
          <g key={`${sr.label}-${bin.predicted}`} role="presentation"
            onPointerEnter={() => setHover({ label: sr.label, slot: sr.slot, bin })} onPointerLeave={() => setHover(null)}>
            <circle className={s.hit} cx={x(bin.predicted)} cy={y(bin.actual)} r="12" />
            <circle className={s.dot} cx={x(bin.predicted)} cy={y(bin.actual)} r="5" style={{ fill: `var(--series-${sr.slot})` }} />
          </g>
        )))}
      </svg>
      {hover && (
        <div className={cx(s.tooltip, hover.bin.predicted > 0.55 && s.flip)}
          style={{ left: `${(x(hover.bin.predicted) / W) * 100}%`, top: `${(y(hover.bin.actual) / H) * 100}%` }}>
          <div className={s["tip-head"]}><span className={s.swatch} style={{ background: `var(--series-${hover.slot})` }} /> {hover.label}</div>
          <div>Predicted {pct(hover.bin.predicted)}, {words.did} {pct(hover.bin.actual)}</div>
          <div className="muted">{hover.bin.n} games</div>
        </div>
      )}
    </div>
  );
}
