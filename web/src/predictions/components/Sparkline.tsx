import s from "./Sparkline.module.css";

/** Bar sparkline of margins, oldest to newest; wins up, losses down. */
export default function Sparkline({ values, width = 120, height = 34 }: { values: number[]; width?: number; height?: number }) {
  const max = Math.max(7, ...values.map((v) => Math.abs(v)));
  const step = values.length ? width / values.length : width;
  const mid = height / 2;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label="Recent margins">
      <line x1="0" x2={width} y1={mid} y2={mid} className={s.axis} />
      {values.map((v, i) => {
        const h = (Math.abs(v) / max) * (mid - 2);
        return <rect key={i} x={i * step + step * 0.2} y={v >= 0 ? mid - h : mid} width={step * 0.6} height={Math.max(1, h)} rx="1.5"
          className={v > 0 ? s.win : v < 0 ? s.loss : s.tie} />;
      })}
    </svg>
  );
}
