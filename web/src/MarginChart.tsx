import { useState } from "react";
import type { SimResult } from "./api";

// Histogram of simulated final margins (home - away); single series.
export default function MarginChart({ r }: { r: SimResult }) {
  const [hover, setHover] = useState<number | null>(null);
  const entries = Object.entries(r.margin_hist).map(([k, v]) => [Number(k), v] as const);
  const W = 640, H = 170, L = 12, R = 12, T = 12, B = 26;
  const lo = -40, hi = 40, bw = (W - L - R) / (hi - lo + 1);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  const sx = (m: number) => L + (m - lo) * bw;
  const sh = (v: number) => (v / max) * (H - T - B);
  const hv = hover !== null ? entries.find(([m]) => m === hover) : undefined;
  return (
    <figure className="wp">
      <figcaption><span>Final margin, {r.home} minus {r.away} ({r.n} sims)</span></figcaption>
      <div className="wp-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Distribution of simulated final margins"
          onPointerLeave={() => setHover(null)}>
          <line x1={L} x2={W - R} y1={H - B} y2={H - B} className="grid" />
          {entries.map(([m, v]) => (
            <rect key={m} x={sx(m) + 1} width={Math.max(bw - 2, 1)} y={H - B - sh(v)} height={sh(v)} rx="1.5"
              className={`bar ${m > 0 ? "pos" : m < 0 ? "neg" : "tie"} ${hover === m ? "hot" : ""}`}
              onPointerEnter={() => setHover(m)} />
          ))}
          {[-35, -21, -14, -7, 0, 7, 14, 21, 35].map((m) => (
            <text key={m} x={sx(m) + bw / 2} y={H - 8} className="axis" textAnchor="middle">{m > 0 ? `+${m}` : m}</text>
          ))}
        </svg>
        {hv && (
          <div className="tip" style={{ left: `${((sx(hv[0]) + bw / 2) / W) * 100}%` }}>
            <strong>{((hv[1] / r.n) * 100).toFixed(1)}%</strong> of games
            <div className="muted">{hv[0] === 0 ? "Tie" : hv[0] > 0 ? `${r.home} by ${hv[0]}` : `${r.away} by ${-hv[0]}`}</div>
          </div>
        )}
      </div>
    </figure>
  );
}
