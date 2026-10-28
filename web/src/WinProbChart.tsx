import { useState } from "react";
import type { WinProb } from "./api";
import { clock, qtrName } from "./api";

// Single series: home win probability by play number, 50% reference line.
export default function WinProbChart({ data, home, away, total }: {
  data: WinProb[]; home: string; away: string; total: number;
}) {
  const [hover, setHover] = useState<WinProb | null>(null);
  const W = 640, H = 180, L = 40, R = 12, T = 12, B = 24;
  const pts = [...data].sort((a, b) => a.n - b.n);
  const maxN = Math.max(total, 160, ...pts.map((p) => p.n));
  const sx = (n: number) => L + (n / maxN) * (W - L - R);
  const sy = (p: number) => T + (1 - p) * (H - T - B);
  const pHome = (w: WinProb) => w.home_win + w.tie / 2;
  const path = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.n).toFixed(1)},${sy(pHome(p)).toFixed(1)}`).join("");
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const n = ((e.clientX - r.left) / r.width * W - L) / (W - L - R) * maxN;
    let best: WinProb | null = null;
    for (const p of pts) if (!best || Math.abs(p.n - n) < Math.abs(best.n - n)) best = p;
    setHover(best);
  };
  return (
    <figure className="wp">
      <figcaption>
        <span>{home} win probability</span>
        {pts.length > 0 && <strong>{Math.round(pHome(pts[pts.length - 1]) * 100)}%</strong>}
      </figcaption>
      <div className="wp-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} onPointerMove={onMove} onPointerLeave={() => setHover(null)}
          role="img" aria-label={`${home} win probability over the game`}>
          {[0, 0.25, 0.5, 0.75, 1].map((g) => (
            <g key={g}>
              <line x1={L} x2={W - R} y1={sy(g)} y2={sy(g)} className={g === 0.5 ? "ref" : "grid"} />
              <text x={L - 6} y={sy(g) + 4} className="axis" textAnchor="end">{g * 100}%</text>
            </g>
          ))}
          <text x={L + 4} y={T + 10} className="axis">{home}</text>
          <text x={L + 4} y={H - B - 4} className="axis">{away}</text>
          <text x={W - R} y={H - 6} className="axis" textAnchor="end">play #</text>
          {path && <path d={path} className="wp-line" />}
          {pts.length > 0 && (
            <circle cx={sx(pts[pts.length - 1].n)} cy={sy(pHome(pts[pts.length - 1]))} r="4" className="wp-dot" />
          )}
          {hover && (
            <g>
              <line x1={sx(hover.n)} x2={sx(hover.n)} y1={T} y2={H - B} className="cross" />
              <circle cx={sx(hover.n)} cy={sy(pHome(hover))} r="4" className="wp-dot" />
            </g>
          )}
        </svg>
        {hover && (
          <div className="tip" style={{ left: `${(sx(hover.n) / W) * 100}%` }}>
            <strong>{Math.round(pHome(hover) * 100)}%</strong> {home} win
            <div className="muted">Play {hover.n} · {qtrName(hover.qtr)} {clock(hover.qsec)}</div>
            <div className="muted">Proj. {hover.home_score_mean.toFixed(1)}–{hover.away_score_mean.toFixed(1)}</div>
          </div>
        )}
      </div>
    </figure>
  );
}
