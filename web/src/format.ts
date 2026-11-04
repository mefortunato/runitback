// Formatting shared by every sport.
export const clock = (s: number) => {
  if (s < 60 && s % 1 !== 0) return s.toFixed(1);
  const t = Math.round(s);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};
// Periods 1-4 are quarters in both sports; 5 is OT, then 2OT, 3OT, ...
export const qtrName = (q: number) => (q <= 4 ? `Q${q}` : q === 5 ? "OT" : `${q - 4}OT`);
export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;

export interface WinProb {
  n: number; qtr: number; qsec: number; home_win: number; away_win: number; tie: number;
  home_score_mean: number; away_score_mean: number;
}
export interface SimResult {
  home: string; away: string; n: number; home_win: number; away_win: number; tie: number;
  home_score_mean: number; away_score_mean: number; margin_mean: number; total_mean: number;
  margin_quantiles: Record<string, number>; total_quantiles: Record<string, number>;
  margin_hist: Record<string, number>;
  scores?: Array<[number, number]>;
}

// numpy.quantile with its default linear interpolation, on an already-sorted array
const quantile = (sorted: number[], q: number) => {
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
};
const quantiles = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  return Object.fromEntries([0.05, 0.25, 0.5, 0.75, 0.95].map((q) => [String(q), quantile(s, q)]));
};
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

// Summarize final scores [(home, away), ...] the same way the backend's summarize_scores does,
// so batches of simulated games can be pooled in the browser.
export function summarizeScores(home: string, away: string, scores: Array<[number, number]>): SimResult {
  const h = scores.map((s) => s[0]), a = scores.map((s) => s[1]);
  const m = scores.map(([x, y]) => x - y), t = scores.map(([x, y]) => x + y);
  const hist: Record<string, number> = {};
  for (const v of m) { const k = Math.max(-40, Math.min(40, v)); hist[k] = (hist[k] ?? 0) + 1; }
  return {
    home, away, n: scores.length,
    home_win: m.filter((v) => v > 0).length / m.length,
    away_win: m.filter((v) => v < 0).length / m.length,
    tie: m.filter((v) => v === 0).length / m.length,
    home_score_mean: mean(h), away_score_mean: mean(a), margin_mean: mean(m), total_mean: mean(t),
    margin_quantiles: quantiles(m), total_quantiles: quantiles(t), margin_hist: hist,
  };
}
