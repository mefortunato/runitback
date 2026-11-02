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
}
