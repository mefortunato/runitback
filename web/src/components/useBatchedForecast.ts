import { useCallback, useEffect, useRef, useState } from "react";
import { summarizeScores, type SimResult } from "../format";
import { errorText } from "../http";

export const FORECAST_GAMES = 1000;
// Batch schedule: two 2-game batches put first numbers on screen in under a second, then batches
// double from 20 up to 200 (the sampler's internal batch size). The model steps every game in a
// request together, so per-game cost falls steeply with batch size (on a 4-thread CPU: 1 game
// ~0.5s, 20 games ~1.6s, 200 games ~7s); the 2-game warm-up adds only ~0.3-1.4s to a full
// 1,000-game run. Sizes stay even so the NFL's opening-kickoff split is 50/50.
export const FORECAST_WARMUP_BATCHES = [2, 2];
export const FORECAST_FIRST_BATCH = 20;
export const FORECAST_MAX_BATCH = 200;

export function batchSchedule(total = FORECAST_GAMES, warmup = FORECAST_WARMUP_BATCHES,
  first = FORECAST_FIRST_BATCH, max = FORECAST_MAX_BATCH): number[] {
  const out: number[] = [];
  let done = 0, next = first;
  for (const b of warmup) { if (done >= total) break; out.push(Math.min(b, total - done)); done += out[out.length - 1]; }
  while (done < total) { const b = Math.min(next, max, total - done); out.push(b); done += b; next *= 2; }
  return out;
}

type Simulate = (n: number, signal: AbortSignal) => Promise<SimResult>;

// Runs a matchup forecast as a series of growing batches (up to `total` games), pooling each
// batch's final scores and publishing the running summary as it grows. `stop()` keeps what has
// been simulated so far and cancels the batch in flight.
export function useBatchedForecast(total = FORECAST_GAMES) {
  const [forecast, setForecast] = useState<SimResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ctrl = useRef<AbortController | null>(null);

  const stop = useCallback(() => { ctrl.current?.abort(); }, []);
  useEffect(() => stop, [stop]);  // cancel on unmount

  const start = useCallback(async (home: string, away: string, simulate: Simulate) => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setRunning(true);
    setError(null);
    setForecast(null);
    const scores: Array<[number, number]> = [];
    try {
      for (const n of batchSchedule(total)) {
        if (c.signal.aborted) break;
        const r = await simulate(n, c.signal);
        if (c.signal.aborted || !r.scores?.length) break;
        scores.push(...r.scores);
        setForecast(summarizeScores(home, away, scores));
      }
    } catch (e) {
      if (!c.signal.aborted) setError(errorText(e));
    } finally {
      if (ctrl.current === c) { ctrl.current = null; setRunning(false); }
    }
  }, [total]);

  return { forecast, running, error, start, stop, total };
}
