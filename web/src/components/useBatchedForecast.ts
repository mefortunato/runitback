import { useCallback, useEffect, useRef, useState } from "react";
import { summarizeScores, type SimResult } from "../format";
import { errorText } from "../http";

export const FORECAST_GAMES = 1000;
export const FORECAST_BATCH = 100;

type Simulate = (n: number, signal: AbortSignal) => Promise<SimResult>;

// Runs a matchup forecast as a series of small batches (up to `total` games), pooling each
// batch's final scores and publishing the running summary as it grows. `stop()` keeps what has
// been simulated so far and cancels the batch in flight.
export function useBatchedForecast(total = FORECAST_GAMES, batch = FORECAST_BATCH) {
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
      while (scores.length < total && !c.signal.aborted) {
        const r = await simulate(Math.min(batch, total - scores.length), c.signal);
        if (c.signal.aborted || !r.scores?.length) break;
        scores.push(...r.scores);
        setForecast(summarizeScores(home, away, scores));
      }
    } catch (e) {
      if (!c.signal.aborted) setError(errorText(e));
    } finally {
      if (ctrl.current === c) { ctrl.current = null; setRunning(false); }
    }
  }, [total, batch]);

  return { forecast, running, error, start, stop, total };
}
