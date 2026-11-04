import { useCallback, useEffect, useRef, useState } from "react";
import { summarizeScores, type SimResult } from "../format";
import { errorText } from "../http";

export const FORECAST_GAMES = 1000;
// Batches start small so the first numbers show up quickly, then double: the model steps every
// game in a request together, so a 200-game batch costs about as much as a 20-game one (200 is
// the sampler's internal batch size). Sizes stay even so the NFL's opening-kickoff split is 50/50.
export const FORECAST_FIRST_BATCH = 20;
export const FORECAST_MAX_BATCH = 200;

type Simulate = (n: number, signal: AbortSignal) => Promise<SimResult>;

// Runs a matchup forecast as a series of growing batches (up to `total` games), pooling each
// batch's final scores and publishing the running summary as it grows. `stop()` keeps what has
// been simulated so far and cancels the batch in flight.
export function useBatchedForecast(total = FORECAST_GAMES, first = FORECAST_FIRST_BATCH,
  max = FORECAST_MAX_BATCH) {
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
    let batch = first;
    try {
      while (scores.length < total && !c.signal.aborted) {
        const r = await simulate(Math.min(batch, total - scores.length), c.signal);
        batch = Math.min(batch * 2, max);
        if (c.signal.aborted || !r.scores?.length) break;
        scores.push(...r.scores);
        setForecast(summarizeScores(home, away, scores));
      }
    } catch (e) {
      if (!c.signal.aborted) setError(errorText(e));
    } finally {
      if (ctrl.current === c) { ctrl.current = null; setRunning(false); }
    }
  }, [total, first, max]);

  return { forecast, running, error, start, stop, total };
}
