import { call as http } from "../http";
import type { Game, SimResult, WinProb } from "../nfl/api";

export { clock, qtrName, situation } from "../nfl/api";
export type { Game, SimResult, WinProb };

/** A school: its ESPN team id (what the API takes) and its name (what it shows). */
export interface Team { id: string; name: string }

const call = <T,>(path: string, init?: RequestInit) => http<T>(`/api/cfb${path}`, init);

export const api = {
  teams: () => call<{ teams: Team[]; seasons: number[] }>("/teams"),
  newGame: (home: string, away: string, season: number, game_type: string) =>
    call<Game>("/games", { method: "POST", body: JSON.stringify({ home, away, season, game_type }) }),
  step: (id: string, body: { play_type?: string; count?: number; temperature?: number }) =>
    call<Game>(`/games/${id}/step`, { method: "POST", body: JSON.stringify(body) }),
  undo: (id: string) => call<Game>(`/games/${id}/undo`, { method: "POST" }),
  winprob: (id: string, n = 150) => call<WinProb>(`/games/${id}/winprob?n=${n}`),
  simulate: (home: string, away: string, season: number, game_type: string, n: number,
    opts: { include_scores?: boolean; signal?: AbortSignal } = {}) =>
    call<SimResult>("/simulate", {
      method: "POST", signal: opts.signal,
      body: JSON.stringify({ home, away, season, game_type, n, include_scores: opts.include_scores ?? false }),
    }),
};
