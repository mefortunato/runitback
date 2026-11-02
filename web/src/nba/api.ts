import { call as http } from "../http";
import type { SimResult, WinProb } from "../format";

export type { SimResult, WinProb };
export interface State {
  season: number; game_type: string; home: string; away: string;
  qtr: number; qsec: number; phase: "tip" | "live" | "free_throw";
  pos: string; fts_left: number; fts_total: number;
  home_score: number; away_score: number; home_timeouts: number; away_timeouts: number;
  over: boolean;
}
export interface Outcome { play_type: string; result: string; elapsed: number }
export interface Play { n: number; before: State; outcome: Outcome; text: string; after: State }
export interface Game { id: string; state: State; plays: Play[]; winprob: WinProb[]; legal_play_types: string[];
  new_plays?: Play[] }
export interface Teams {
  teams: string[]; names: Record<string, string>; seasons: number[]; model_loaded: boolean; error: string | null;
}
export type StepBody = { play_type?: string; count?: number; until?: "possession" | "period"; temperature?: number };

const call = <T,>(path: string, init?: RequestInit) => http<T>(`/api/nba${path}`, init);

export const api = {
  teams: () => call<Teams>("/teams"),
  newGame: (home: string, away: string, season: number, game_type: string) =>
    call<Game>("/games", { method: "POST", body: JSON.stringify({ home, away, season, game_type }) }),
  step: (id: string, body: StepBody) => call<Game>(`/games/${id}/step`, { method: "POST", body: JSON.stringify(body) }),
  undo: (id: string) => call<Game>(`/games/${id}/undo`, { method: "POST" }),
  winprob: (id: string, n = 120) => call<WinProb>(`/games/${id}/winprob?n=${n}`),
  simulate: (home: string, away: string, season: number, game_type: string, n: number) =>
    call<SimResult>("/simulate", { method: "POST", body: JSON.stringify({ home, away, season, game_type, n }) }),
};

// Seasons are named by the year they end in, as in the ESPN data: 2026 is the 2025-26 season.
export const seasonLabel = (y: number) => `${y - 1}–${String(y % 100).padStart(2, "0")}`;

export function situation(s: State): string {
  if (s.over) return "Final";
  if (s.phase === "tip") return s.qtr === 1 ? "Opening tip" : "Overtime tip";
  if (s.phase === "free_throw") {
    const k = s.fts_total - s.fts_left + 1;
    return `${s.pos} at the line · ${k} of ${s.fts_total}`;
  }
  return `${s.pos} ball`;
}

export const isShot = (t: string) => t === "shot2" || t === "shot3";
export const isMade = (r: string) => r === "made" || r === "made_fouled";
