export interface State {
  season: number; game_type: string; home: string; away: string;
  qtr: number; qsec: number; phase: "kickoff" | "pat" | "scrimmage";
  pos: string; down: number; ytg: number; yl: number;
  home_score: number; away_score: number; home_timeouts: number; away_timeouts: number;
  over: boolean;
}
export interface Outcome { play_type: string; result: string; yards: number; elapsed: number }
export interface Play { n: number; before: State; outcome: Outcome; text: string; after: State }
export interface WinProb { n: number; qtr: number; qsec: number; home_win: number; away_win: number; tie: number;
  home_score_mean: number; away_score_mean: number }
export interface Game { id: string; state: State; plays: Play[]; winprob: WinProb[]; legal_play_types: string[];
  new_plays?: Play[] }
export interface SimResult {
  home: string; away: string; n: number; home_win: number; away_win: number; tie: number;
  home_score_mean: number; away_score_mean: number; margin_mean: number; total_mean: number;
  margin_quantiles: Record<string, number>; total_quantiles: Record<string, number>;
  margin_hist: Record<string, number>;
}
export interface SeasonGame {
  game_id: string; season: number; week: number; game_type: string; stage: string; round: string;
  home: string; away: string; home_score: number; away_score: number; winner: string | null;
  home_seed: number | null; away_seed: number | null; site: string;
}
export interface Standing {
  team: string; conference: string; division: string; wins: number; losses: number; ties: number;
  points_for: number; points_against: number; point_diff: number; win_pct: number;
  division_rank?: number; seed: number | null; playoff_team?: boolean; bye_week: number | null;
}
export interface ConferenceStandings {
  divisions: Record<string, Standing[]>;
  seeds: Standing[];
}
export interface SeasonResult {
  season: number; seed: number; strength_mode: string; regular_game_count: number;
  position_source: string | null; bye_weeks: Record<string, number>;
  playoff_game_count: number; playoff_team_count: number; games: SeasonGame[];
  standings: Record<string, ConferenceStandings>;
  playoffs: Array<{ round: string; games: SeasonGame[] }>; champion: string;
  super_bowl: SeasonGame; training_data_path?: string;
}
export interface SeasonJob {
  id: string; status: "running" | "complete" | "failed";
  progress: { stage: string; completed: number; total: number };
  result: SeasonResult | null; error: string | null;
}
export interface SeasonRequest {
  season: number; seed: number;
  temperature: number; strength_mode: "checkpoint" | "neutral" | "randomized";
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, {
    ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

export const api = {
  teams: () => call<{ teams: string[]; seasons: number[] }>("/teams"),
  newGame: (home: string, away: string, season: number, game_type: string) =>
    call<Game>("/games", { method: "POST", body: JSON.stringify({ home, away, season, game_type }) }),
  step: (id: string, body: { play_type?: string; count?: number; temperature?: number }) =>
    call<Game>(`/games/${id}/step`, { method: "POST", body: JSON.stringify(body) }),
  undo: (id: string) => call<Game>(`/games/${id}/undo`, { method: "POST" }),
  winprob: (id: string, n = 150) => call<WinProb>(`/games/${id}/winprob?n=${n}`),
  simulate: (home: string, away: string, season: number, game_type: string, n: number) =>
    call<SimResult>("/simulate", { method: "POST", body: JSON.stringify({ home, away, season, game_type, n }) }),
  startSeason: (request: SeasonRequest) =>
    call<SeasonJob>("/seasons", { method: "POST", body: JSON.stringify(request) }),
  seasonJob: (id: string) => call<SeasonJob>(`/seasons/${id}`),
};

export const clock = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
export const qtrName = (q: number) => (q <= 4 ? `Q${q}` : q === 5 ? "OT" : `OT${q - 4}`);
export const ordinal = (d: number) => ["", "1st", "2nd", "3rd", "4th"][d] || "";
export function spot(s: State): string {
  const opp = s.pos === s.home ? s.away : s.home;
  if (s.yl === 50) return "50";
  return s.yl < 50 ? `${opp} ${s.yl}` : `${s.pos} ${100 - s.yl}`;
}
export function situation(s: State): string {
  if (s.over) return "Final";
  if (s.phase === "kickoff") return `${s.pos} kickoff`;
  if (s.phase === "pat") return `${s.pos} try`;
  const dist = s.ytg >= s.yl ? "Goal" : String(s.ytg);
  return `${ordinal(s.down)} & ${dist} · ${spot(s)}`;
}
