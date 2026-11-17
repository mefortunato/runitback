// Internal links of the predictions pages (and the generator tab next to them).
// Each sport lives under /<sport>: the predictions at /<sport>[/<season>/<date>],
// the play-by-play generator at /<sport>/generator.
const opt = (season?: number | null, date?: number | null) =>
  season == null ? "" : date == null ? `/${season}` : `/${season}/${date}`;

export const links = {
  home: () => "/",
  upcoming: (sport: string, season?: number | null, date?: number | null) => `/${sport}${opt(season, date)}`,
  recent: (sport: string, season?: number | null, date?: number | null) => `/${sport}/recent${opt(season, date)}`,
  game: (sport: string, gameId: string) => `/${sport}/games/${encodeURIComponent(gameId)}`,
  challenge: (sport: string) => `/${sport}/challenge`,
  rankings: (sport: string) => `/${sport}/rankings`,
  players: (sport: string) => `/${sport}/players`,
  teamMap: (sport: string) => `/${sport}/map`,
  performance: (sport: string) => `/${sport}/performance`,
  generator: (sport: string) => `/${sport}/generator`,
};

export type PageKey = "upcoming" | "recent" | "rankings" | "players" | "map" | "performance" | "challenge";

export interface SportPageLink {
  key: PageKey;
  label: string;
  to: string;
}

/** A sport's predictions pages, in menu order (only those its export has data for). Shared by the
 * sport's page tabs and the header search. */
export function sportPages(info: {
  sport: string; has_players?: boolean; has_team_map?: boolean; challenge_games: number; has_spread: boolean; markets?: unknown[];
}): SportPageLink[] {
  const s = info.sport;
  return [
    { key: "upcoming" as const, label: "Upcoming", to: links.upcoming(s) },
    { key: "recent" as const, label: "Recent", to: links.recent(s) },
    { key: "rankings" as const, label: "Rankings", to: links.rankings(s) },
    ...(info.has_players ? [{ key: "players" as const, label: "Players", to: links.players(s) }] : []),
    ...(info.has_team_map ? [{ key: "map" as const, label: "Team map", to: links.teamMap(s) }] : []),
    ...(info.challenge_games ? [{ key: "performance" as const, label: "Performance", to: links.performance(s) }] : []),
    ...(info.challenge_games && (info.has_spread || info.markets?.length)
      ? [{ key: "challenge" as const, label: "Beat the model", to: links.challenge(s) }] : []),
  ];
}
