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
  teamMap: (sport: string) => `/${sport}/map`,
  performance: (sport: string) => `/${sport}/performance`,
  generator: (sport: string) => `/${sport}/generator`,
};
