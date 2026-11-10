// Every sport the site knows, in nav order (as the predictions export orders them:
// runitback.wager.web.views.BUILTIN_VIEWS). /<sport>/... shows its predictions pages when
// `runitback-wager web export` wrote them; /<sport>/generator its play-by-play generator, for the
// sports with a simulator served by the API (runitback.server).
export interface SportDef {
  id: string;
  label: string;
  /** Has a play-by-play generator page (a simulator and an API under /api/<sport>). */
  generator: boolean;
}

export const SPORTS: SportDef[] = [
  { id: "nfl", label: "NFL", generator: true },
  { id: "cfb", label: "CFB", generator: false },
  { id: "nba", label: "NBA", generator: true },
  { id: "cbb", label: "CBB", generator: false },
  { id: "wnba", label: "WNBA", generator: false },
  { id: "wcbb", label: "WCBB", generator: false },
  { id: "nhl", label: "NHL", generator: false },
  { id: "mlb", label: "MLB", generator: false },
];

export const sportById = (id: string | undefined) => SPORTS.find((s) => s.id === id);

/** The sports to list in the nav: those with a generator, plus those with exported predictions
 * (`exported`; undefined while sports.json loads, or when it can't be). */
export function navSports(exported: string[] | undefined): SportDef[] {
  const has = new Set(exported ?? []);
  return SPORTS.filter((s) => s.generator || has.has(s.id));
}
