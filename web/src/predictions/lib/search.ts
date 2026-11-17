// The header search: every sport's pages and teams (ported from sportsml's web app).
import type { Team } from "./types";

/** Lowercase, accents dropped, punctuation as spaces ("São Paulo" -> "sao paulo"). */
export function normalize(text: string): string {
  return text
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function names(team: Team): string[] {
  return [team.team, team.abbr, team.name, team.nickname].filter((n): n is string => !!n).map(normalize);
}

/** How well `query` names `team`, higher is better; null when it doesn't.
 * An exact key, abbreviation, name or nickname beats a name that starts with the query, which beats
 * every query word starting a word of the names, which beats the query appearing anywhere in them. */
export function teamScore(team: Team, query: string): number | null {
  const q = normalize(query);
  if (!q) return 0;
  const all = names(team);
  if (all.includes(q)) return 4;
  if (all.some((n) => n.startsWith(q))) return 3;
  const words = all.flatMap((n) => n.split(" "));
  if (q.split(" ").every((w) => words.some((word) => word.startsWith(w)))) return 2;
  if (all.some((n) => n.includes(q))) return 1;
  return null;
}

/** A page the header search can go to. */
export interface PageResult {
  sport: string;
  sportLabel: string;
  label: string;
  href: string;
}

/** A team the header search can show, with the sport it plays in. */
export interface TeamResult {
  sport: string;
  sportLabel: string;
  team: Team;
}

export interface SearchResults {
  pages: PageResult[];
  teams: TeamResult[];
}

const words = (text: string) => normalize(text).split(" ").filter(Boolean);

/** Search every sport's pages and teams. A word naming a sport ("nba") narrows both to it; the
 * other words find pages by their name ("rank") and teams as `teamScore` does. The current sport's
 * results come first among equals. */
export function globalSearch(
  query: string,
  pages: PageResult[],
  teams: TeamResult[],
  current?: string | null,
  limits = { pages: 6, teams: 8 },
): SearchResults {
  const all = words(query);
  if (!all.length) return { pages: [], teams: [] };
  const sportNames = new Map<string, string>();
  for (const p of [...pages, ...teams]) {
    sportNames.set(normalize(p.sport), p.sport);
    sportNames.set(normalize(p.sportLabel), p.sport);
  }
  const sports = new Set(all.flatMap((w) => sportNames.get(w) ?? []));
  const rest = all.filter((w) => !sportNames.has(w));
  const inSport = (sport: string) => !sports.size || sports.has(sport);
  const currentFirst = (a: { sport: string }, b: { sport: string }) =>
    Number(b.sport === current) - Number(a.sport === current);

  const foundPages = pages
    .filter((p) => inSport(p.sport) && (rest.length || sports.size)
      && rest.every((w) => words(p.label).some((word) => word.startsWith(w))))
    .map((p, i) => ({ p, i }))
    .sort((a, b) => currentFirst(a.p, b.p) || a.i - b.i)
    .slice(0, limits.pages)
    .map(({ p }) => p);

  const teamQuery = rest.join(" ");
  const foundTeams = teamQuery
    ? teams
      .filter((t) => inSport(t.sport))
      .map((t, i) => ({ t, i, score: teamScore(t.team, teamQuery) }))
      .filter((m): m is { t: TeamResult; i: number; score: number } => m.score !== null)
      .sort((a, b) => b.score - a.score || currentFirst(a.t, b.t) || a.i - b.i)
      .slice(0, limits.teams)
      .map(({ t }) => t)
    : [];
  return { pages: foundPages, teams: foundTeams };
}
