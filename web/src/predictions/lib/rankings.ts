import type { PlayoffStage, RankingSeason, Team } from './types';

/** Highlighted teams: team key -> series slot (1-3). The slot stays with the
 * team while it is highlighted, so its color never changes under it. */
export type Pins = Record<string, number>;

export const MAX_PINS = 3;

export interface RankRow {
	team: Team;
	rating: number;
	rank: number;
	/** Places gained since the previous date (null on the first date). */
	change: number | null;
}

function ranked(season: RankingSeason, index: number): RankRow[] {
	return season.teams
		.filter((t) => t.ratings[index] !== null && t.ratings[index] !== undefined)
		.map((t) => ({ team: t.team, rating: t.ratings[index] as number, rank: 0, change: null }))
		.sort((a, b) => b.rating - a.rating)
		.map((row, i) => ({ ...row, rank: i + 1 }));
}

/** Teams ordered by rating at `index`, with the change in rank since index - 1. */
export function rankAt(season: RankingSeason, index: number): RankRow[] {
	const rows = ranked(season, index);
	if (index === 0) return rows;
	const before = new Map(ranked(season, index - 1).map((r) => [r.team.team, r.rank]));
	return rows.map((row) => {
		const previous = before.get(row.team.team);
		return { ...row, change: previous === undefined ? null : previous - row.rank };
	});
}

/** The top teams at the season's latest date. */
export function defaultPins(season: RankingSeason, count = MAX_PINS): Pins {
	const top = rankAt(season, season.dates.length - 1).slice(0, count);
	return Object.fromEntries(top.map((row, i) => [row.team.team, i + 1]));
}

/** Highlight or un-highlight `team`. A new team takes the lowest free slot;
 * with every slot taken it replaces the team in the highest slot. */
export function togglePin(pins: Pins, team: string): Pins {
	if (team in pins) {
		const { [team]: _, ...rest } = pins;
		return rest;
	}
	const used = new Set(Object.values(pins));
	for (let slot = 1; slot <= MAX_PINS; slot++) {
		if (!used.has(slot)) return { ...pins, [team]: slot };
	}
	const [replaced] = Object.entries(pins).find(([, slot]) => slot === MAX_PINS) ?? [];
	const { [replaced as string]: _, ...rest } = pins;
	return { ...rest, [team]: MAX_PINS };
}

/** Symmetric y extent around zero: whole-number ticks, at most 5 per side. */
export function yExtent(season: RankingSeason): { max: number; step: number } {
	let largest = 0;
	for (const t of season.teams)
		for (const v of t.ratings) if (v !== null) largest = Math.max(largest, Math.abs(v));
	const step = [1, 2, 4, 5, 10, 20].find((s) => Math.ceil(largest / s) <= 5) ?? 50;
	return { max: Math.max(step, Math.ceil(largest / step) * step), step };
}

/** A conference, optionally narrowed to one of its divisions, or the
 * teams that reached the last playoff rounds. */
export interface TeamFilter {
	conference?: string;
	division?: string;
	playoffs?: boolean;
}

/** Conferences and their divisions, from the teams' own metadata (empty
 * when the sport's teams have none). */
export function filterOptions(teams: Team[]): {
	conferences: string[];
	divisions: Record<string, string[]>;
} {
	const divisions: Record<string, Set<string>> = {};
	for (const team of teams) {
		if (!team.conference) continue;
		divisions[team.conference] ??= new Set();
		if (team.division) divisions[team.conference].add(team.division);
	}
	const conferences = Object.keys(divisions).sort();
	return {
		conferences,
		divisions: Object.fromEntries(conferences.map((c) => [c, [...divisions[c]].sort()]))
	};
}

export function matchesFilter(
	team: Team,
	filter: TeamFilter,
	playoffs: Record<string, PlayoffStage> = {}
): boolean {
	if (filter.playoffs && !(team.team in playoffs)) return false;
	if (filter.conference && team.conference !== filter.conference) return false;
	if (filter.division && team.division !== filter.division) return false;
	return true;
}

const STAGE_ORDER: PlayoffStage[] = ['champion', 'runner_up', 'semifinal'];

/** The final teams with their league rank at `index`: champion, runner-up,
 * then the rest by rank. */
export function playoffRanks(
	season: RankingSeason,
	index: number
): { row: RankRow; stage: PlayoffStage }[] {
	const playoffs = season.playoffs ?? {};
	return rankAt(season, index)
		.filter((row) => row.team.team in playoffs)
		.map((row) => ({ row, stage: playoffs[row.team.team] }))
		.sort((a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage) || a.row.rank - b.row.rank);
}

/** Highlights after the visible teams change: teams still visible keep their
 * slot (and color), hidden ones are dropped, and free slots go to the top
 * visible teams. `rows` are the visible teams in rank order. */
export function reconcilePins(pins: Pins, rows: RankRow[], count = MAX_PINS): Pins {
	const visible = new Set(rows.map((r) => r.team.team));
	const kept: Pins = Object.fromEntries(Object.entries(pins).filter(([team]) => visible.has(team)));
	const used = new Set(Object.values(kept));
	const free = Array.from({ length: count }, (_, i) => i + 1).filter((slot) => !used.has(slot));
	for (const row of rows) {
		if (!free.length) break;
		if (row.team.team in kept) continue;
		kept[row.team.team] = free.shift() as number;
	}
	return kept;
}

/** A fill on the diverging scale: gray at zero, toward red below and blue
 * above, fully saturated at ±`max`. */
export function divergeColor(value: number, max: number): string {
	const t = max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
	const pole = value < 0 ? 'var(--diverge-neg)' : 'var(--diverge-pos)';
	return `color-mix(in oklab, ${pole} ${Math.round(t * 100)}%, var(--diverge-mid))`;
}

/** The color scale's half-width: the 90th percentile of |rating| (a few
 * outliers, like a college league's weakest teams, would otherwise wash
 * everyone else out to gray), rounded up to a whole point. */
export function divergeMax(values: number[]): number {
	const sorted = values.map(Math.abs).sort((a, b) => a - b);
	const at = sorted[Math.min(sorted.length - 1, Math.floor(0.9 * sorted.length))] ?? 0;
	return Math.max(1, Math.ceil(at));
}
