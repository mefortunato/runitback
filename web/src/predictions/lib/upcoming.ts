import type { Upcoming, UpcomingGame } from './types';

/** How the upcoming games are ordered: by kickoff, or biggest edge first. */
export type GameSort = 'kickoff' | 'spread' | 'moneyline' | 'total';

export const SORT_LABELS: Record<GameSort, string> = {
	kickoff: 'Kickoff',
	spread: 'Spread edge',
	moneyline: 'Moneyline edge',
	total: 'Over/under edge'
};

/** The model's disagreement with the market on `sort`'s market, or null
 * where the game has no line (or the model no prediction) for it. */
export function edgeOf(game: UpcomingGame, sort: GameSort): number | null {
	const p = game.prediction;
	const value =
		sort === 'spread'
			? p.edge
			: sort === 'moneyline'
				? p.ml_edge
				: sort === 'total'
					? p.total_edge
					: null;
	return value === null || value === undefined || !Number.isFinite(value) ? null : value;
}

/** The orders worth offering: kickoff, and each market some game has an edge on. */
export function sortOptions(games: UpcomingGame[]): GameSort[] {
	const markets: GameSort[] = ['spread', 'moneyline', 'total'];
	return ['kickoff', ...markets.filter((m) => games.some((g) => edgeOf(g, m) !== null))];
}

/** `games` biggest absolute edge first on `sort`'s market; games without
 * that line follow in kickoff order. Kickoff keeps the given order. */
export function sortGames(games: UpcomingGame[], sort: GameSort): UpcomingGame[] {
	if (sort === 'kickoff') return games;
	const size = (g: UpcomingGame) => {
		const edge = edgeOf(g, sort);
		return edge === null ? -1 : Math.abs(edge);
	};
	// Array.prototype.sort is stable: equal edges keep kickoff order
	return [...games].sort((a, b) => size(b) - size(a));
}

/** Today in US Eastern (kickoffs are listed in it), `YYYY-MM-DD`. */
export function todayEastern(now: Date = new Date()): string {
	// en-CA formats as YYYY-MM-DD
	return new Intl.DateTimeFormat('en-CA', {
		timeZone: 'America/New_York',
		year: 'numeric',
		month: '2-digit',
		day: '2-digit'
	}).format(now);
}

/** Whether `game`'s kickoff day is before `today` (both US Eastern).
 * Games without a kickoff time are kept: there is no telling. */
export function isPast(game: UpcomingGame, today: string): boolean {
	return game.kickoff != null && game.kickoff.slice(0, 10) < today;
}

/** The exported upcoming file (`{default, by_date}`), written at build time. */
export interface UpcomingFile {
	default: Upcoming;
	by_date: Record<string, Upcoming>;
}

/** The upcoming games as of `today`, not as of the last build: games whose
 * kickoff day has passed are dropped, dates left with no games leave the
 * date list, and with no `date` asked for the first date still to come is
 * shown. The site is static and refreshed by a daily build, so between
 * builds yesterday's games would otherwise still be listed as upcoming. */
export function currentUpcoming(
	file: UpcomingFile,
	season: number | null = null,
	date: number | null = null,
	today: string = todayEastern()
): Upcoming {
	const base = file.default;
	const weekOf = (d: number): Upcoming | undefined =>
		base.season === null ? undefined : file.by_date[`${base.season}-${d}`];
	const remaining = (week: Upcoming): UpcomingGame[] => week.games.filter((g) => !isPast(g, today));
	// a date the export has no file for is kept, as before
	const dates = base.dates.filter((d) => {
		const week = weekOf(d);
		return !week || remaining(week).length > 0;
	});

	let week: Upcoming;
	if (season == null || date == null) {
		const next = dates.find((d) => weekOf(d));
		week = (next === undefined ? undefined : weekOf(next)) ?? base;
	} else {
		const found = file.by_date[`${season}-${date}`];
		if (!found) throw new Error(`No games for ${season} ${date}`);
		week = found;
	}
	return {
		...week,
		dates: week.season === base.season ? dates : week.dates,
		games: remaining(week)
	};
}
