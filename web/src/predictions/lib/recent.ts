// The recent-games page: how the model's pre-game predictions did on the
// last played dates, game by game and in total, next to the market's.
import { SPREAD_ODDS, units } from './markets';
import type { Market, Pick, PlayedGame, Recent, RecentFile, Side } from './types';

/** One market's grade for a game: the model's pick and how it went. `hit`
 * is null when it did not count (no pick, or a push / tie). */
export interface Grade {
	pick: Pick | null;
	outcome: Pick | 'push' | null;
	hit: boolean | null;
	/** Profit on a 1-unit bet at the game's odds (0 when it did not count). */
	units: number;
}

/** What `pick` won against `outcome`: null when either is missing or it pushed. */
function hit(pick: Pick | null | undefined, outcome: Pick | 'push' | null | undefined) {
	if (!pick || !outcome || outcome === 'push') return null;
	return pick === outcome;
}

/** The side the model's win probability favors (null at a coin flip). */
export function modelWinner(game: PlayedGame): Side | null {
	const p = game.prediction.win_prob;
	return p > 0.5 ? 'home' : p < 0.5 ? 'away' : null;
}

/** The side the market favors: by the moneyline where there is one, else
 * by the spread (null without a line, or at a pick'em). */
export function marketWinner(game: PlayedGame): Side | null {
	const prob = game.prediction.market_prob;
	if (prob != null) return prob > 0.5 ? 'home' : prob < 0.5 ? 'away' : null;
	if (game.spread == null || game.spread === 0) return null;
	return game.spread > 0 ? 'home' : 'away';
}

/** How the model did on `market` for a played game. */
export function grade(game: PlayedGame, market: Market | 'winner'): Grade {
	const p = game.prediction;
	const r = game.result;
	if (market === 'winner') {
		const pick = modelWinner(game);
		const h = hit(pick, r.winner);
		return { pick, outcome: r.winner ?? null, hit: h, units: 0 };
	}
	if (market === 'spread') {
		const pick = p.ats_pick ?? null;
		const h = game.spread == null ? null : hit(pick, r.covered);
		return { pick, outcome: r.covered, hit: h, units: h === null ? 0 : units(SPREAD_ODDS, h) };
	}
	if (market === 'moneyline') {
		const pick = p.ml_pick ?? null;
		const h = hit(pick, r.winner);
		const odds = pick ? game.moneyline?.[pick] : undefined;
		return {
			pick,
			outcome: r.winner ?? null,
			hit: odds === undefined ? null : h,
			units: h === null || odds === undefined ? 0 : units(odds, h)
		};
	}
	const pick = p.ou_pick ?? null;
	const h = game.total_line == null ? null : hit(pick, r.went);
	const odds = pick === 'over' || pick === 'under' ? (game.total_odds?.[pick] ?? SPREAD_ODDS) : SPREAD_ODDS;
	return { pick, outcome: r.went ?? null, hit: h, units: h === null ? 0 : units(odds, h) };
}

export interface MarketRecord {
	wins: number;
	losses: number;
	pushes: number;
	units: number;
}

/** The model's record and Vegas's on a set of played games. */
export interface RecentSummary {
	games: number;
	/** Picking winners: the model's favorite vs the market's. */
	winners: { model: MarketRecord; market: MarketRecord | null };
	spread: MarketRecord | null;
	moneyline: MarketRecord | null;
	total: MarketRecord | null;
	/** Mean absolute miss of the margin: the model's prediction vs the spread
	 * (games with a spread). */
	margin: { n: number; model: number; market: number } | null;
	/** Mean absolute miss of combined points: model total vs the line. */
	points: { n: number; model: number; market: number } | null;
	/** Brier score of the home win probability: model vs the moneyline's. */
	brier: { n: number; model: number; market: number } | null;
}

function tally(grades: Grade[], counted: boolean[]): MarketRecord | null {
	const rows = grades.filter((_, i) => counted[i]);
	if (!rows.length) return null;
	return {
		wins: rows.filter((g) => g.hit === true).length,
		losses: rows.filter((g) => g.hit === false).length,
		pushes: rows.filter((g) => g.hit === null && g.pick !== null).length,
		units: rows.reduce((sum, g) => sum + g.units, 0)
	};
}

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

export function summarize(games: PlayedGame[]): RecentSummary {
	const winners = games.map((g) => grade(g, 'winner'));
	const marketWins = games.map((g): Grade => {
		const pick = marketWinner(g);
		return { pick, outcome: g.result.winner ?? null, hit: hit(pick, g.result.winner), units: 0 };
	});
	const lined = games.map((g) => g.spread != null);
	const withMl = games.map((g) => g.moneyline != null);
	const withTotal = games.map((g) => g.total_line != null && g.prediction.total != null);

	const spreadGames = games.filter((g) => g.spread != null);
	const totalGames = games.filter(
		(g) => g.total_line != null && g.prediction.total != null && g.result.total != null
	);
	const decided = games.filter(
		(g) => g.prediction.market_prob != null && (g.result.winner === 'home' || g.result.winner === 'away')
	);
	const won = (g: PlayedGame) => (g.result.winner === 'home' ? 1 : 0);

	return {
		games: games.length,
		winners: {
			model: tally(winners, games.map(() => true)) ?? { wins: 0, losses: 0, pushes: 0, units: 0 },
			market: tally(marketWins, lined.map((l, i) => l || withMl[i]))
		},
		spread: tally(games.map((g) => grade(g, 'spread')), lined),
		moneyline: tally(games.map((g) => grade(g, 'moneyline')), withMl),
		total: tally(games.map((g) => grade(g, 'total')), withTotal),
		margin: spreadGames.length
			? {
					n: spreadGames.length,
					model: mean(spreadGames.map((g) => Math.abs(g.result.margin - g.prediction.margin))),
					market: mean(spreadGames.map((g) => Math.abs(g.result.margin - (g.spread as number))))
				}
			: null,
		points: totalGames.length
			? {
					n: totalGames.length,
					model: mean(
						totalGames.map((g) => Math.abs((g.result.total as number) - (g.prediction.total as number)))
					),
					market: mean(
						totalGames.map((g) => Math.abs((g.result.total as number) - (g.total_line as number)))
					)
				}
			: null,
		brier: decided.length
			? {
					n: decided.length,
					model: mean(decided.map((g) => (g.prediction.win_prob - won(g)) ** 2)),
					market: mean(decided.map((g) => ((g.prediction.market_prob as number) - won(g)) ** 2))
				}
			: null
	};
}

/** How a recent game's cards can be ordered. */
export type RecentSort = 'kickoff' | 'miss' | 'edge';

export const RECENT_SORT_LABELS: Record<RecentSort, string> = {
	kickoff: 'Kickoff',
	miss: 'Biggest model miss',
	edge: 'Biggest spread edge'
};

/** `games` in `sort` order: kickoff (as given), the model's largest margin
 * miss first, or its largest disagreement with the spread first. */
export function sortPlayed(games: PlayedGame[], sort: RecentSort): PlayedGame[] {
	if (sort === 'kickoff') return games;
	const size = (g: PlayedGame) =>
		sort === 'miss'
			? Math.abs(g.result.margin - g.prediction.margin)
			: g.prediction.edge == null
				? -1
				: Math.abs(g.prediction.edge);
	return [...games].sort((a, b) => size(b) - size(a));
}

/** The recent games of one date (the newest without `season` / `date`). */
export function recentDay(
	file: RecentFile | null,
	season: number | null = null,
	date: number | null = null
): Recent {
	const dates = file?.dates ?? [];
	const all = dates.flatMap((d) => file?.by_date[`${d.season}-${d.date}`] ?? []);
	const chosen =
		season == null || date == null
			? dates[0]
			: dates.find((d) => d.season === season && d.date === date);
	if (season != null && date != null && !chosen) throw new Error(`No games for ${season} ${date}`);
	if (!chosen) return { season: null, date: null, name: null, dates, games: [], all };
	return {
		season: chosen.season,
		date: chosen.date,
		name: chosen.name ?? null,
		dates,
		games: file?.by_date[`${chosen.season}-${chosen.date}`] ?? [],
		all
	};
}
