// Applies a game's current sportsbook lines to the site's data: the model's
// predictions stay as they are, only the lines and the picks against them
// change (the same rules as the snapshot's `_prediction`).
import { noVig } from './pricing';
import type { BookLine, GameHeader, GameOdds, OddsFile, Prediction } from './types';

/** The game's header and prediction at `line` (missing markets keep the
 * line the site was built with). */
export function withLine<G extends GameHeader>(
	game: G,
	prediction: Prediction | null,
	line: Partial<BookLine> | null | undefined
): { game: G; prediction: Prediction | null } {
	if (!line) return { game, prediction };
	const spread = line.spread ?? game.spread;
	const moneyline =
		line.ml_home != null && line.ml_away != null
			? { home: line.ml_home, away: line.ml_away }
			: (game.moneyline ?? null);
	const totalLine = line.total ?? game.total_line ?? null;
	const totalOdds =
		line.over_odds != null && line.under_odds != null
			? { over: line.over_odds, under: line.under_odds }
			: (game.total_odds ?? null);
	const next = { ...game, spread, moneyline, total_line: totalLine, total_odds: totalOdds };
	if (!prediction) return { game: next, prediction };

	const edge = spread == null ? null : prediction.margin - spread;
	const ats_pick = edge === null || edge === 0 ? null : edge > 0 ? 'home' : 'away';
	const market_prob = moneyline ? noVig(moneyline.home, moneyline.away) : null;
	const ml_edge = market_prob === null ? null : prediction.win_prob - market_prob;
	const ml_pick = ml_edge === null || ml_edge === 0 ? null : ml_edge > 0 ? 'home' : 'away';
	const total_edge =
		prediction.total == null || totalLine == null ? null : prediction.total - totalLine;
	const ou_pick =
		total_edge === null || total_edge === 0 ? null : total_edge > 0 ? 'over' : 'under';
	return {
		game: next,
		prediction: { ...prediction, edge, ats_pick, market_prob, ml_edge, ml_pick, total_edge, ou_pick }
	};
}

export function gameOdds(file: OddsFile | null | undefined, gameId: string): GameOdds | null {
	return file?.games?.[gameId] ?? null;
}

/** "−3 → −4.5": how a home handicap moved from the opening line. */
export function moved(open: number | null | undefined, now: number | null | undefined): number | null {
	if (open == null || now == null) return null;
	const delta = now - open;
	return Math.abs(delta) < 1e-9 ? 0 : delta;
}

/** "3 hours ago" style age of an ISO time, relative to `now`. */
export function age(iso: string | null | undefined, now: Date = new Date()): string {
	if (!iso) return 'unknown';
	const minutes = Math.round((now.getTime() - new Date(iso).getTime()) / 60000);
	if (minutes < 1) return 'just now';
	if (minutes < 60) return `${minutes} min ago`;
	const hours = Math.round(minutes / 60);
	if (hours < 48) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
	return `${Math.round(hours / 24)} days ago`;
}
