// Betting-market helpers shared by the game views, performance page and the
// challenge (which scores picks here, in the browser).
import { num } from './format';
import type { Market, Pick, Side, Team } from './types';

const MINUS = '−';

export const MARKET_LABELS: Record<Market, string> = {
	spread: 'Spread',
	moneyline: 'Moneyline',
	total: 'Over/under'
};

/** Odds assumed for spread bets. */
export const SPREAD_ODDS = -110;

/** Win rate needed to break even at `odds`. */
export function breakEven(odds = SPREAD_ODDS): number {
	return 1 / (1 + payout(odds));
}

/** Profit per unit staked on a winning bet at American `odds`. */
export function payout(odds: number): number {
	return odds > 0 ? odds / 100 : 100 / -odds;
}

/** Units won (or −1 lost) on one unit at `odds`. */
export function units(odds: number, won: boolean): number {
	return won ? payout(odds) : -1;
}

/** "+130" / "−150". */
export function americanOdds(odds: number | null | undefined): string {
	if (odds === null || odds === undefined) return '—';
	const rounded = Math.round(odds);
	return rounded > 0 ? `+${rounded}` : `${MINUS}${Math.abs(rounded)}`;
}

/** "+1.9 u" / "−3.0 u". */
export function unitsLabel(value: number, digits = 1): string {
	const text = Math.abs(value).toFixed(digits);
	return `${value > 0 ? '+' : value < 0 ? MINUS : ''}${text} u`;
}

/** "45.5" or "No line". */
export function totalLabel(line: number | null | undefined): string {
	return line === null || line === undefined ? 'No line' : num(line);
}

/** "KC" / "BUF" / "Over" / "Under" for a pick. */
export function pickLabel(pick: Pick | null | undefined, home: Team, away: Team): string {
	if (!pick) return '—';
	if (pick === 'over') return 'Over';
	if (pick === 'under') return 'Under';
	return (pick === 'home' ? home : away).team;
}

export function isSide(pick: Pick | null | undefined): pick is Side {
	return pick === 'home' || pick === 'away';
}
