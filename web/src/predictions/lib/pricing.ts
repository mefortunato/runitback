// Prices any line in the browser: the model's predictions don't depend on
// the odds, so a different spread, total or price only changes the
// comparison. Probabilities are the model's, moved toward the market's by
// how much the model's disagreements have paid in the backtest (`trust`).
import { payout } from './markets';
import type { Pick, Prediction, Pricing } from './types';

/** Standard normal CDF (Abramowitz & Stegun 7.1.26, error < 1.5e-7). */
export function normalCdf(x: number): number {
	const t = 1 / (1 + 0.3275911 * (Math.abs(x) / Math.SQRT2));
	const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
	const erf = 1 - poly * Math.exp(-(x * x) / 2);
	return x >= 0 ? (1 + erf) / 2 : (1 - erf) / 2;
}

/** P(home wins) implied by both moneylines, the bookmaker's margin removed. */
export function noVig(home: number, away: number): number {
	const implied = (odds: number) => 1 / (1 + payout(odds));
	const h = implied(home);
	const a = implied(away);
	return h / (h + a);
}

/** Fraction of Kelly staked: a quarter, since the edge estimate is noisy. */
export const KELLY_FRACTION = 0.25;

export interface Quote {
	pick: Pick;
	odds: number;
	/** The model's own probability of the pick winning (a push left out). */
	raw: number;
	/** After moving toward the market by the backtest's trust. */
	prob: number;
	/** Win rate the odds need to break even. */
	breakEven: number;
	/** Expected profit per unit staked. */
	ev: number;
	/** Suggested stake, a fraction of bankroll (fractional Kelly; 0 below break-even). */
	stake: number;
}

function quote(pick: Pick, odds: number, raw: number, prob: number): Quote {
	const b = payout(odds);
	const ev = prob * b - (1 - prob);
	const kelly = Math.max(0, prob - (1 - prob) / b);
	return { pick, odds, raw, prob, breakEven: 1 / (1 + b), ev, stake: KELLY_FRACTION * kelly };
}

const toward = (market: number, model: number, trust = 0) => market + trust * (model - market);

/** Both sides of a spread. `spread` is the line priced, as the home team's
 * expected margin (positive: home favored), i.e. minus the home handicap books
 * quote. `market` is the consensus margin the model is moved toward (default:
 * `spread` itself, a coin flip); a line off the consensus is then priced from
 * the market's own view, so moving the line moves the win chance. */
export function priceSpread(
	prediction: Prediction,
	spread: number,
	odds: { home: number; away: number },
	pricing: Pricing,
	market: number = spread
): [Quote, Quote] {
	const raw = normalCdf((prediction.margin - spread) / pricing.sigma);
	const anchor = normalCdf((market - spread) / pricing.sigma);
	const prob = toward(anchor, raw, pricing.trust.spread);
	return [quote('away', odds.away, 1 - raw, 1 - prob), quote('home', odds.home, raw, prob)];
}

/** Both sides of the moneyline at `odds`. `market` is the market's no-vig
 * P(home wins) the model is moved toward (default: from `odds` themselves). */
export function priceMoneyline(
	prediction: Prediction,
	odds: { home: number; away: number },
	pricing: Pricing,
	market: number = noVig(odds.home, odds.away)
): [Quote, Quote] {
	const raw = prediction.win_prob;
	const prob = toward(market, raw, pricing.trust.moneyline);
	return [quote('away', odds.away, 1 - raw, 1 - prob), quote('home', odds.home, raw, prob)];
}

/** Over and under of a total line, for a model that predicts the total.
 * `market` is the consensus total the model is moved toward (default: `line`). */
export function priceTotal(
	prediction: Prediction,
	line: number,
	odds: { over: number; under: number },
	pricing: Pricing,
	market: number = line
): [Quote, Quote] | null {
	if (prediction.total == null || !pricing.total_sigma) return null;
	const raw = normalCdf((prediction.total - line) / pricing.total_sigma);
	const anchor = normalCdf((market - line) / pricing.total_sigma);
	const prob = toward(anchor, raw, pricing.trust.total);
	return [quote('over', odds.over, raw, prob), quote('under', odds.under, 1 - raw, 1 - prob)];
}

/** American odds are valid at or beyond ±100. */
export function validOdds(odds: number): boolean {
	return Number.isFinite(odds) && Math.abs(odds) >= 100;
}
