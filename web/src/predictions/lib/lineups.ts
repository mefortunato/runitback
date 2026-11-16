// The lineup editor's arithmetic. The model reads player values only through additive terms, so an
// edited lineup's prediction is the published one plus a weighted sum of the change in each
// position group's value: computed here exactly as the model would (src/runitback/wager/web/lineups.py).
import { normalCdf } from './pricing';
import type { GameLineups, LineupPlayer, Prediction, Pricing, Side } from './types';

/** A team's lineup: per group, the players in depth order (the first `slots` start) and those out. */
export interface TeamLineup {
	[group: string]: { active: string[]; out: string[] };
}
export type LineupState = Record<Side, TeamLineup>;

export function initialLineup(lineups: GameLineups, side: Side): TeamLineup {
	const out: TeamLineup = {};
	for (const g of lineups.groups) {
		const players = lineups[side][g.key] ?? [];
		out[g.key] = { active: players.filter((p) => !p.out).map((p) => p.id), out: players.filter((p) => p.out).map((p) => p.id) };
	}
	return out;
}

export function initialState(lineups: GameLineups): LineupState {
	return { home: initialLineup(lineups, 'home'), away: initialLineup(lineups, 'away') };
}

export function playerIndex(lineups: GameLineups, side: Side, group: string): Map<string, LineupPlayer> {
	return new Map((lineups[side][group] ?? []).map((p) => [p.id, p]));
}

/** Points per game over replacement of a player at slot `rank` of `group` (0 past the starters). */
export function slotPoints(lineups: GameLineups, group: string, rank: number, player: LineupPlayer | undefined): number {
	const mult = lineups.slots[group]?.[rank] ?? 0;
	return player ? player.value * mult : 0;
}

/** Points per game over replacement of a player at slot `rank` of `group`, over every column he adds to
 * (a rotation player's offense and defense together). */
export function playerPoints(lineups: GameLineups, group: string, rank: number, player: LineupPlayer | undefined): number {
	if (!player?.values) return slotPoints(lineups, group, rank, player);
	const mult = lineups.slots[group]?.[rank] ?? 0;
	return Object.values(player.values).reduce((a, v) => a + v, 0) * mult;
}

/** Each value column's points for one side's lineup. */
export function columnPoints(lineups: GameLineups, side: Side, team: TeamLineup): number[] {
	const sums = lineups.columns.map(() => 0);
	for (const g of lineups.groups) {
		const idx = lineups.columns.indexOf(g.column);
		const players = playerIndex(lineups, side, g.key);
		(team[g.key]?.active ?? []).forEach((id, rank) => {
			const p = players.get(id);
			if (p?.values) {
				// a rotation: the slot's usage times his offense and defense, in his position's columns
				const mult = lineups.slots[g.key]?.[rank] ?? 0;
				for (const [col, v] of Object.entries(p.values)) {
					const i = lineups.columns.indexOf(col);
					if (i >= 0) sums[i] += v * mult;
				}
			} else if (idx >= 0) {
				sums[idx] += slotPoints(lineups, g.key, rank, p);
			}
		});
	}
	return sums;
}

/** How an edited lineup moves the predicted home margin and the total, in points. */
export function lineupEffect(lineups: GameLineups, state: LineupState): { margin: number; total: number } {
	const base = initialState(lineups);
	let margin = 0;
	let total = 0;
	const h0 = columnPoints(lineups, 'home', base.home);
	const a0 = columnPoints(lineups, 'away', base.away);
	const h1 = columnPoints(lineups, 'home', state.home);
	const a1 = columnPoints(lineups, 'away', state.away);
	lineups.columns.forEach((_, i) => {
		const dh = h1[i] - h0[i];
		const da = a1[i] - a0[i];
		margin += lineups.weights.margin[i] * (dh - da);
		total += lineups.weights.total[i] * (dh + da);
	});
	return { margin, total };
}

/** The prediction under an edited lineup: margin and total moved, probabilities from the model's
 * residual spreads (as the model computes them), edges left to `withLine`. */
export function adjustPrediction(
	prediction: Prediction,
	effect: { margin: number; total: number },
	pricing: Pricing | null | undefined,
	totalLine: number | null | undefined
): Prediction {
	if (!pricing || (Math.abs(effect.margin) < 1e-9 && Math.abs(effect.total) < 1e-9)) return prediction;
	const margin = prediction.margin + effect.margin;
	const total = prediction.total == null ? prediction.total : prediction.total + effect.total;
	const over_prob =
		total == null || totalLine == null || !pricing.total_sigma
			? prediction.over_prob
			: normalCdf((total - totalLine) / pricing.total_sigma);
	return { ...prediction, margin, win_prob: normalCdf(margin / pricing.sigma), total, over_prob };
}

/** Put `id` at starting slot `rank`: a player already active swaps places with the one there, one
 * ruled out comes back in (the rest move down a slot). */
export function setStarter(team: TeamLineup, group: string, rank: number, id: string): TeamLineup {
	const { active, out } = team[group];
	const next = [...active];
	const at = next.indexOf(id);
	if (at >= 0) {
		const r = Math.min(rank, next.length - 1);
		[next[at], next[r]] = [next[r], next[at]];
		return { ...team, [group]: { active: next, out } };
	}
	next.splice(Math.min(rank, next.length), 0, id);
	return { ...team, [group]: { active: next, out: out.filter((x) => x !== id) } };
}

/** Rule a player out: everyone behind him moves up a slot. */
export function ruleOut(team: TeamLineup, group: string, id: string): TeamLineup {
	const { active, out } = team[group];
	return { ...team, [group]: { active: active.filter((x) => x !== id), out: [id, ...out.filter((x) => x !== id)] } };
}

export function sameLineup(a: TeamLineup, b: TeamLineup): boolean {
	return Object.keys(a).every(
		(g) => a[g].active.join() === b[g]?.active.join() && a[g].out.join() === b[g]?.out.join()
	);
}
