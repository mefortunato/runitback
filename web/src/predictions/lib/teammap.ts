import { signed } from './format';
import type { TeamMapMetric, TeamMapSeries } from './types';

export type Vec2 = [number, number];

/** Linear interpolation between dates `floor(t)` and the next one; where
 * either is missing, the value at the nearest date that has one. */
function between<T>(values: (T | null)[], t: number, mix: (a: T, b: T, f: number) => T): T | null {
	const n = values.length;
	if (!n) return null;
	const i = Math.max(0, Math.min(n - 1, Math.floor(t)));
	const j = Math.min(n - 1, i + 1);
	const f = Math.max(0, Math.min(1, t - i));
	const [a, b] = [values[i], values[j]];
	if (a !== null && b !== null) return mix(a, b, f);
	const at = Math.max(0, Math.min(n - 1, t));
	let best: T | null = null;
	let distance = Infinity;
	values.forEach((v, k) => {
		if (v !== null && Math.abs(k - at) < distance) [best, distance] = [v, Math.abs(k - at)];
	});
	return best;
}

/** A team's position at a fractional date index: between two dates it
 * moves in a straight line; around a date without state it stays at the
 * nearest one that has it. */
export function positionAt(series: TeamMapSeries, t: number): Vec2 | null {
	return between(series.positions, t, (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
}

/** A metric's value at a fractional date index, interpolated the same way
 * (but with no value before a team's first game). */
export function valueAt(series: TeamMapSeries, key: string, t: number): number | null {
	const values = series.values[key] ?? [];
	const n = values.length;
	if (!n) return null;
	const i = Math.max(0, Math.min(n - 1, Math.floor(t)));
	const j = Math.min(n - 1, i + 1);
	const f = Math.max(0, Math.min(1, t - i));
	const [a, b] = [values[i], values[j]];
	if (a !== null && b !== null) return a + (b - a) * f;
	return f < 0.5 ? a : b;
}

/** The box around every team's every position. */
export function extent(teams: TeamMapSeries[]): { x0: number; x1: number; y0: number; y1: number } {
	let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
	for (const t of teams)
		for (const p of t.positions)
			if (p) [x0, x1, y0, y1] = [Math.min(x0, p[0]), Math.max(x1, p[0]), Math.min(y0, p[1]), Math.max(y1, p[1])];
	return Number.isFinite(x0) ? { x0, x1, y0, y1 } : { x0: -1, x1: 1, y0: -1, y1: 1 };
}

/** Teams ranked by a metric at a date index (1 = best), skipping nulls. */
export function ranksAt(teams: TeamMapSeries[], index: number, metric?: TeamMapMetric): Map<string, number> {
	const key = metric?.key ?? 'rating';
	const sign = metric && !metric.higher_is_better ? -1 : 1;
	const at = (t: TeamMapSeries) => t.values[key]?.[index] ?? null;
	return new Map(
		teams
			.filter((t) => at(t) !== null)
			.sort((a, b) => sign * ((at(b) as number) - (at(a) as number)))
			.map((t, i) => [t.team.team, i + 1])
	);
}

/** The neutral value of a metric's color scale at a date index. */
export function centerAt(metric: TeamMapMetric, teams: TeamMapSeries[], index: number): number {
	if (metric.center === 'zero') return 0;
	if (metric.center === 'half') return 0.5;
	const values = teams.map((t) => t.values[metric.key]?.[index]).filter((v) => v != null) as number[];
	return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

/** The scale's half-width: the 90th percentile of every team's distance
 * from the center over the whole season, so it holds still during
 * playback and a few outliers saturate instead of washing out the rest. */
export function spread(metric: TeamMapMetric, teams: TeamMapSeries[]): number {
	const n = teams[0]?.values[metric.key]?.length ?? 0;
	const gaps: number[] = [];
	for (let i = 0; i < n; i++) {
		const center = centerAt(metric, teams, i);
		for (const t of teams) {
			const v = t.values[metric.key]?.[i];
			if (v != null) gaps.push(Math.abs(v - center));
		}
	}
	gaps.sort((a, b) => a - b);
	const at = gaps[Math.min(gaps.length - 1, Math.floor(0.9 * gaps.length))] ?? 0;
	return at > 0 ? at : 1;
}

/** How good a value is on the color scale: -1 (worst) .. 1 (best), 0 at the center. */
export function goodness(metric: TeamMapMetric, value: number, center: number, half: number): number {
	const t = Math.max(-1, Math.min(1, (value - center) / half));
	return metric.higher_is_better ? t : -t;
}

/** A metric's value as the page shows it: margins signed, win percentage
 * like .625, stats with their own decimals. */
export function formatMetric(metric: TeamMapMetric, value: number | null): string {
	if (value === null) return '–';
	if (metric.key === 'win_pct') return value >= 1 ? '1.000' : value.toFixed(3).replace(/^0/, '');
	if (metric.center === 'zero') return signed(value, metric.decimals);
	return value.toFixed(metric.decimals);
}

/** A change in a metric, always signed: +1.5, −.125. */
export function formatChange(metric: TeamMapMetric, change: number | null): string {
	if (change === null) return '–';
	const sign = change > 0 ? '+' : change < 0 ? '−' : '';
	const size = Math.abs(change);
	const text = metric.key === 'win_pct' ? size.toFixed(3).replace(/^0/, '') : size.toFixed(metric.decimals);
	return sign + text;
}

/** "5–3" or "5–3–1" (with ties). */
export function recordText(record: [number, number, number] | null | undefined): string | null {
	if (!record || record[0] + record[1] + record[2] === 0) return null;
	return record[2] ? record.join('–') : `${record[0]}–${record[1]}`;
}
