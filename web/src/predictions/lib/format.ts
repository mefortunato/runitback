import type { AtsRecord, Side, Team, WinLoss } from './types';

const MINUS = '−';

/** 3 -> "3", 3.5 -> "3.5", 2.25 -> "2.3" (with digits = 1). */
export function num(value: number, digits = 1): string {
	if (Number.isInteger(value)) return String(value);
	return value.toFixed(digits);
}

/** +3.5 / −2.0 with a real minus sign. */
export function signed(value: number, digits = 1): string {
	const text = Math.abs(value).toFixed(digits);
	if (value > 0) return `+${text}`;
	if (value < 0) return `${MINUS}${text}`;
	return text;
}

/** A date as the sport names it: its own name where the export gives one
 * (NBA game days: "Oct 28"), else "Week 9". */
export function dateText(label: string, date: number | null, name?: string | null): string {
	if (name) return name;
	return date === null ? '—' : `${label} ${date}`;
}

/** A season as the sport names it: 2026, or 2025–26 for a season that spans
 * two years (keyed by the year it ends). */
export function seasonText(season: number, split = false): string {
	return split ? `${season - 1}–${String(season).slice(2)}` : String(season);
}

/** A run of seasons: "2021–2025", "2020–21 to 2024–25", or just one. */
export function seasonsText(seasons: number[], split = false): string {
	if (!seasons.length) return '';
	const [first, last] = [seasons[0], seasons[seasons.length - 1]];
	if (first === last) return seasonText(first, split);
	return split ? `${seasonText(first, true)} to ${seasonText(last, true)}` : `${first}–${last}`;
}

/** Sportsbook-style line: "KC −3.5", "Pick'em" or "No line".
 * `spread` is the market's expected home margin (positive = home favored). */
export function spreadLabel(home: Team, away: Team, spread: number | null | undefined): string {
	if (spread === null || spread === undefined) return 'No line';
	if (spread === 0) return "Pick'em";
	const favorite = spread > 0 ? home : away;
	return `${favorite.team} ${MINUS}${num(Math.abs(spread))}`;
}

/** A side's handicap: the home team gets −spread, the away team +spread. */
export function handicap(spread: number, side: Side): string {
	const line = side === 'home' ? -spread : spread;
	if (line === 0) return 'PK';
	return line > 0 ? `+${num(line)}` : `${MINUS}${num(Math.abs(line))}`;
}

/** "KC by 5.2" for a home-perspective margin. */
export function marginLabel(home: Team, away: Team, margin: number): string {
	if (Math.abs(margin) < 0.05) return 'Even';
	const team = margin > 0 ? home : away;
	return `${team.team} by ${Math.abs(margin).toFixed(1)}`;
}

export function pct(probability: number, digits = 0): string {
	return `${(probability * 100).toFixed(digits)}%`;
}

export function winLoss(record: WinLoss): string {
	const base = `${record.wins}-${record.losses}`;
	return record.ties ? `${base}-${record.ties}` : base;
}

export function atsRecord(record: AtsRecord): string {
	const base = `${record.wins}-${record.losses}`;
	return record.pushes ? `${base}-${record.pushes}` : base;
}

/** Share of decided games won, or null with none decided. */
export function winRate(wins: number, losses: number): number | null {
	return wins + losses ? wins / (wins + losses) : null;
}

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** "2026-11-01 13:00" (US Eastern) -> "Sun, Nov 1 · 1:00 PM ET". */
export function kickoffLabel(kickoff: string | null): string {
	if (!kickoff) return 'TBD';
	const [day, time] = kickoff.split(' ');
	const [year, month, date] = day.split('-').map(Number);
	if (!year || !month || !date) return kickoff;
	const weekday = DAYS[new Date(Date.UTC(year, month - 1, date)).getUTCDay()];
	let text = `${weekday}, ${MONTHS[month - 1]} ${date}`;
	if (time) {
		const [hour, minute] = time.split(':').map(Number);
		const suffix = hour >= 12 ? 'PM' : 'AM';
		const twelve = hour % 12 === 0 ? 12 : hour % 12;
		text += ` · ${twelve}:${String(minute).padStart(2, '0')} ${suffix} ET`;
	}
	return text;
}
