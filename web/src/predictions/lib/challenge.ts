import { useCallback, useEffect, useState } from 'react';
import type { ChallengeAnswer, Market } from './types';

interface Tally {
	wins: number;
	losses: number;
	/** Profit in units, one unit per game at the market's odds. */
	units: number;
}

/** The player's and the model's record on one market, on the same games. */
export interface ChallengeRecord {
	played: number;
	player: Tally;
	model: Tally;
	/** Games where the player picked the same side as the model. */
	agreed: number;
	streak: number;
	bestStreak: number;
	/** Recently played game ids, so they are not served again soon. */
	seen: string[];
}

const SEEN_LIMIT = 300;

export function emptyRecord(): ChallengeRecord {
	return {
		played: 0,
		player: { wins: 0, losses: 0, units: 0 },
		model: { wins: 0, losses: 0, units: 0 },
		agreed: 0,
		streak: 0,
		bestStreak: 0,
		seen: []
	};
}

/** The record after one answered game (pure). */
export function applyAnswer(record: ChallengeRecord, answer: ChallengeAnswer): ChallengeRecord {
	const streak = answer.correct ? record.streak + 1 : 0;
	return {
		played: record.played + 1,
		player: {
			wins: record.player.wins + (answer.correct ? 1 : 0),
			losses: record.player.losses + (answer.correct ? 0 : 1),
			units: record.player.units + answer.units
		},
		model: {
			wins: record.model.wins + (answer.model.correct ? 1 : 0),
			losses: record.model.losses + (answer.model.correct ? 0 : 1),
			units: record.model.units + answer.model.units
		},
		agreed: record.agreed + (answer.pick === answer.model.pick ? 1 : 0),
		streak,
		bestStreak: Math.max(record.bestStreak, streak),
		seen: [...record.seen.filter((id) => id !== answer.game_id), answer.game_id].slice(
			-SEEN_LIMIT
		)
	};
}

type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem' | 'removeItem'>;

// the spread keeps the key it had before there were other markets
function key(sport: string, market: Market) {
	return market === 'spread' ? `sportsml:challenge:${sport}` : `sportsml:challenge:${sport}:${market}`;
}

function browserStorage(): Storage | null {
	try {
		return globalThis.localStorage ?? null;
	} catch {
		return null;
	}
}

export function loadRecord(
	sport: string,
	market: Market = 'spread',
	storage: Storage | null = browserStorage()
): ChallengeRecord {
	try {
		const saved = storage?.getItem(key(sport, market));
		if (!saved) return emptyRecord();
		const parsed = JSON.parse(saved);
		const empty = emptyRecord();
		// records saved before units existed lack them
		return {
			...empty,
			...parsed,
			player: { ...empty.player, ...parsed.player },
			model: { ...empty.model, ...parsed.model }
		};
	} catch {
		return emptyRecord();
	}
}

export function saveRecord(
	sport: string,
	market: Market,
	record: ChallengeRecord,
	storage: Storage | null = browserStorage()
) {
	try {
		storage?.setItem(key(sport, market), JSON.stringify(record));
	} catch {
		// storage full or blocked: the record still lives for this visit
	}
}

/** The sport's record on `market`, persisted to localStorage on every change. */
export function useChallengeRecord(sport: string, market: Market = 'spread') {
	const [record, setRecord] = useState<ChallengeRecord>(() => loadRecord(sport, market));
	useEffect(() => setRecord(loadRecord(sport, market)), [sport, market]);
	const update = useCallback(
		(next: ChallengeRecord) => {
			saveRecord(sport, market, next);
			setRecord(next);
		},
		[sport, market]
	);
	return {
		record,
		add: (answer: ChallengeAnswer) => update(applyAnswer(record, answer)),
		reset: () => update(emptyRecord())
	};
}
