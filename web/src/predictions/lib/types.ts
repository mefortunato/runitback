// Shapes of the data `sportsml web export` writes (src/sportsml/web/snapshot.py).

export type Side = 'home' | 'away';
export type TotalSide = 'over' | 'under';
/** A side of any market: a team, or the over / under. */
export type Pick = Side | TotalSide;
/** Markets the model is scored on and the challenge can be played on. */
export type Market = 'spread' | 'moneyline' | 'total';

export interface Team {
	team: string;
	name: string;
	/** Short label for chips (college teams are keyed by school name). */
	abbr?: string;
	nickname?: string;
	color?: string;
	color2?: string;
	logo?: string;
	conference?: string;
	division?: string;
}

export interface Metrics {
	n: number;
	rmse: number;
	mae: number;
	accuracy: number;
	brier?: number;
	log_loss?: number;
}

export interface SportInfo {
	sport: string;
	label: string;
	date_label: string;
	/** Seasons span two years and are keyed by the year they end (NBA). */
	split_season?: boolean;
	has_spread: boolean;
	/** Whether the model has team embeddings to map (absent: older exports). */
	has_team_map?: boolean;
	challenge_games: number;
	/** Markets the challenge offers (older exports: spread only). */
	markets?: Market[];
	/** What pricing a line in the browser needs (older exports: absent). */
	pricing?: Pricing | null;
	built_at?: string | null;
	refreshed_at?: string | null;
	model?: { family: string; metrics: Record<string, Metrics> } | null;
	backtest?: {
		seasons: number[];
		overall: { model: Metrics; market?: Metrics };
		ats?: { win_pct?: number; bets?: number };
		moneyline?: Partial<BetRow>;
		totals?: { n?: number; record?: Partial<BetRow> };
	} | null;
}

/** The model's residual spreads and how much of its disagreement with the
 * market to believe, per market (0: none, 1: all), from the backtest. */
export interface Pricing {
	sigma: number;
	total_sigma?: number | null;
	trust: Partial<Record<Market, number>>;
}

export interface Prediction {
	/** Home team's predicted margin. */
	margin: number;
	/** P(home team wins). */
	win_prob: number;
	/** Model margin minus the spread (home perspective). */
	edge: number | null;
	ats_pick: Side | null;
	/** P(home team wins) implied by the moneylines, bookmaker margin removed. */
	market_prob?: number | null;
	/** Model minus market win probability (home perspective). */
	ml_edge?: number | null;
	/** The side the model rates above the market's odds. */
	ml_pick?: Side | null;
	/** Predicted combined points (models that predict the total). */
	total?: number | null;
	/** P(combined points > the total line). */
	over_prob?: number | null;
	/** Model total minus the line. */
	total_edge?: number | null;
	ou_pick?: TotalSide | null;
}

export interface GameHeader {
	game_id: string;
	season: number;
	date: number;
	/** The date's display name, where the sport has one (NBA: "Oct 28"). */
	date_name?: string | null;
	kickoff: string | null;
	home: Team;
	away: Team;
	neutral: boolean;
	/** Market's expected home margin: positive means the home team is favored. */
	spread: number | null;
	/** American moneylines. */
	moneyline?: { home: number; away: number } | null;
	/** The over/under line on combined points. */
	total_line?: number | null;
	/** American odds on the over and the under. */
	total_odds?: { over: number; under: number } | null;
}

export interface UpcomingGame extends GameHeader {
	prediction: Prediction;
}

export interface Upcoming {
	season: number | null;
	date: number | null;
	dates: number[];
	/** Display name per date, where the sport has them. */
	date_names?: Record<string, string | null> | null;
	games: UpcomingGame[];
}

export interface WinLoss {
	wins: number;
	losses: number;
	ties: number;
}

export interface AtsRecord {
	wins: number;
	losses: number;
	pushes: number;
}

export interface RecentGame {
	game_id: string;
	season: number;
	date: number;
	date_name?: string | null;
	opp: Team;
	location: number | null;
	score: number | null;
	opp_score: number | null;
	margin: number;
	spread: number | null;
	ats: 'W' | 'L' | 'P' | null;
}

export interface TeamSummary {
	team: Team;
	/** Season the averages and records cover (last season before an opener). */
	season: number;
	current_season: boolean;
	games: number;
	record: WinLoss;
	ats: AtsRecord;
	averages: Record<string, number | null>;
	recent: RecentGame[];
}

export interface StatDef {
	key: string;
	label: string;
	decimals: number;
	higher_is_better: boolean;
}

export interface GameResult {
	home_score: number | null;
	away_score: number | null;
	margin: number;
	covered: Side | 'push' | null;
	winner?: Side | 'push' | null;
	/** Combined points. */
	total?: number | null;
	went?: TotalSide | 'push' | null;
}

export interface Matchup extends GameHeader {
	summaries: { home: TeamSummary; away: TeamSummary };
	stats: StatDef[];
}

export interface GameDetail extends Matchup {
	prediction: Prediction | null;
	result: GameResult | null;
}

export interface ChallengeGame extends Matchup {
	pool_size: number;
	/** Markets this game can be played on (it did not land on their line). */
	markets?: Market[];
}

/** A market's outcome for one game, before the player's pick is scored. */
export interface MarketAnswer {
	game_id: string;
	market: Market;
	/** The winning pick. */
	outcome: Pick;
	/** American odds of each pick. */
	odds: Partial<Record<Pick, number>>;
	result: GameResult;
	spread?: number;
	covered?: Side;
	total_line?: number;
	model: Prediction & { pick: Pick | null; correct: boolean; units: number };
}

export interface ChallengeAnswer extends MarketAnswer {
	pick: Pick;
	correct: boolean;
	/** Profit on a 1-unit bet at the odds (−1 when it loses). */
	units: number;
}

/** Model and (when the sport has lines) market metrics on the same games. */
export interface SourceMetrics {
	model: Metrics;
	market?: Metrics;
}

/** Betting the model's side when it disagrees with the market by > min_edge
 * (points for the spread and total, win probability for the moneyline). */
export interface BetRow {
	min_edge: number;
	bets: number;
	wins: number;
	losses: number;
	pushes: number;
	win_pct: number | null;
	ci95_low: number | null;
	ci95_high: number | null;
	/** Profit in units (1 per bet) at the odds. */
	units?: number;
	roi: number;
}

export type AtsRow = BetRow;

/** One staking policy scored on the same bets (relative units: the average
 * stake on its training bets is 1, so ROI is comparable with flat). */
export interface StakingPolicy {
	policy: string;
	bets: number;
	staked: number;
	profit: number;
	roi: number;
	roi_ci95_low: number;
	roi_ci95_high: number;
	roi_vs_flat: number;
	roi_vs_flat_ci95_low: number;
	roi_vs_flat_ci95_high: number;
	max_drawdown: number;
}

/** Stakes sized from the model's edge, learned walk-forward, for one market. */
export interface StakingSummary {
	seasons: number[];
	n: number;
	policies: StakingPolicy[];
	/** Units staked at a ladder of edges (points; win probability for the moneyline). */
	curve: { edge: number; units: number }[];
	max_units: number;
	fraction: number;
}

export interface TotalsSummary {
	n: number;
	rmse: { model: number; market: number };
	/** Over/under probability Brier score (games that did not land on the line). */
	brier: { model?: number; market?: number };
	edge_correlation: { r: number; ci95_low: number; ci95_high: number };
	record: BetRow[];
}

export interface CalibrationBin {
	predicted: number;
	actual: number;
	n: number;
}

/** Walk-forward backtest results (the same predictions as the challenge). */
export interface Performance {
	seasons: number[];
	overall: SourceMetrics;
	by_season: Record<string, SourceMetrics>;
	by_phase: Record<string, SourceMetrics>;
	ats?: AtsRow[];
	ats_by_season?: Record<string, AtsRow>;
	edge_correlation?: { r: number; ci95_low: number; ci95_high: number };
	calibration: { model: CalibrationBin[]; market?: CalibrationBin[] };
	moneyline?: BetRow[];
	moneyline_by_season?: Record<string, BetRow>;
	totals?: TotalsSummary;
	totals_by_season?: Record<string, BetRow>;
	over_calibration?: { model: CalibrationBin[]; market?: CalibrationBin[] };
	staking?: Partial<Record<Market, StakingSummary>>;
}

export interface RankingSeries {
	team: Team;
	/** Average predicted margin vs every other team, before each date. */
	ratings: (number | null)[];
}

/** How a team's season ended, for the last playoff rounds. */
export type PlayoffStage = 'champion' | 'runner_up' | 'semifinal';

export interface RankingSeason {
	season: number;
	/** backtest: that season's walk-forward model; current: the served model. */
	source: 'backtest' | 'current';
	dates: number[];
	/** Display name per date (aligned with `dates`), where the sport has them. */
	date_names?: (string | null)[] | null;
	teams: RankingSeries[];
	/** Teams that reached the last playoff rounds (empty until they are played). */
	playoffs?: Record<string, PlayoffStage>;
}

/** Something the team map can be colored by. */
export interface TeamMapMetric {
	key: string;
	label: string;
	decimals: number;
	higher_is_better: boolean;
	/** Where the color scale is neutral: 0, 0.5, or the league average at that date. */
	center: 'zero' | 'half' | 'mean';
}

/** One team's path through the team map, one entry per date. */
export interface TeamMapSeries {
	team: Team;
	/** UMAP coordinates in [-1, 1] (null: no state that date). Only
	 * distances mean anything; the axes have no units. */
	positions: ([number, number] | null)[];
	/** Each metric's value before each date (null: no games yet). */
	values: Record<string, (number | null)[]>;
	/** [wins, losses, ties] before each date. */
	records?: [number, number, number][] | null;
}

/** The latest season's teams in 2-D before each ranking date: every
 * (team, date) laid out together, so a team's path is its movement. */
export interface TeamMap {
	season: number;
	dates: number[];
	date_names?: (string | null)[] | null;
	metrics: TeamMapMetric[];
	teams: TeamMapSeries[];
}

export interface Rankings {
	seasons: RankingSeason[];
	date_label: string;
	/** Display label per playoff stage, e.g. "Won Super Bowl". */
	playoff_labels?: Partial<Record<PlayoffStage, string>>;
}

/** One sportsbook's line, from the home team's side (`spread` is the home
 * team's expected margin, the opposite sign of the handicap books quote). */
export interface BookLine {
	spread: number | null;
	spread_home_odds: number | null;
	spread_away_odds: number | null;
	ml_home: number | null;
	ml_away: number | null;
	total: number | null;
	over_odds: number | null;
	under_odds: number | null;
}

export interface BookOffer {
	book: string;
	odds: number | null;
	/** Spread (home expected margin) or total of the offer. */
	points?: number;
}

export interface GameOdds {
	updated: string;
	books: (BookLine & { book: string })[];
	/** Median over books. */
	consensus: BookLine;
	best: {
		spread: { home: BookOffer | null; away: BookOffer | null };
		moneyline: { home: BookOffer | null; away: BookOffer | null };
		total: { over: BookOffer | null; under: BookOffer | null };
	};
	open: { spread: number | null; total: number | null };
	history: { t: string; spread: number | null; total: number | null; ml_home?: number | null; ml_away?: number | null }[];
}

/** `<sport>/odds.json`: current lines for upcoming games (absent without it). */
export interface OddsFile {
	updated: string | null;
	provider: string | null;
	remaining: number | null;
	games: Record<string, GameOdds>;
}

/** Where a recent game's prediction came from: the walk-forward backtest, or
 * the served model replayed with team state from earlier games only. */
export type PredictionSource = 'backtest' | 'replay';

/** A played game on the recent-games page: its lines, the model's pre-game
 * prediction and the result. */
export interface PlayedGame extends GameHeader {
	prediction: Prediction;
	result: GameResult;
	prediction_source: PredictionSource;
}

export interface RecentDate {
	season: number;
	date: number;
	/** The date's display name, where the sport has them. */
	name?: string | null;
}

/** `<sport>/recent.json`: the last played dates (newest first) and their games. */
export interface RecentFile {
	dates: RecentDate[];
	by_date: Record<string, PlayedGame[]>;
}

/** One recent date's games, and every recent game for the window's totals. */
export interface Recent {
	season: number | null;
	date: number | null;
	name: string | null;
	dates: RecentDate[];
	games: PlayedGame[];
	all: PlayedGame[];
}
