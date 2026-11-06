import type { Matchup } from "../lib/types";
import RecentGames from "./RecentGames";
import StatComparison from "./StatComparison";

/** Pre-game team stats, records and recent form (shared by game and challenge). */
export default function MatchupDetails({ matchup, dateLabel = "Week", splitSeason = false }: {
  matchup: Matchup; dateLabel?: string; splitSeason?: boolean;
}) {
  return (
    <>
      <div className="section card">
        <h2>Team stats</h2>
        <StatComparison stats={matchup.stats} away={matchup.summaries.away} home={matchup.summaries.home}
          dateLabel={dateLabel} splitSeason={splitSeason} />
      </div>
      <div className="section">
        <h2>Recent form</h2>
        <div className="grid-2">
          <div className="card"><RecentGames summary={matchup.summaries.away} dateLabel={dateLabel} /></div>
          <div className="card"><RecentGames summary={matchup.summaries.home} dateLabel={dateLabel} /></div>
        </div>
      </div>
    </>
  );
}
