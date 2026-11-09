import { Link } from "../../router";
import { kickoffLabel, spreadLabel } from "../lib/format";
import { links } from "../lib/links";
import { age, withLine } from "../lib/odds";
import type { GameOdds, UpcomingGame } from "../lib/types";
import MarketTable from "./MarketTable";
import SpreadChart from "./SpreadChart";
import TeamBadge from "./TeamBadge";
import WinProbBar from "./WinProbBar";
import s from "./GameCard.module.css";

export default function GameCard({ game: built, sport, odds = null, hasSpread = true }: {
  game: UpcomingGame; sport: string; odds?: GameOdds | null; hasSpread?: boolean;
}) {
  // current sportsbook lines, when the site has them, replace the ones it was built with; the
  // picks follow the lines
  const live = withLine(built, built.prediction, odds?.consensus);
  const game = live.game;
  const prediction = live.prediction ?? built.prediction;
  const openSpread = odds?.open.spread ?? null;
  return (
    <Link className={`card ${s.game}`} to={links.game(sport, game.game_id)}>
      <div className={s.top}>
        <div className={s.teams}>
          <TeamBadge team={game.away} label="name" />
          <span className={`muted ${s.at}`}>{game.neutral ? "vs" : "@"}</span>
          <TeamBadge team={game.home} label="name" />
        </div>
        <span className="muted small">{kickoffLabel(game.kickoff)}</span>
      </div>
      <SpreadChart home={game.home} away={game.away} spread={game.spread} model={prediction.margin} />
      <WinProbBar home={game.home} away={game.away} probability={prediction.win_prob} />
      <MarketTable game={game} prediction={prediction} hasSpread={hasSpread} />
      {odds && (
        <div className={`${s.lines} muted small`}>
          Lines from {odds.books.length} book{odds.books.length === 1 ? "" : "s"}, {age(odds.updated)}
          {openSpread !== null && game.spread !== null && openSpread !== game.spread &&
            <> · opened {spreadLabel(game.home, game.away, openSpread)}</>}
        </div>
      )}
    </Link>
  );
}
