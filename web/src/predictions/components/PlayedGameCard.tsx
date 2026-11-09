import { Link } from "../../router";
import { kickoffLabel } from "../lib/format";
import { links } from "../lib/links";
import { grade } from "../lib/recent";
import type { PlayedGame } from "../lib/types";
import MarketTable from "./MarketTable";
import SpreadChart from "./SpreadChart";
import TeamBadge from "./TeamBadge";
import WinProbBar from "./WinProbBar";
import { cx } from "../cx";
import s from "./PlayedGameCard.module.css";

// A played game: the final score, where the result landed next to the line and the model's
// pre-game margin, and how each of its picks did.
export default function PlayedGameCard({ game, sport, hasSpread = true }: { game: PlayedGame; sport: string; hasSpread?: boolean }) {
  const result = game.result;
  const verdicts = [
    { label: "Winner", grade: grade(game, "winner") },
    ...(hasSpread ? [{ label: "Spread", grade: grade(game, "spread") }] : []),
    { label: "Moneyline", grade: grade(game, "moneyline") },
    { label: "Over/under", grade: grade(game, "total") },
  ].filter((v) => v.grade.hit !== null);
  return (
    <Link className={cx("card", s.game)} to={links.game(sport, game.game_id)}>
      <div className={s.top}>
        <div className={s.score}>
          <div className={cx(s.line, result.winner === "away" && s.won)}>
            <TeamBadge team={game.away} label="name" />
            <span className={cx(s.points, "tabular")}>{result.away_score ?? "—"}</span>
          </div>
          <div className={cx(s.line, result.winner === "home" && s.won)}>
            <TeamBadge team={game.home} label="name" />
            <span className={cx(s.points, "tabular")}>{result.home_score ?? "—"}</span>
          </div>
        </div>
        <div className={s.meta}>
          <span className="muted small">Final{game.neutral ? " · neutral" : ""}</span>
          {game.kickoff && <span className="muted small">{kickoffLabel(game.kickoff)}</span>}
        </div>
      </div>
      <SpreadChart home={game.home} away={game.away} spread={game.spread} model={game.prediction.margin} result={result.margin} />
      <WinProbBar home={game.home} away={game.away} probability={game.prediction.win_prob} />
      <MarketTable game={game} prediction={game.prediction} result={result} hasSpread={hasSpread} />
      {verdicts.length > 0 && (
        <div className={s.verdicts} aria-label="How the model's picks did">
          {verdicts.map((v) => (
            <span key={v.label} className={cx("pill", v.grade.hit ? "good" : "bad")}>{v.grade.hit ? "✓" : "✗"} {v.label}</span>
          ))}
          {game.prediction_source === "backtest" && <span className="muted small">walk-forward backtest prediction</span>}
        </div>
      )}
    </Link>
  );
}
