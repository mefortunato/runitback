import { useEffect, useState } from "react";
import { Link } from "../../router";
import { api } from "../lib/api";
import { dateText, kickoffLabel, marginLabel, seasonText, signed, spreadLabel } from "../lib/format";
import { links } from "../lib/links";
import { gameOdds, withLine } from "../lib/odds";
import type { SportInfo } from "../lib/types";
import LineCalculator from "../components/LineCalculator";
import LineupEditor from "../components/LineupEditor";
import { adjustPrediction, initialState, lineupEffect, type LineupState } from "../lib/lineups";
import MarketTable from "../components/MarketTable";
import MatchupDetails from "../components/MatchupDetails";
import ProbabilityCompare from "../components/ProbabilityCompare";
import Sportsbooks from "../components/Sportsbooks";
import SpreadChart from "../components/SpreadChart";
import TeamBadge from "../components/TeamBadge";
import TotalChart from "../components/TotalChart";
import WinProbBar from "../components/WinProbBar";
import { useData } from "../useData";
import { cx } from "../cx";
import s from "./Game.module.css";

export default function Game({ info, gameId }: { info: SportInfo; gameId: string }) {
  const { data, error } = useData(
    async () => {
      const game = await api.game(info.sport, gameId);
      const lineups = game.result || !game.prediction ? null : (game.lineups ?? (await api.lineups(info.sport, gameId)));
      return { game, lineups, odds: await api.odds(info.sport) };
    },
    [info.sport, gameId],
  );
  const lineups = data?.lineups ?? null;
  const [edits, setEdits] = useState<LineupState | null>(null);
  useEffect(() => setEdits(lineups ? initialState(lineups) : null), [lineups]);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  const built = data.game;
  const result = built.result;
  // an upcoming game's current sportsbook lines replace the built ones
  const odds = result ? null : gameOdds(data.odds, built.game_id);
  // edited lineups re-price the game (the model's lineup terms are additive: lib/lineups.ts)
  const effect = lineups && edits ? lineupEffect(lineups, edits) : { margin: 0, total: 0 };
  const editedLineup = Math.abs(effect.margin) > 1e-9 || Math.abs(effect.total) > 1e-9;
  const basePrediction = built.prediction;
  const ownPrediction = basePrediction && editedLineup
    ? adjustPrediction(basePrediction, effect, info.pricing, odds?.consensus?.total ?? built.total_line)
    : basePrediction;
  const live = withLine(built, ownPrediction, odds?.consensus ?? (editedLineup ? {} : null));
  const game = live.game;
  const prediction = live.prediction;
  const pickTeam = prediction?.ats_pick ? (prediction.ats_pick === "home" ? game.home : game.away) : null;
  const coveredTeam = result?.covered === "home" ? game.home : result?.covered === "away" ? game.away : null;
  const generate = `${links.generator(info.sport)}?home=${encodeURIComponent(game.home.team)}&away=${encodeURIComponent(game.away.team)}`;
  return (
    <div>
      {result
        ? <Link className="small" to={links.recent(info.sport, game.season, game.date)}>← {info.label} recent games</Link>
        : <Link className="small" to={links.upcoming(info.sport)}>← {info.label} upcoming</Link>}
      <div className={cx("card section", s.hero)}>
        <div className={cx(s.meta, "muted small")}>
          {dateText(info.date_label, game.date, game.date_name)}, {seasonText(game.season, info.split_season)} · {kickoffLabel(game.kickoff)}{game.neutral ? " · neutral site" : ""}
        </div>
        <div className={s.matchup}>
          <div className={s.team}>
            <TeamBadge team={game.away} size={56} label="none" />
            <div className={s.name}>{game.away.name}</div>
            {result && <div className={cx(s.score, "tabular")}>{result.away_score}</div>}
          </div>
          <div className={cx(s.at, "muted")}>{game.neutral ? "vs" : "@"}</div>
          <div className={s.team}>
            <TeamBadge team={game.home} size={56} label="none" />
            <div className={s.name}>{game.home.name}</div>
            {result && <div className={cx(s.score, "tabular")}>{result.home_score}</div>}
          </div>
        </div>
        <div className={s.chips}>
          {info.has_spread && <span className="pill">Vegas {spreadLabel(game.home, game.away, game.spread)}</span>}
          {prediction ? (
            <>
              <span className="pill accent">Model {marginLabel(game.home, game.away, prediction.margin)}{editedLineup ? " (your lineups)" : ""}</span>
              {pickTeam && prediction.edge !== null &&
                <span className="pill accent">Model pick: {pickTeam.team} covers (edge {signed(Math.abs(prediction.edge))})</span>}
            </>
          ) : <span className="pill">No model prediction for this game</span>}
          {result && (
            <span className={cx("pill", coveredTeam !== null && "good")}>
              {result.covered === "push" ? "Push" : coveredTeam ? `${coveredTeam.team} covered` : "Final"}
            </span>
          )}
          <Link className="pill" to={generate}>▶ Generate this matchup play by play</Link>
        </div>
        <div className={s.chart}>
          <SpreadChart home={game.home} away={game.away} spread={game.spread} model={prediction?.margin ?? null} result={result?.margin ?? null} />
        </div>
        {prediction && (prediction.market_prob != null
          ? <ProbabilityCompare home={game.home} away={game.away} model={prediction.win_prob} market={prediction.market_prob} />
          : <WinProbBar home={game.home} away={game.away} probability={prediction.win_prob} />)}
        {(game.total_line != null || prediction?.total != null) && (
          <div className={s.chart}>
            <TotalChart line={game.total_line} model={prediction?.total ?? null} result={result?.total ?? null} />
          </div>
        )}
        <MarketTable game={game} prediction={prediction} result={result} hasSpread={info.has_spread} />
      </div>
      {lineups && edits && basePrediction && prediction && (
        <div className="section">
          <LineupEditor game={game} lineups={lineups} state={edits} onChange={setEdits}
            base={withLine(built, basePrediction, odds?.consensus).prediction ?? basePrediction} edited={prediction} />
        </div>
      )}
      {odds && <div className="section"><Sportsbooks home={game.home} away={game.away} odds={odds} /></div>}
      {prediction && !result && info.pricing && (
        <div className="section">
          <LineCalculator game={game} prediction={prediction} pricing={info.pricing} books={odds?.books ?? []} hasSpread={info.has_spread} />
        </div>
      )}
      <MatchupDetails matchup={game} dateLabel={info.date_label} splitSeason={info.split_season} />
    </div>
  );
}
