import { useState } from "react";
import { Link } from "../../router";
import { api } from "../lib/api";
import { dateText, seasonText } from "../lib/format";
import { links } from "../lib/links";
import { gameOdds } from "../lib/odds";
import type { SportInfo } from "../lib/types";
import { SORT_LABELS, sortGames, sortOptions, type GameSort } from "../lib/upcoming";
import GameCard from "../components/GameCard";
import { remember, remembered, useData } from "../useData";
import { cx } from "../cx";
import s from "./Upcoming.module.css";

const SORT_KEY = "runitback:upcoming-sort";

export default function Upcoming({ info, season, date }: { info: SportInfo; season?: number | null; date?: number | null }) {
  const { data, error } = useData(
    async () => ({ upcoming: await api.upcoming(info.sport, season, date), odds: await api.odds(info.sport) }),
    [info.sport, season, date],
  );
  const [sort, setSort] = useState<GameSort>(() => remembered(SORT_KEY, "kickoff" as GameSort));
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  const { upcoming, odds } = data;
  const options = sortOptions(upcoming.games);
  // a remembered market this week's games have no lines for falls back to kickoff
  const active = options.includes(sort) ? sort : "kickoff";
  const games = sortGames(upcoming.games, active);
  const choose = (next: GameSort) => { setSort(next); remember(SORT_KEY, next); };
  return (
    <div>
      <div className={s.heading}>
        <div>
          <h1>
            {info.label} · {dateText(info.date_label, upcoming.date, upcoming.date_names?.[String(upcoming.date)])}{" "}
            {upcoming.season ? <span className="muted">{seasonText(upcoming.season, info.split_season)}</span> : null}
          </h1>
          <p className={`muted small ${s.legend}`}>
            {info.has_spread ? (
              <>
                <span className={cx(s.key, s.vegas)} /> Vegas line
                <span className={cx(s.key, s.model)} /> model's predicted margin
                <span className={cx(s.key, s.edge)} /> where they disagree · right of center favors the home team
              </>
            ) : (
              <><span className={cx(s.key, s.model)} /> model's predicted margin · right of center favors the home team</>
            )}
          </p>
        </div>
        {upcoming.dates.length > 1 && (
          <nav className={s.dates}>
            {upcoming.dates.map((d) => (
              <Link key={d} className={cx("pill", d === upcoming.date && "accent")} to={links.upcoming(info.sport, upcoming.season, d)}>
                {dateText(info.date_label, d, upcoming.date_names?.[String(d)])}
              </Link>
            ))}
          </nav>
        )}
      </div>
      {upcoming.games.length === 0 ? (
        <div className="card">
          <h2>No upcoming games</h2>
          <p className="muted">
            The schedule has no unplayed games right now. Try <Link to={links.challenge(info.sport)}>beating the model on past games</Link>,
            or <Link to={links.generator(info.sport)}>generate a game play by play</Link>.
          </p>
        </div>
      ) : (
        <>
          {options.length > 1 && (
            <div className={s.sort} role="group" aria-label="Order games">
              <span className="muted small">Sort by</span>
              {options.map((o) => (
                <button key={o} className={cx("pill", active === o && "accent")} aria-pressed={active === o} onClick={() => choose(o)}>
                  {SORT_LABELS[o]}
                </button>
              ))}
              {active !== "kickoff" && <span className="muted small">biggest disagreement with the market first</span>}
            </div>
          )}
          <div className={s.games}>
            {games.map((g) => (
              <GameCard key={g.game_id} game={g} sport={info.sport} odds={gameOdds(odds, g.game_id)} hasSpread={info.has_spread} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
