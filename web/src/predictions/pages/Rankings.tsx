import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { dateText, seasonText, signed } from "../lib/format";
import { filterOptions, matchesFilter, MAX_PINS, playoffRanks, rankAt, reconcilePins, togglePin, type Pins, type TeamFilter } from "../lib/rankings";
import type { PlayoffStage, Rankings as RankingsData, SportInfo } from "../lib/types";
import RankingsChart from "../components/RankingsChart";
import TeamBadge from "../components/TeamBadge";
import { useData } from "../useData";
import { cx } from "../cx";
import s from "./Rankings.module.css";

const SHORT: Record<PlayoffStage, string> = { champion: "Champion", runner_up: "Runner-up", semifinal: "Final four" };

export default function Rankings({ info }: { info: SportInfo }) {
  const { data, error } = useData(() => api.rankings(info.sport), [info.sport]);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  return <RankingsView info={info} rankings={data} />;
}

function RankingsView({ info, rankings }: { info: SportInfo; rankings: RankingsData }) {
  const [chosen, setChosen] = useState<number | null>(null);
  const [pins, setPins] = useState<Pins>({});
  const [index, setIndex] = useState<number | null>(null);
  const [filter, setFilter] = useState<TeamFilter>({});
  const season = rankings.seasons.find((x) => x.season === chosen) ?? rankings.seasons[0];
  const options = filterOptions(season ? season.teams.map((t) => t.team) : []);
  // how the season ended for the last teams standing (these teams should rank near the top)
  const playoffs = season?.playoffs ?? {};
  const hasPlayoffs = Object.keys(playoffs).length > 0;
  const stageLabel = (stage: PlayoffStage) => rankings.playoff_labels?.[stage] ?? SHORT[stage];
  // a season whose playoffs are not played yet has no final four to show
  const active: TeamFilter = filter.playoffs && !hasPlayoffs ? {} : filter;
  const filtering = Boolean(active.conference || active.playoffs);
  const visible = season && filtering
    ? new Set(season.teams.filter((t) => matchesFilter(t.team, active, playoffs)).map((t) => t.team.team)) : null;
  const filterLabel = active.playoffs ? "final four" : (active.division ?? active.conference ?? null);
  const isVisible = (team: string) => !visible || visible.has(team);

  // Every season or filter change starts from that view's top teams; teams highlighted by hand
  // last until the next change.
  const viewKey = `${season?.season}|${active.conference ?? ""}|${active.division ?? ""}|${active.playoffs ? 1 : 0}`;
  useEffect(() => {
    if (!season) return;
    const top = rankAt(season, season.dates.length - 1).filter((r) => isVisible(r.team.team));
    setPins(reconcilePins({}, top));
    setIndex(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewKey]);

  if (!season) return <div><h1>Power rankings</h1><div className="card">No rankings yet.</div></div>;
  const shown = index ?? season.dates.length - 1;
  const rows = rankAt(season, shown).filter((r) => isVisible(r.team.team));
  const finalists = hasPlayoffs ? playoffRanks(season, shown) : [];
  const divisionLabel = (conference: string, division: string) =>
    division.startsWith(`${conference} `) ? division.slice(conference.length + 1) : division;
  const pinnedTeams = season.teams.filter((t) => t.team.team in pins).sort((a, b) => pins[a.team.team] - pins[b.team.team]);
  const toggle = (team: string) => setPins((p) => togglePin(p, team));
  const lower = info.date_label.toLowerCase();

  return (
    <div>
      <h1>Power rankings</h1>
      <p className={cx("muted", s.intro)}>
        Each team's average predicted margin against every other team on a neutral field, before each {lower}'s games.
        Above zero, the model expects the team to beat an average opponent by that many points; higher is better.
      </p>
      <nav className={s.filters} aria-label="Season">
        {rankings.seasons.map((x) => (
          <button key={x.season} className={cx("pill", x.season === season.season && "accent")} onClick={() => setChosen(x.season)}>
            {seasonText(x.season, info.split_season)}
          </button>
        ))}
      </nav>
      {(options.conferences.length > 0 || hasPlayoffs) && (
        <nav className={s.filters} aria-label="Teams">
          <button className={cx("pill", !filtering && "accent")} aria-pressed={!filtering} onClick={() => setFilter({})}>All teams</button>
          {options.conferences.map((c) => (
            <button key={c} className={cx("pill", active.conference === c && "accent")} aria-pressed={active.conference === c}
              onClick={() => setFilter({ conference: c })}>{c}</button>
          ))}
          {hasPlayoffs && (
            <button className={cx("pill", active.playoffs && "accent")} aria-pressed={!!active.playoffs}
              onClick={() => setFilter({ playoffs: true })}>Final four</button>
          )}
        </nav>
      )}
      {active.conference && options.divisions[active.conference]?.length ? (
        <nav className={cx(s.filters, s.divisions)} aria-label={`${active.conference} divisions`}>
          <button className={cx("pill", !active.division && "accent")} aria-pressed={!active.division}
            onClick={() => setFilter({ conference: active.conference })}>All {active.conference}</button>
          {options.divisions[active.conference].map((d) => (
            <button key={d} className={cx("pill", active.division === d && "accent")} aria-pressed={active.division === d}
              onClick={() => setFilter({ conference: active.conference, division: d })}>{divisionLabel(active.conference!, d)}</button>
          ))}
        </nav>
      ) : null}
      <p className={cx("muted small", s.source)}>
        {season.source === "backtest"
          ? `${seasonText(season.season, info.split_season)}: from that season's walk-forward model, trained only on seasons before ${seasonText(season.season, info.split_season)}, with each ${lower}'s results added as they happened.`
          : `${seasonText(season.season, info.split_season)}: from the current model, with each ${lower}'s results added as they happened.`}
      </p>
      <div className="card">
        <div className={s.legend}>
          {pinnedTeams.map((t) => (
            <button key={t.team.team} className={s.chip} onClick={() => toggle(t.team.team)} aria-label={`Stop highlighting ${t.team.name}`}>
              <span className={s.swatch} style={{ background: `var(--series-${pins[t.team.team]})` }} />
              {t.team.name}
              <span className={s.x} aria-hidden="true">×</span>
            </button>
          ))}
          <span className="muted small">
            Gray lines are the other {filterLabel ? `${filterLabel} ` : ""}teams. Click a line or a row to highlight it (up to {MAX_PINS}).
          </span>
        </div>
        <RankingsChart season={season} visible={visible} dateLabel={info.date_label} pins={pins} index={index} onIndex={setIndex} onToggle={toggle} />
      </div>
      <div className="card section">
        <h2>
          Before {dateText(info.date_label, season.dates[shown], season.date_names?.[shown])}, {seasonText(season.season, info.split_season)}
          {filterLabel && <span className="muted"> · {filterLabel}</span>}
          {index !== null && <span className="muted small"> (hovered)</span>}
        </h2>
        {finalists.length > 0 && (
          <p className={cx("muted small", s.check)}>
            Where the final four ranked:{" "}
            {finalists.map(({ row, stage }, i) => (
              <span key={row.team.team}>{i ? " · " : ""}<strong className={s.finalist}>{row.team.team} #{row.rank}</strong> ({stageLabel(stage)})</span>
            ))}
          </p>
        )}
        <table className={s.ranks}>
          <thead>
            <tr>
              <th className={s.num}>{visible ? "League rank" : "Rank"}</th><th>Team</th>
              <th className={s.num}>Avg margin</th><th className={s.num}>Change</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const slot = pins[row.team.team];
              const stage = playoffs[row.team.team];
              return (
                <tr key={row.team.team} className={cx(slot !== undefined && s.pinned)}>
                  <td className={cx(s.num, "tabular")}>{row.rank}</td>
                  <td className={s["team-cell"]}>
                    <button className={s.team} onClick={() => toggle(row.team.team)} aria-pressed={slot !== undefined}>
                      <span className={s.swatch} style={{ background: slot ? `var(--series-${slot})` : "transparent" }} />
                      <TeamBadge team={row.team} label="none" size={22} />
                      <span className={cx(s.name, s.full)}>{row.team.name}</span>
                      <span className={cx(s.name, s.short)}>{row.team.nickname ?? row.team.team}</span>
                      {stage && (
                        <span className={cx(s.stage, stage === "champion" && s.champion)} title={stageLabel(stage)}>
                          <span className={s.full}>{stageLabel(stage)}</span><span className={s.short}>{SHORT[stage]}</span>
                        </span>
                      )}
                    </button>
                  </td>
                  <td className={cx(s.num, "tabular")}>{signed(row.rating)}</td>
                  <td className={cx(s.num, "tabular", s.change, (row.change ?? 0) > 0 && s.up, (row.change ?? 0) < 0 && s.down)}>
                    {row.change === null || row.change === 0 ? "–" : row.change > 0 ? `▲ ${row.change}` : `▼ ${-row.change}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
