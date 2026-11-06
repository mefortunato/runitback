import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { dateText, seasonText } from "../lib/format";
import { filterOptions, matchesFilter, MAX_PINS, togglePin, type Pins, type TeamFilter } from "../lib/rankings";
import { formatChange, formatMetric, positionAt, ranksAt } from "../lib/teammap";
import type { SportInfo, TeamMap as TeamMapData } from "../lib/types";
import TeamBadge from "../components/TeamBadge";
import TeamMap from "../components/TeamMap";
import { useData } from "../useData";
import { cx } from "../cx";
import s from "./Map.module.css";

export default function MapPage({ info }: { info: SportInfo }) {
  const { data, error } = useData(() => api.teamMap(info.sport), [info.sport]);
  if (error) return <p className="error">{error}</p>;
  if (data === undefined) return <p className="muted">Loading…</p>;
  return <MapView info={info} map={data} />;
}

function MapView({ info, map }: { info: SportInfo; map: TeamMapData | null }) {
  const [frame, setFrame] = useState(map ? map.dates.length - 1 : 0);
  const [metricKey, setMetricKey] = useState("rating");
  const [filter, setFilter] = useState<TeamFilter>({});
  const [pins, setPins] = useState<Pins>({});
  const onFrame = useCallback((f: number) => setFrame(f), []);
  const metric = map?.metrics.find((m) => m.key === metricKey) ?? map?.metrics[0];
  const last = map ? map.dates.length - 1 : 0;
  const shown = Math.round(frame);
  const options = filterOptions(map ? map.teams.map((t) => t.team) : []);
  const filtering = Boolean(filter.conference);
  const visible = map && filtering ? new Set(map.teams.filter((t) => matchesFilter(t.team, filter)).map((t) => t.team.team)) : null;
  const isVisible = (team: string) => !visible || visible.has(team);
  const divisionLabel = (conference: string, division: string) =>
    division.startsWith(`${conference} `) ? division.slice(conference.length + 1) : division;
  // every filter change starts from that view's top teams at the latest date
  const filterKey = `${filter.conference ?? ""}|${filter.division ?? ""}`;
  useEffect(() => {
    if (!map) return;
    const top = [...ranksAt(map.teams, last).keys()].filter(isVisible).slice(0, MAX_PINS);
    setPins(Object.fromEntries(top.map((team, i) => [team, i + 1])));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterKey, map]);
  const toggle = useCallback((team: string) => setPins((p) => togglePin(p, team)), []);
  const lower = info.date_label.toLowerCase();

  if (!map) {
    return <div><h1>Team map</h1><div className="card">No team map for this sport yet.</div></div>;
  }
  const pinned = map.teams.filter((t) => t.team.team in pins).sort((a, b) => pins[a.team.team] - pins[b.team.team]);
  const dateAt = (i: number) => dateText(info.date_label, map.dates[i], map.date_names?.[i]);
  // who the model sees most differently than at the season's first date
  const movers = !metric || shown === 0 ? [] : (() => {
    const ranks = ranksAt(map.teams, shown, metric);
    return map.teams
      .filter((t) => isVisible(t.team.team))
      .map((t) => {
        const values = t.values[metric.key] ?? [];
        const [a, b] = [positionAt(t, 0), positionAt(t, shown)];
        const moved = a && b ? Math.hypot(a[0] - b[0], a[1] - b[1]) : null;
        // the change since the first date the team had a value
        const now = values[shown] ?? null;
        const first = values.slice(0, shown).find((v) => v !== null) ?? null;
        return { t, moved, value: now, change: now !== null && first !== null ? now - first : null, rank: ranks.get(t.team.team) ?? null };
      })
      .filter((m) => m.moved !== null)
      .sort((a, b) => (b.moved as number) - (a.moved as number))
      .slice(0, 10);
  })();

  return (
    <div>
      <h1>Team map</h1>
      <p className={cx("muted", s.intro)}>
        The model keeps an internal picture of each team, a long list of numbers learned from its recent games. This map
        squeezes those pictures into two dimensions with UMAP, so teams the model sees as alike sit close together. Every{" "}
        {lower} of the season is laid out in the same space: press play to watch teams move as their results come in. The
        axes have no units; only closeness means anything. Color the teams by any stat to see what the model's picture
        lines up with.
      </p>
      <p className={cx("muted small", s.source)}>
        {seasonText(map.season, info.split_season)}, from the current model, before each {lower}'s games.
      </p>
      {options.conferences.length > 0 && (
        <nav className={s.filters} aria-label="Teams">
          <button className={cx("pill", !filtering && "accent")} aria-pressed={!filtering} onClick={() => setFilter({})}>All teams</button>
          {options.conferences.map((c) => (
            <button key={c} className={cx("pill", filter.conference === c && !filter.division && "accent")} aria-pressed={filter.conference === c}
              onClick={() => setFilter({ conference: c })}>{c}</button>
          ))}
        </nav>
      )}
      {filter.conference && options.divisions[filter.conference]?.length ? (
        <nav className={cx(s.filters, s.divisions)} aria-label={`${filter.conference} divisions`}>
          {options.divisions[filter.conference].map((d) => (
            <button key={d} className={cx("pill", filter.division === d && "accent")} aria-pressed={filter.division === d}
              onClick={() => setFilter(filter.division === d ? { conference: filter.conference } : { conference: filter.conference, division: d })}>
              {divisionLabel(filter.conference!, d)}
            </button>
          ))}
        </nav>
      ) : null}
      <div className="card">
        <div className={s.controls}>
          <label className={s["color-by"]} htmlFor="team-map-metric">
            <span className="muted small">Color by</span>
            <select id="team-map-metric" value={metricKey} onChange={(e) => setMetricKey(e.target.value)}>
              {map.metrics.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </label>
        </div>
        <div className={s.legend}>
          {pinned.map((t) => (
            <button key={t.team.team} className={s.chip} onClick={() => toggle(t.team.team)} aria-label={`Stop highlighting ${t.team.name}`}>
              <span className={s.swatch} style={{ background: `var(--series-${pins[t.team.team]})` }} />
              {t.team.name}
              <span className={s.x} aria-hidden="true">×</span>
            </button>
          ))}
          <span className="muted small">Highlighted teams trace their path. Click a dot to highlight it (up to {MAX_PINS}).</span>
        </div>
        {metric && <TeamMap map={map} metric={metric} pins={pins} visible={visible} dateLabel={info.date_label} frame={frame} onFrame={onFrame} onToggle={toggle} />}
      </div>
      <div className="card section">
        <h2>Moved most since {dateAt(0)} <span className="muted small">before {dateAt(shown)}</span></h2>
        {!movers.length ? <p className="muted small">Move the timeline past the first {lower} to compare.</p> : (
          <>
            <p className={cx("muted small", s.note)}>
              Distance each team has traveled through the map since the first {lower}: the teams the model has changed its mind
              about most. Change is in {metric?.label.toLowerCase()} since the team's first games.
            </p>
            <div className={s.scroll}>
              <table className={s.movers}>
                <thead>
                  <tr><th>Team</th><th className={s.num}>Moved</th><th className={s.num}>{metric?.label}</th><th className={s.num}>Rank</th><th className={s.num}>Change</th></tr>
                </thead>
                <tbody>
                  {movers.map((m) => {
                    const slot = pins[m.t.team.team];
                    const up = metric && !!m.change && m.change > 0 === metric.higher_is_better;
                    const down = metric && !!m.change && m.change > 0 !== metric.higher_is_better;
                    return (
                      <tr key={m.t.team.team} className={cx(slot !== undefined && s.pinned)}>
                        <td className={s["team-cell"]}>
                          <button className={s.team} onClick={() => toggle(m.t.team.team)} aria-pressed={slot !== undefined}>
                            <span className={s.swatch} style={{ background: slot ? `var(--series-${slot})` : "transparent" }} />
                            <TeamBadge team={m.t.team} label="none" size={22} />
                            <span className={s.name}>{m.t.team.name}</span>
                          </button>
                        </td>
                        <td className={cx(s.num, "tabular")}>{(m.moved as number).toFixed(2)}</td>
                        <td className={cx(s.num, "tabular")}>{metric ? formatMetric(metric, m.value) : ""}</td>
                        <td className={cx(s.num, "tabular")}>{m.rank ?? "–"}</td>
                        <td className={cx(s.num, "tabular", up && s.up, down && s.down)}>{metric ? formatChange(metric, m.change) : "–"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
