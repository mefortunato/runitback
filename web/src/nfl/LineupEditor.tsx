import { useEffect, useMemo, useState } from "react";
import { api, NEW_PLAYER, type CoachInfo, type LineupEdit, type PlayerInfo, type TeamLineupView } from "./api";

const GROUP_NAMES: Record<string, string> = {
  QB: "Quarterbacks", RB: "Running backs", WR: "Wide receivers", TE: "Tight ends", OL: "Offensive line",
  DL: "Defensive line", LB: "Linebackers", DB: "Defensive backs", K: "Kicker", P: "Punter", LS: "Long snapper",
};

export function lineupFromView(v: TeamLineupView): LineupEdit {
  return { coach: v.coach ?? "", depth_chart: structuredClone(v.depth_chart), edited: false };
}

function Badge({ p }: { p: PlayerInfo }) {
  if (p.status === "new") return <span className="badge new" title="Modeled with the generic NEW PLAYER token for this position">new player</span>;
  if (p.status === "limited history") return <span className="badge generic" title="Too little history for an individual rating; modeled as a generic veteran at this position">generic</span>;
  return null;
}

function pct(x: number) { return x.toFixed(3).replace(/^0/, ""); }

function CoachLine({ name, info }: { name: string; info: CoachInfo | undefined }) {
  if (!name) return <p className="muted small">No coach: the model uses an unknown-coach input.</p>;
  if (!info || info.status === "new") {
    return <p className="small"><span className="badge new">new coach</span> No head-coaching history; modeled as NEW COACH.</p>;
  }
  return (
    <p className="small">
      {info.hc_seasons} prior season{info.hc_seasons === 1 ? "" : "s"} as head coach
      ({info.team_seasons} with this team) · {info.career_games} games · {pct(info.career_win_pct)} win% (shrunk to .500)
      {info.status === "limited history" && <> <span className="badge generic">generic</span></>}
    </p>
  );
}

interface Props {
  side: string;
  view: TeamLineupView | null;
  value: LineupEdit | null;
  coaches: CoachInfo[];
  season: number;
  onChange: (v: LineupEdit) => void;
  onReset: () => void;
}

export default function LineupEditor({ side, view, value, coaches, season, onChange, onReset }: Props) {
  const [query, setQuery] = useState<Record<string, string>>({});
  const [results, setResults] = useState<Record<string, PlayerInfo[]>>({});
  const [coachInfo, setCoachInfo] = useState<CoachInfo | undefined>();

  const coachName = value?.coach ?? "";
  useEffect(() => {
    // The record of a coach depends on the team (seasons with this team), so ask the server.
    const known = coaches.find((c) => c.name === coachName);
    if (!known || !view) { setCoachInfo(undefined); return; }
    let live = true;
    api.coaches(season, view.team, coachName)
      .then((d) => live && setCoachInfo(d.coaches.find((c) => c.name === coachName)))
      .catch(() => live && setCoachInfo(known));
    return () => { live = false; };
  }, [coachName, coaches, view, season]);

  const inChart = useMemo(() => new Set(Object.values(value?.depth_chart ?? {}).flat().map((p) => p.id)), [value]);
  if (!view || !value) return <div className="lineup-team muted">Loading {side} lineup…</div>;

  const set = (group: string, players: PlayerInfo[]) =>
    onChange({ ...value, depth_chart: { ...value.depth_chart, [group]: players }, edited: true });
  const move = (group: string, i: number, d: number) => {
    const list = [...value.depth_chart[group]];
    const j = i + d;
    if (j < 0 || j >= list.length) return;
    [list[i], list[j]] = [list[j], list[i]];
    set(group, list);
  };
  const add = (group: string, p: PlayerInfo) => {
    const list = value.depth_chart[group] ?? [];
    if (list.length >= view.max_depth[group]) return;
    set(group, [...list, p]);
    setQuery((q) => ({ ...q, [group]: "" }));
    setResults((r) => ({ ...r, [group]: [] }));
  };
  const search = async (group: string, q: string) => {
    setQuery((s) => ({ ...s, [group]: q }));
    if (q.trim().length < 2) { setResults((r) => ({ ...r, [group]: [] })); return; }
    try {
      const d = await api.players(q, group);
      setResults((r) => ({ ...r, [group]: d.players.filter((p) => !inChart.has(p.id)) }));
    } catch { /* ignore */ }
  };

  const src = view.source;
  return (
    <div className="lineup-team">
      <div className="lineup-head">
        <h3>{view.team} <span className="muted">({side})</span></h3>
        <button onClick={onReset} disabled={!value.edited} title="Back to the most recent game's lineup">Reset</button>
      </div>
      <p className="muted small">
        {src ? <>Default: who played in {src.season} week {src.week} ({src.game_id.split("_").slice(2).join(" @ ")}), ranked by snaps.</>
          : <>No recent game found; add players below.</>}
        {value.edited && <strong> · edited</strong>}
      </p>
      <label className="coach-field">Head coach
        <input list={`coaches-${side}`} value={coachName} placeholder="Coach name"
          onChange={(e) => onChange({ ...value, coach: e.target.value, edited: true })} />
        <datalist id={`coaches-${side}`}>{coaches.map((c) => <option key={c.name} value={c.name} />)}</datalist>
      </label>
      <CoachLine name={coachName} info={coachInfo} />
      <div className="groups">
        {view.groups.map((g) => {
          const list = value.depth_chart[g] ?? [];
          const full = list.length >= view.max_depth[g];
          const pool = (view.pool[g] ?? []).filter((p) => !inChart.has(p.id));
          return (
            <fieldset key={g} className="group">
              <legend>{GROUP_NAMES[g] ?? g} <span className="muted">{list.length}/{view.max_depth[g]}</span></legend>
              {list.length === 0 && <p className="muted small empty">Empty: plays at this position use a generic player.</p>}
              <ol>
                {list.map((p, i) => (
                  <li key={`${p.id}-${i}`}>
                    <span className="depth">{g}{i + 1}</span>
                    <span className="pname" title={p.id}>{p.name}</span>
                    <span className="muted small">{p.career_games != null ? `${p.career_games} g` : ""}</span>
                    <Badge p={p} />
                    <span className="row-actions">
                      <button aria-label={`Move ${p.name} up`} disabled={i === 0} onClick={() => move(g, i, -1)}>↑</button>
                      <button aria-label={`Move ${p.name} down`} disabled={i === list.length - 1} onClick={() => move(g, i, 1)}>↓</button>
                      <button aria-label={`Remove ${p.name}`} onClick={() => set(g, list.filter((_, k) => k !== i))}>✕</button>
                    </span>
                  </li>
                ))}
              </ol>
              {!full && (
                <div className="add-row">
                  <select value="" aria-label={`Add to ${g}`} onChange={(e) => {
                    const v = e.target.value;
                    if (v === "NEW") add(g, { ...NEW_PLAYER, group: g });
                    else { const p = pool.find((x) => x.id === v); if (p) add(g, p); }
                  }}>
                    <option value="">Add…</option>
                    <option value="NEW">＋ New player (no NFL history)</option>
                    {pool.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.position}, {p.career_games} g)</option>)}
                  </select>
                  <input className="search" placeholder="Search league" value={query[g] ?? ""}
                    onChange={(e) => search(g, e.target.value)} aria-label={`Search players for ${g}`} />
                </div>
              )}
              {(results[g]?.length ?? 0) > 0 && (
                <ul className="results">
                  {results[g].map((p) => (
                    <li key={p.id}><button onClick={() => add(g, p)}>{p.name} <span className="muted">{p.team} · {p.position} · {p.career_games} g</span></button></li>
                  ))}
                </ul>
              )}
            </fieldset>
          );
        })}
      </div>
    </div>
  );
}
