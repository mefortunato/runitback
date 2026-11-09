import { useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { dateText, signed } from "../lib/format";
import { rankAt, yExtent, type Pins } from "../lib/rankings";
import type { RankingSeason } from "../lib/types";
import { cx } from "../cx";
import s from "./RankingsChart.module.css";

const CHAR = 7; // px per character of a 12px label, a little generous
const LABEL_ALL = 4;
const labelSpace = (texts: string[]) => 24 + CHAR * Math.max(0, ...texts.map((t) => t.length));

// Every team's rating by date: highlighted teams in their series color with an end label, the rest
// as thin gray context. Hover finds the nearest line; click highlights it.
export default function RankingsChart({ season, dateLabel = "Week", pins, visible = null, index, onIndex, onToggle }: {
  season: RankingSeason; dateLabel?: string; pins: Pins;
  /** Teams to draw (null: all). The axis and ranks stay league-wide. */
  visible?: Set<string> | null;
  /** Hovered date index (null: none). */
  index: number | null; onIndex: (i: number | null) => void; onToggle: (team: string) => void;
}) {
  // drawn at the container's own width, so text stays 12px on phones too
  const box = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(760);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  const [hoverTeam, setHoverTeam] = useState<string | null>(null);

  const W = Math.max(300, Math.round(width));
  const narrow = W < 560;
  const H = narrow ? 320 : 400;
  // end labels for highlighted teams (for every line when only a few are drawn), nudged apart
  const drawn = season.teams.filter((t) => !visible || visible.has(t.team.team));
  const labelled = (drawn.length <= LABEL_ALL ? drawn : drawn.filter((t) => t.team.team in pins))
    .map((t) => {
      let i = t.ratings.length - 1;
      while (i >= 0 && t.ratings[i] === null) i--;
      return i < 0 ? null : { key: t.team.team, team: t.team, value: t.ratings[i] as number };
    })
    .filter((l): l is NonNullable<typeof l> => l !== null);
  // short abbreviations when full names would take more than a third of the chart
  const full = labelled.map((l) => `${l.key} ${signed(l.value)}`);
  const labelTexts = labelSpace(full) <= W / 3
    ? new Map(labelled.map((l, i) => [l.key, full[i]]))
    : new Map(labelled.map((l) => [l.key, `${l.team.abbr ?? l.key} ${signed(l.value)}`]));
  const M = { top: 14, right: Math.max(narrow ? 74 : 92, labelSpace([...labelTexts.values()])), bottom: 34, left: 40 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const extent = yExtent(season);
  const first = season.dates[0];
  const last = season.dates[season.dates.length - 1];
  const x = (d: number) => M.left + (last === first ? plotW / 2 : ((d - first) / (last - first)) * plotW);
  const y = (v: number) => M.top + ((extent.max - v) / (2 * extent.max)) * plotH;
  const yTicks: number[] = [];
  for (let v = -extent.max; v <= extent.max + 1e-9; v += extent.step) yTicks.push(v);
  const every = Math.ceil(season.dates.length / (narrow ? 8 : 14));
  const xTicks = season.dates.filter((_, i) => i % every === 0 || i === season.dates.length - 1);
  const path = (ratings: (number | null)[]) => {
    let d = "", open = false;
    ratings.forEach((v, i) => {
      if (v === null) { open = false; return; }
      d += `${open ? "L" : "M"}${x(season.dates[i]).toFixed(1)},${y(v).toFixed(1)}`;
      open = true;
    });
    return d;
  };
  const lines = drawn.map((t) => ({ key: t.team.team, team: t.team, ratings: t.ratings, d: path(t.ratings) }));
  const pinned = lines.filter((l) => l.key in pins).sort((a, b) => pins[a.key] - pins[b.key]);
  const endLabels = labelled
    .map((l) => ({ key: l.key, slot: pins[l.key] as number | undefined, text: labelTexts.get(l.key) ?? l.key, at: y(l.value) }))
    .sort((a, b) => a.at - b.at);
  for (let i = 1; i < endLabels.length; i++) endLabels[i].at = Math.max(endLabels[i].at, endLabels[i - 1].at + 15);

  const shown = index ?? season.dates.length - 1;
  const rankOf = new Map(rankAt(season, shown).map((r) => [r.team.team, r]));
  const tipTeams = [...pinned.map((l) => l.key), ...(hoverTeam && !(hoverTeam in pins) ? [hoverTeam] : [])]
    .map((k) => rankOf.get(k))
    .filter((r): r is NonNullable<typeof r> => r !== undefined)
    .sort((a, b) => a.rank - b.rank);

  const locate = (e: PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const py = ((e.clientY - r.top) / r.height) * H;
    let best = 0;
    season.dates.forEach((d, i) => { if (Math.abs(x(d) - px) < Math.abs(x(season.dates[best]) - px)) best = i; });
    onIndex(best);
    // nearest line at that date, within a forgiving hit distance
    let nearest: string | null = null, distance = 18;
    for (const l of lines) {
      const v = l.ratings[best];
      if (v === null) continue;
      const dy = Math.abs(y(v) - py);
      if (dy < distance) [nearest, distance] = [l.key, dy];
    }
    setHoverTeam(nearest);
  };
  const leave = () => { onIndex(null); setHoverTeam(null); };
  const key = (e: KeyboardEvent) => {
    const n = season.dates.length;
    if (e.key === "ArrowLeft") onIndex(Math.max(0, (index ?? n - 1) - 1));
    else if (e.key === "ArrowRight") onIndex(Math.min(n - 1, (index ?? n - 1) + 1));
    else if (e.key === "Escape") leave();
    else return;
    e.preventDefault();
  };
  const tipLeft = index === null ? 0 : (x(season.dates[index]) / W) * 100;
  const color = (team: string) => (team in pins ? `var(--series-${pins[team]})` : "var(--text)");

  return (
    <div ref={box} className={cx(s.root, s.chart, hoverTeam !== null && s.pointing)} role="slider" tabIndex={0}
      aria-label={`${dateLabel} shown in the table below; left and right arrows step through ${dateLabel.toLowerCase()}s`}
      aria-valuemin={first} aria-valuemax={last} aria-valuenow={season.dates[shown]}
      aria-valuetext={dateText(dateLabel, season.dates[shown], season.date_names?.[shown])}
      onPointerMove={locate} onPointerDown={locate} onPointerLeave={leave}
      onClick={() => hoverTeam && onToggle(hoverTeam)} onKeyDown={key}>
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Average predicted margin of every team by ${dateLabel.toLowerCase()}`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line className={cx(s.grid, t === 0 && s.zero)} x1={M.left} x2={M.left + plotW} y1={y(t)} y2={y(t)} />
            <text className={s.tick} x={M.left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle">{t === 0 ? "0" : signed(t, 0)}</text>
          </g>
        ))}
        {xTicks.map((d) => <text key={d} className={s.tick} x={x(d)} y={M.top + plotH + 20} textAnchor="middle">{d}</text>)}
        <text className={s.axis} x={M.left + plotW / 2} y={H - 2} textAnchor="middle">{dateLabel}</text>
        {index !== null && <line className={s.crosshair} x1={x(season.dates[index])} x2={x(season.dates[index])} y1={M.top} y2={M.top + plotH} />}
        {lines.filter((l) => !(l.key in pins) && l.key !== hoverTeam).map((l) => <path key={l.key} className={s.context} d={l.d} />)}
        {lines.filter((l) => l.key === hoverTeam && !(l.key in pins)).map((l) => (
          <g key={l.key}><path className={s.halo} d={l.d} /><path className={s.hover} d={l.d} /></g>
        ))}
        {pinned.map((l) => (
          <g key={l.key}>
            <path className={s.halo} d={l.d} />
            <path className={cx(s.pinned, l.key === hoverTeam && s.emphasis)} d={l.d} style={{ stroke: `var(--series-${pins[l.key]})` }} />
          </g>
        ))}
        {index !== null && tipTeams.map((r) => (
          <circle key={r.team.team} className={s.marker} cx={x(season.dates[index])} cy={y(r.rating)} r="4.5" style={{ fill: color(r.team.team) }} />
        ))}
        {endLabels.map((l) => (
          <g key={l.key}>
            <circle className={s.swatch} cx={M.left + plotW + 10} cy={l.at} r="4" style={{ fill: l.slot ? `var(--series-${l.slot})` : "var(--context-line)" }} />
            <text className={cx(s.end, !l.slot && s.context)} x={M.left + plotW + 18} y={l.at} dominantBaseline="middle">{l.text}</text>
          </g>
        ))}
      </svg>
      {index !== null && tipTeams.length > 0 && (
        <div className={cx(s.tooltip, tipLeft > 60 && s.flip)} style={{ left: `${tipLeft}%` }}>
          <div className={s["tip-head"]}>{dateText(dateLabel, season.dates[index], season.date_names?.[index])}</div>
          {tipTeams.map((r) => (
            <div key={r.team.team} className={s["tip-row"]}>
              <span className={s.dot} style={{ background: color(r.team.team) }} />
              <span className={s["tip-team"]}>#{r.rank} {r.team.team}</span>
              <span className="tabular">{signed(r.rating)}</span>
            </div>
          ))}
          {hoverTeam && !(hoverTeam in pins) && <div className={s["tip-hint"]}>Click to highlight</div>}
        </div>
      )}
    </div>
  );
}
