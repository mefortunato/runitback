import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { dateText } from "../lib/format";
import { divergeColor, type Pins } from "../lib/rankings";
import { centerAt, extent, formatMetric, goodness, positionAt, ranksAt, recordText, spread, valueAt } from "../lib/teammap";
import type { TeamMap as TeamMapData, TeamMapMetric } from "../lib/types";
import { cx } from "../cx";
import s from "./TeamMap.module.css";

const PAD = 24, LABEL_ALL = 12, CHAR_W = 7, LINE = 14, SECONDS_PER_DATE = 0.9;

// Every team as a dot placed by the model's team embedding (UMAP to 2-D), filled by the chosen
// metric on a worse-to-better scale. The timeline (or play) moves every team to where it stood
// before that date; highlighted teams draw their path so far.
export default function TeamMap({ map, metric, pins, visible = null, dateLabel = "Week", frame, onFrame, onToggle }: {
  map: TeamMapData; metric: TeamMapMetric; pins: Pins;
  /** Teams to emphasize (null: all); the rest fade. */
  visible?: Set<string> | null; dateLabel?: string;
  /** Fractional date index shown. */
  frame: number; onFrame: (f: number) => void; onToggle: (team: string) => void;
}) {
  const stage = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(760);
  useLayoutEffect(() => {
    const el = stage.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  const [hover, setHover] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const last = map.dates.length - 1;

  // ---- playback
  const frameRef = useRef(frame);
  frameRef.current = frame;
  useEffect(() => {
    if (!playing) return;
    let raf = 0, before = 0;
    const tick = (now: number) => {
      const dt = before ? Math.min(0.1, (now - before) / 1000) : 0;
      before = now;
      const next = Math.min(last, frameRef.current + (dt * speed) / SECONDS_PER_DATE);
      onFrame(next);
      if (next >= last) { setPlaying(false); return; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, last, onFrame]);
  const play = () => {
    if (playing) return setPlaying(false);
    if (frame >= last) onFrame(0);
    setPlaying(true);
  };

  const W = Math.max(300, Math.round(width));
  const narrow = W < 560;
  const H = narrow ? 360 : 480;
  const SIDE = narrow ? 12 : 40;
  const R = map.teams.length > 100 ? 4.5 : 6.5;
  // every team's every position fitted to the plot, equal scale on both axes
  const box = extent(map.teams);
  const scale = Math.min((W - 2 * PAD - 2 * SIDE) / Math.max(box.x1 - box.x0, 1e-6), (H - 2 * PAD) / Math.max(box.y1 - box.y0, 1e-6));
  const px = (x: number) => W / 2 + (x - (box.x0 + box.x1) / 2) * scale;
  const py = (y: number) => H / 2 - (y - (box.y0 + box.y1) / 2) * scale;
  const shown = Math.round(frame);
  const rating = map.metrics.find((m) => m.key === "rating") ?? metric;
  // the color scale: centered per date, its width fixed for the season
  const half = spread(metric, map.teams);
  const center = centerAt(metric, map.teams, shown);
  const colorOf = (v: number | null) => (v === null ? "var(--diverge-mid)" : divergeColor(goodness(metric, v, center, half), 1));
  const ranks = ranksAt(map.teams, shown, metric);
  const isVisible = (team: string) => !visible || visible.has(team);
  const dots = map.teams
    .map((t) => {
      const p = positionAt(t, frame);
      if (!p) return null;
      return { key: t.team.team, series: t, x: px(p[0]), y: py(p[1]), value: valueAt(t, metric.key, frame),
        faded: !isVisible(t.team.team), slot: pins[t.team.team] as number | undefined };
    })
    .filter((d): d is NonNullable<typeof d> => d !== null)
    // faded teams first, highlighted ones on top
    .sort((a, b) => Number(!a.faded) - Number(!b.faded) || Number(!!a.slot) - Number(!!b.slot));
  const hovered = dots.find((d) => d.key === hover) ?? null;
  // paths so far for highlighted teams and the hovered one
  const trails = map.teams.filter((t) => t.team.team in pins || t.team.team === hover).map((t) => {
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i <= Math.floor(frame); i++) {
      const p = t.positions[i];
      if (p) points.push({ x: px(p[0]), y: py(p[1]) });
    }
    const now = positionAt(t, frame);
    if (now) points.push({ x: px(now[0]), y: py(now[1]) });
    return { key: t.team.team, slot: pins[t.team.team] as number | undefined,
      d: points.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(""), stops: points.slice(0, -1) };
  });
  // labels: highlighted teams, or every team when only a few are shown; a label overlapping one
  // above it moves below it
  const shownDots = dots.filter((d) => !d.faded);
  const labels = (shownDots.length <= LABEL_ALL ? shownDots : shownDots.filter((d) => d.slot))
    .map((d) => {
      const text = d.series.team.abbr ?? d.series.team.team;
      const right = d.x < W - 90;
      const x = right ? d.x + R + 7 : d.x - R - 7;
      return { key: d.key, text, x, right, at: d.y, pinned: !!d.slot, span: [right ? x : x - text.length * CHAR_W, right ? x + text.length * CHAR_W : x] };
    })
    .sort((a, b) => a.at - b.at);
  for (let i = 1; i < labels.length; i++)
    for (let j = 0; j < i; j++) {
      const overlaps = labels[j].span[0] < labels[i].span[1] && labels[i].span[0] < labels[j].span[1] && Math.abs(labels[i].at - labels[j].at) < LINE - 0.5;
      if (overlaps) { labels[i].at = labels[j].at + LINE; j = -1; }
    }

  // ---- pointer: hover finds the nearest dot, a click highlights it
  const nearest = (e: PointerEvent): string | null => {
    const r = svg.current!.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const y = ((e.clientY - r.top) / r.height) * H;
    let found: string | null = null, distance = 18;
    for (const d of dots) {
      if (d.faded) continue;
      const gap = Math.hypot(d.x - x, d.y - y);
      if (gap < distance) [found, distance] = [d.key, gap];
    }
    return found;
  };
  const key = (e: KeyboardEvent) => {
    if (e.key === "ArrowLeft") onFrame(Math.max(0, Math.round(frame) - 1));
    else if (e.key === "ArrowRight") onFrame(Math.min(last, Math.round(frame) + 1));
    else if (e.key === " ") play();
    else return;
    e.preventDefault();
  };
  const dateAt = (i: number) => dateText(dateLabel, map.dates[i], map.date_names?.[i]);
  const tipLeft = hovered ? (hovered.x / W) * 100 : 0;
  const tipTop = hovered ? (hovered.y / H) * 100 : 0;
  // legend ends, worse on the left: for lower-is-better metrics the high value
  const worse = metric.higher_is_better ? center - half : center + half;
  const better = metric.higher_is_better ? center + half : center - half;
  const record = hovered ? recordText(hovered.series.records?.[shown]) : null;

  return (
    <div className={s.viewer}>
      <div ref={stage} className={cx(s.stage, hover !== null && s.pointing)} role="slider" tabIndex={0} aria-roledescription="team map"
        aria-label={`Teams placed by how alike the model sees them, colored by ${metric.label.toLowerCase()}. Left and right arrows step through ${dateLabel.toLowerCase()}s, space plays.`}
        aria-valuemin={0} aria-valuemax={last} aria-valuenow={shown} aria-valuetext={`Before ${dateAt(shown)}`}
        onPointerMove={(e) => setHover(nearest(e))} onPointerDown={(e) => setHover(nearest(e))} onPointerLeave={() => setHover(null)}
        onClick={() => hover && onToggle(hover)} onKeyDown={key}>
        <svg ref={svg} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
          {trails.map((t) => (
            <g key={t.key}>
              <path className={s["trail-halo"]} d={t.d} />
              <path className={s.trail} d={t.d} style={{ stroke: t.slot ? `var(--series-${t.slot})` : "var(--text)" }} />
              {t.stops.map((p, i) => <circle key={i} className={s.stop} cx={p.x} cy={p.y} r="2.25" style={{ fill: t.slot ? `var(--series-${t.slot})` : "var(--text)" }} />)}
            </g>
          ))}
          {dots.map((d) => (
            <g key={d.key}>
              <circle className={cx(s.dot, d.faded && s.faded, d.key === hover && s.hovered)} cx={d.x} cy={d.y}
                r={d.slot || d.key === hover ? R + 1.5 : R} style={{ fill: colorOf(d.value) }} />
              {d.slot && <circle className={s.pin} cx={d.x} cy={d.y} r={R + 5} style={{ stroke: `var(--series-${d.slot})` }} />}
            </g>
          ))}
          {labels.map((l) => (
            <text key={l.key} className={cx(s.label, l.pinned && s.pinned)} x={l.x} y={l.at} textAnchor={l.right ? "start" : "end"} dominantBaseline="middle">{l.text}</text>
          ))}
        </svg>
        {hovered && (
          <div className={cx(s.tooltip, tipLeft > 60 && s.flip)} style={{ left: `${tipLeft}%`, top: `${tipTop}%` }}>
            <div className={s["tip-head"]}>{hovered.series.team.name}{record && <span className="muted">&nbsp;· {record}</span>}</div>
            <div className={s["tip-row"]}>
              <span className={s.key} style={{ background: colorOf(hovered.value) }} />
              <span>{metric.label}{ranks.has(hovered.key) && ` · #${ranks.get(hovered.key)}`}</span>
              <span className="tabular">{formatMetric(metric, hovered.value)}</span>
            </div>
            {metric.key !== "rating" && (
              <div className={s["tip-row"]}>
                <span /><span className="muted">{rating.label}</span>
                <span className="tabular">{formatMetric(rating, valueAt(hovered.series, "rating", frame))}</span>
              </div>
            )}
            <div className={s["tip-hint"]}>{hovered.key in pins ? "Click to stop highlighting" : "Click to highlight and trace its path"}</div>
          </div>
        )}
      </div>
      <div className={s.timeline}>
        <button className={s.play} onClick={play} aria-label={playing ? "Pause" : "Play the season"}>
          {playing
            ? <svg viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" rx="1" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" /></svg>
            : <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 4 2.5z" /></svg>}
        </button>
        <div className={s.track}>
          <input id="team-map-date" type="range" min="0" max={last} step="any" value={frame}
            onChange={(e) => { setPlaying(false); onFrame(+e.target.value); }}
            onPointerUp={(e) => onFrame(Math.round(+(e.target as HTMLInputElement).value))}
            aria-label={`${dateLabel} shown`} aria-valuetext={dateAt(shown)} />
          <div className={cx(s.ends, "muted small")}>
            <span>{dateAt(0)}</span><span className={s.now}>Before {dateAt(shown)}</span><span>{dateAt(last)}</span>
          </div>
        </div>
        <button className={cx("pill", s.speed)} onClick={() => setSpeed(speed >= 4 ? 1 : speed * 2)} aria-label={`Playback speed ${speed}×`}>{speed}×</button>
      </div>
      <div className={s.scale}>
        <span className="muted">Worse</span>
        <span className="tabular">{formatMetric(metric, worse)}</span>
        <span className={s.ramp} aria-hidden="true" />
        <span className="tabular">{formatMetric(metric, better)}</span>
        <span className="muted">Better</span>
        <span className={cx("muted", s.center)}>· {metric.center === "mean" ? "gray: league average" : "gray"} {formatMetric(metric, center)}</span>
      </div>
    </div>
  );
}
