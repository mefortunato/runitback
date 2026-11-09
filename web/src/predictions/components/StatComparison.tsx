import { atsRecord, seasonText, winLoss } from "../lib/format";
import type { StatDef, TeamSummary } from "../lib/types";
import TeamBadge from "./TeamBadge";
import { cx } from "../cx";
import s from "./StatComparison.module.css";

function better(stat: StatDef, a: number | null, h: number | null): "away" | "home" | null {
  if (a === null || h === null || a === h) return null;
  return (a > h) === stat.higher_is_better ? "away" : "home";
}

// bar length (% of its half) grows with the value; stats that can be negative (EPA) are measured
// from a baseline below the lower of the two
function share(a: number | null, h: number | null, side: "away" | "home") {
  if (a === null || h === null) return 0;
  const value = side === "away" ? a : h;
  const lo = Math.min(a, h, 0);
  const hi = Math.max(a, h);
  const base = lo < 0 ? lo - (Math.abs(lo) + Math.abs(hi)) / 2 : 0;
  return hi === base ? 0 : ((value - base) / (hi - base)) * 100;
}

const fmt = (v: number | null, decimals: number) => (v === null || v === undefined ? "—" : v.toFixed(decimals));

export default function StatComparison({ stats, away, home, dateLabel = "Week", splitSeason = false }: {
  stats: StatDef[]; away: TeamSummary; home: TeamSummary; dateLabel?: string; splitSeason?: boolean;
}) {
  const basis = (summary: TeamSummary) => {
    const games = `${summary.games} game${summary.games === 1 ? "" : "s"}`;
    const season = seasonText(summary.season, splitSeason);
    return summary.current_season ? `${season} season, ${games}` : `${season} season (last season), ${games}`;
  };
  return (
    <div className={s.compare}>
      <div className={s.head}>
        <div>
          <TeamBadge team={away.team} label="name" />
          <div className="muted small">{basis(away)}</div>
        </div>
        <div className={cx(s.center, "muted small")}>per game,<br />before this {dateLabel.toLowerCase()}</div>
        <div className={s.right}>
          <TeamBadge team={home.team} label="name" />
          <div className="muted small">{basis(home)}</div>
        </div>
      </div>
      <div className={cx(s.row, s.record)}>
        <span className={cx(s.value, "tabular")}>{winLoss(away.record)}</span>
        <span className={s.label}>Record</span>
        <span className={cx(s.value, "tabular", s.right)}>{winLoss(home.record)}</span>
      </div>
      <div className={cx(s.row, s.record)}>
        <span className={cx(s.value, "tabular")}>{atsRecord(away.ats)}</span>
        <span className={s.label}>Against the spread</span>
        <span className={cx(s.value, "tabular", s.right)}>{atsRecord(home.ats)}</span>
      </div>
      {stats.map((stat) => {
        const a = away.averages[stat.key] ?? null;
        const h = home.averages[stat.key] ?? null;
        const best = better(stat, a, h);
        return (
          <div key={stat.key} className={s.row}>
            <span className={cx(s.value, "tabular", best === "away" && s.best)}>{fmt(a, stat.decimals)}</span>
            <span className={s.label}>
              {stat.label}
              <span className={s.bars}>
                <span className={cx(s.half, s.left)}>
                  <span className={cx(s.bar, best === "away" && s.best)} style={{ width: `${share(a, h, "away")}%` }} />
                </span>
                <span className={s.half}>
                  <span className={cx(s.bar, best === "home" && s.best)} style={{ width: `${share(a, h, "home")}%` }} />
                </span>
              </span>
            </span>
            <span className={cx(s.value, "tabular", s.right, best === "home" && s.best)}>{fmt(h, stat.decimals)}</span>
          </div>
        );
      })}
    </div>
  );
}
