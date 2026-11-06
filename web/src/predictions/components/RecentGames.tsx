import { handicap } from "../lib/format";
import type { TeamSummary } from "../lib/types";
import Sparkline from "./Sparkline";
import TeamBadge from "./TeamBadge";
import { cx } from "../cx";
import s from "./RecentGames.module.css";

const outcome = (m: number) => (m > 0 ? "W" : m < 0 ? "L" : "T");

export default function RecentGames({ summary, dateLabel = "Week" }: { summary: TeamSummary; dateLabel?: string }) {
  const short = dateLabel.charAt(0).toUpperCase();
  const margins = summary.recent.map((g) => g.margin).reverse();
  return (
    <div className={cx(s.root, s.recent)}>
      <div className={s.title}>
        <TeamBadge team={summary.team} label="name" size={24} />
        {margins.length > 0 && <Sparkline values={margins} />}
      </div>
      {summary.recent.length === 0 ? <p className="muted small">No earlier games.</p> : (
        <table className="small tabular">
          <thead>
            <tr className="muted"><th></th><th>Opponent</th><th>Score</th><th>Line</th><th>ATS</th><th className={s.right}>{dateLabel}</th></tr>
          </thead>
          <tbody>
            {summary.recent.map((g) => (
              <tr key={g.game_id}>
                <td><span className={cx("pill", g.margin > 0 && "good", g.margin < 0 && "bad")}>{outcome(g.margin)}</span></td>
                <td className={s.opp}><span className="muted">{g.location === -1 ? "@" : "vs"}</span> <TeamBadge team={g.opp} size={18} /></td>
                <td>{g.score ?? "?"}–{g.opp_score ?? "?"}</td>
                <td className="muted">{g.spread === null ? "—" : handicap(g.spread, "home")}</td>
                <td>{g.ats ? <span className={cx(s.ats, g.ats === "W" && s.good, g.ats === "L" && s.bad)}>{g.ats}</span> : <span className="muted">—</span>}</td>
                <td className={cx(s.right, "muted")}>{g.date_name ? g.date_name : <>{short}{g.date} ’{String(g.season).slice(2)}</>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
