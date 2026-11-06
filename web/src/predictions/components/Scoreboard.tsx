import type { ChallengeRecord } from "../lib/challenge";
import { pct, winRate } from "../lib/format";
import { breakEven, MARKET_LABELS, unitsLabel } from "../lib/markets";
import type { Market } from "../lib/types";
import { cx } from "../cx";
import s from "./Scoreboard.module.css";

// spreads pay -110; moneyline and over/under picks are scored in units at their posted odds
const BREAKEVEN = breakEven();

export default function Scoreboard({ record, market = "spread", onReset }: { record: ChallengeRecord; market?: Market; onReset?: () => void }) {
  const you = winRate(record.player.wins, record.player.losses);
  const model = winRate(record.model.wins, record.model.losses);
  const good = (rate: number | null, units: number) => (market === "spread" ? rate !== null && rate >= BREAKEVEN : units > 0);
  return (
    <div className={cx("card", s.scoreboard)}>
      <div className={s.side}>
        <div className="muted small">You</div>
        <div className={cx(s.big, "tabular")}>{record.player.wins}-{record.player.losses}</div>
        <div className={cx(s.rate, good(you, record.player.units) && s.good)}>{you === null ? "—" : pct(you, 1)} · {unitsLabel(record.player.units)}</div>
      </div>
      <div className={cx(s.vs, "muted")}>vs</div>
      <div className={s.side}>
        <div className="muted small">Model</div>
        <div className={cx(s.big, "tabular")}>{record.model.wins}-{record.model.losses}</div>
        <div className={cx(s.rate, good(model, record.model.units) && s.good)}>{model === null ? "—" : pct(model, 1)} · {unitsLabel(record.model.units)}</div>
      </div>
      <div className={cx(s.extra, "small muted")}>
        <div>{MARKET_LABELS[market]} · {record.played} played · streak {record.streak} (best {record.bestStreak})</div>
        <div className={s.parts}>
          <span>agreed with the model {record.played ? pct(record.agreed / record.played) : "—"}</span>
          <span>{market === "spread" ? `break-even ${pct(BREAKEVEN, 1)} at −110` : "units at the posted odds"}</span>
        </div>
        {onReset && (
          // always laid out, so the page below never shifts when it appears
          <button className={cx("small", !record.played && s.hidden)} disabled={!record.played} aria-hidden={!record.played}
            tabIndex={record.played ? 0 : -1} onClick={onReset}>Reset record</button>
        )}
      </div>
    </div>
  );
}
