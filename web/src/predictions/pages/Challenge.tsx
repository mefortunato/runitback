import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useChallengeRecord } from "../lib/challenge";
import { atsRecord, dateText, handicap, marginLabel, pct, seasonText, spreadLabel, winLoss } from "../lib/format";
import { americanOdds, MARKET_LABELS, pickLabel, SPREAD_ODDS, totalLabel, unitsLabel } from "../lib/markets";
import type { ChallengeAnswer, ChallengeGame, Market, Pick, SportInfo, StatDef, Team, TeamSummary } from "../lib/types";
import MatchupDetails from "../components/MatchupDetails";
import ProbabilityCompare from "../components/ProbabilityCompare";
import Scoreboard from "../components/Scoreboard";
import SpreadChart from "../components/SpreadChart";
import TeamBadge from "../components/TeamBadge";
import TotalChart from "../components/TotalChart";
import { remember, remembered } from "../useData";
import { cx } from "../cx";
import s from "./Challenge.module.css";

const MARKET_KEY = "runitback:challenge-market";

// Blind picks: until the pick is locked in, the two teams show only as "Away" and "Home". Their
// opponents stay named, so recent form still has context. Answering reveals the real teams.
const BLIND: Record<"home" | "away", Team> = {
  away: { team: "Away", name: "Away team", abbr: "A" },
  home: { team: "Home", name: "Home team", abbr: "H" },
};
const blindSummary = (summary: TeamSummary, side: "home" | "away"): TeamSummary => ({ ...summary, team: BLIND[side] });

interface Choice { pick: Pick; title: string; line: string; note: string }
// a team's season at a glance, for its card: record, ATS, scoring, last 5
interface Glance { record: string; ats: string; scored: string; allowed: string; form: ("W" | "L" | "T")[]; games: string }

const PROMPT: Record<Market, string> = {
  spread: "Pick the side that covers the spread",
  moneyline: "Pick the winner, at the posted moneyline odds",
  total: "Pick whether the combined score goes over or under",
};

export default function Challenge({ info }: { info: SportInfo }) {
  const sport = info.sport;
  const markets: Market[] = info.markets?.length ? info.markets : ["spread"];
  const [market, setMarket] = useState<Market>(() => {
    const saved = remembered<Market | "">(MARKET_KEY, "");
    return saved && markets.includes(saved) ? saved : markets[0];
  });
  // one record per sport and market, kept in this browser
  const { record, add, reset } = useChallengeRecord(sport, market);
  const [game, setGame] = useState<ChallengeGame | null>(null);
  const [answer, setAnswer] = useState<ChallengeAnswer | null>(null);
  const [pick, setPick] = useState<Pick | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const pickCard = useRef<HTMLDivElement>(null);
  const seenRef = useRef(record.seen);
  seenRef.current = record.seen;

  const next = useCallback(async () => {
    setBusy(true);
    setFailure(null);
    try {
      const g = await api.challengeNext(sport, seenRef.current.slice(-200), market);
      setGame(g);
      setAnswer(null);
      setPick(null);
      // the buttons stay where they are from game to game, so only scroll when the pick card's top
      // has gone off screen
      const top = pickCard.current?.getBoundingClientRect().top;
      if (top !== undefined && top < 0) window.scrollBy({ top: top - 12 });
    } catch (e) {
      setFailure(String(e));
    } finally {
      setBusy(false);
    }
  }, [sport, market]);
  useEffect(() => { next(); }, [next]);

  const submit = async () => {
    if (!game || !pick) return;
    setBusy(true);
    try {
      const a = await api.challengeAnswer(sport, game.game_id, pick, market);
      setAnswer(a);
      add(a);
    } catch (e) {
      setFailure(String(e));
    } finally {
      setBusy(false);
    }
  };
  const choose = (m: Market) => { if (m !== market) { setMarket(m); remember(MARKET_KEY, m); } };

  const shown: ChallengeGame | null = !game || answer ? game : {
    ...game, home: BLIND.home, away: BLIND.away,
    summaries: { home: blindSummary(game.summaries.home, "home"), away: blindSummary(game.summaries.away, "away") },
  };

  const glance = (g: ChallengeGame, side: "home" | "away"): Glance => {
    const sm = g.summaries[side];
    const [scored, allowed] = g.stats;
    const fmt = (stat: StatDef | undefined) => {
      const v = stat ? sm.averages[stat.key] : null;
      return v === null || v === undefined ? "—" : v.toFixed(stat?.decimals ?? 1);
    };
    return {
      record: winLoss(sm.record), ats: atsRecord(sm.ats), scored: fmt(scored), allowed: fmt(allowed),
      // oldest first, so the newest result is on the right
      form: sm.recent.slice(0, 5).reverse().map((r) => (r.margin > 0 ? "W" : r.margin < 0 ? "L" : "T")),
      // the season the numbers cover: this one, or last season before an opener
      games: sm.current_season ? `${sm.games} game${sm.games === 1 ? "" : "s"}` : `${seasonText(sm.season, info.split_season)} season`,
    };
  };

  const choices: Choice[] = !shown ? [] : market === "moneyline" && shown.moneyline
    ? (["away", "home"] as const).map((side) => ({ pick: side, title: (side === "home" ? shown.home : shown.away).name, line: americanOdds(shown.moneyline?.[side]), note: "wins" }))
    : market === "total"
    ? (["over", "under"] as const).map((side) => ({ pick: side, title: side === "over" ? "Over" : "Under", line: totalLabel(shown.total_line),
        note: shown.total_odds ? americanOdds(shown.total_odds[side]) : americanOdds(SPREAD_ODDS) }))
    : (["away", "home"] as const).map((side) => ({ pick: side, title: (side === "home" ? shown.home : shown.away).name, line: `${side} ${handicap(shown.spread ?? 0, side)}`, note: "covers" }));

  const lineText = !shown ? "" : market === "moneyline" && shown.moneyline
    ? `Moneyline ${shown.away.team} ${americanOdds(shown.moneyline.away)} · ${shown.home.team} ${americanOdds(shown.moneyline.home)}`
    : market === "total" ? `Over/under ${totalLabel(shown.total_line)}` : `Line ${spreadLabel(shown.home, shown.away, shown.spread)}`;

  const outcomeText = (a: ChallengeAnswer, g: ChallengeGame) =>
    a.market === "moneyline" ? `${pickLabel(a.outcome, g.home, g.away)} won`
      : a.market === "total" ? `${a.result.total} points · ${a.outcome === "over" ? "over" : "under"} ${totalLabel(a.total_line)}`
      : `${pickLabel(a.outcome, g.home, g.away)} covered`;
  const modelText = (a: ChallengeAnswer, g: ChallengeGame) => {
    const m = a.model;
    if (a.market === "moneyline") {
      const side = m.pick === "away" ? "away" : "home";
      const team = side === "home" ? g.home : g.away;
      const p = side === "home" ? m.win_prob : 1 - m.win_prob;
      const v = m.market_prob == null ? null : side === "home" ? m.market_prob : 1 - m.market_prob;
      return `${team.team} ${pct(p)}${v === null ? "" : ` vs Vegas ${pct(v)}`}`;
    }
    if (a.market === "total") return `${m.total?.toFixed(1) ?? "—"} points`;
    return marginLabel(g.home, g.away, m.margin);
  };

  const panel = (g: ChallengeGame, teams: Record<"home" | "away", Glance>, side: "away" | "home", choice: Choice | null) => {
    const t = teams[side];
    const team = side === "home" ? g.home : g.away;
    return (
      <>
        <span className={s.head}>
          <TeamBadge team={team} size={28} label="none" />
          <span className={s.names}>
            <span className={s["team-name"]}>{team.name}</span>
            <span className={cx(s.where, "muted")}>{side === "home" ? "Home" : "Away"} · {t.games}</span>
          </span>
          {choice && (
            <span className={s["line-box"]}>
              <span className={cx(s.line, "tabular")}>{choice.line.replace(/^(home|away) /, "")}</span>
              <span className={s.covers}>{choice.note}</span>
            </span>
          )}
        </span>
        <span className={cx(s.glance, "tabular")}>
          <span><b>{t.record}</b><small>Record</small></span>
          <span><b>{t.ats}</b><small>ATS</small></span>
          <span><b>{t.scored}</b><small>Scored</small></span>
          <span><b>{t.allowed}</b><small>Allowed</small></span>
        </span>
        <span className={s.form}>
          <small className="muted">Last {t.form.length || 5}</small>
          {t.form.length ? t.form.map((r, i) => <span key={i} className={cx(s.dot, r === "W" && s.w, r === "L" && s.l)}>{r}</span>)
            : <small className="muted">no games yet</small>}
        </span>
      </>
    );
  };

  return (
    <div>
      <div className={s.intro}>
        <div>
          <h1>Beat the model</h1>
          <p className="muted">
            A past game, its Vegas lines, and what both teams had done before kickoff, with the two teams kept anonymous:
            you see their stats, records and recent opponents, but not who they are. Choose a market, make your pick, then
            see the teams, the result and what the model picked.
          </p>
          <p className="muted small">
            The model never saw these games: its pick for a game from a given season comes from a model trained only on
            earlier seasons.
          </p>
        </div>
      </div>
      {markets.length > 1 && (
        <div className={s.tabs} role="tablist" aria-label="Market">
          {markets.map((m) => (
            <button key={m} role="tab" aria-selected={market === m} className={cx(market === m && s.active)} disabled={busy} onClick={() => choose(m)}>
              {MARKET_LABELS[m]}
            </button>
          ))}
        </div>
      )}
      <Scoreboard record={record} market={market} onReset={reset} />
      {failure && <p className="error">{failure}</p>}
      {shown ? (() => {
        const g = shown;
        const teams = { away: glance(g, "away"), home: glance(g, "home") };
        return (
          <>
            <div className={cx("card section", s.pick)} ref={pickCard}>
              <div className={cx(s.context, "muted small")}>
                {dateText(info.date_label, g.date, g.date_name)}, {seasonText(g.season, info.split_season)}{g.neutral ? " · neutral site" : ""} · {lineText}
              </div>
              <div className={cx("small", s.prompt)}>{PROMPT[market]}</div>
              {market === "total" && (
                <div className={cx(s.choices, s["teams-row"])}>
                  {(["away", "home"] as const).map((side) => <div key={side} className={cx(s.choice, s.static)}>{panel(g, teams, side, null)}</div>)}
                </div>
              )}
              <div className={cx(s.choices, market === "total" && s.totals)} role="group" aria-label={PROMPT[market]}>
                {choices.map((c) => {
                  const chosen = pick === c.pick;
                  const won = !!answer && answer.outcome === c.pick;
                  return (
                    <button key={c.pick} className={cx(s.choice, chosen && s.chosen, won && s.won, answer && chosen && !won && s.lost)}
                      disabled={!!answer || busy} onClick={() => setPick(c.pick)}>
                      {c.pick === "home" || c.pick === "away" ? panel(g, teams, c.pick, c) : (
                        <span className={s["total-choice"]}>
                          <span className={s["total-icon"]} aria-hidden="true">{c.pick === "over" ? "▲" : "▼"}</span>
                          <span className={s["team-name"]}>{c.title}</span>
                          <span className={cx(s.line, "tabular")}>{c.line}</span>
                          <span className={s.covers}>{c.note}</span>
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
              {/* one slot for both actions, so the button never moves */}
              {!answer
                ? <button className={cx("primary", s.action)} disabled={!pick || busy} onClick={submit}>{pick ? `Lock in ${pickLabel(pick, g.home, g.away)}` : "Make a pick"}</button>
                : <button className={cx("primary", s.action)} onClick={next} disabled={busy}>Next game →</button>}
              {answer && (
                <div className={s.reveal}>
                  <div className={s.final}>
                    Final: {g.away.team} <strong className="tabular">{answer.result.away_score}</strong> – <strong className="tabular">{answer.result.home_score}</strong> {g.home.team} · {outcomeText(answer, g)}
                  </div>
                  <div className={s.verdicts}>
                    <span className={cx("pill", answer.correct ? "good" : "bad")}>
                      You: {pickLabel(answer.pick, g.home, g.away)} {answer.correct ? "✓" : "✗"} {unitsLabel(answer.units, 2)}
                    </span>
                    <span className={cx("pill", answer.model.correct ? "good" : "bad")}>
                      Model: {modelText(answer, g)} → {pickLabel(answer.model.pick, g.home, g.away)} {answer.model.correct ? "✓" : "✗"} {unitsLabel(answer.model.units, 2)}
                    </span>
                  </div>
                  <p className={cx("muted small", s.trained)}>
                    The model's pick for this {seasonText(g.season, info.split_season)} game comes from a model trained only on seasons before {seasonText(g.season, info.split_season)}.
                  </p>
                  <div className={s.chart}>
                    {answer.market === "moneyline"
                      ? <ProbabilityCompare home={g.home} away={g.away} model={answer.model.win_prob} market={answer.model.market_prob ?? null} />
                      : answer.market === "total"
                      ? <TotalChart line={answer.total_line ?? null} model={answer.model.total ?? null} result={answer.result.total ?? null} />
                      : <SpreadChart home={g.home} away={g.away} spread={answer.spread ?? g.spread} model={answer.model.margin} result={answer.result.margin} />}
                  </div>
                </div>
              )}
            </div>
            <details className={cx(s.advanced, "section")}>
              <summary className="card">
                <span><strong>Advanced stats</strong> <span className="muted small">· {g.stats.length} stats &amp; recent games</span></span>
                <span className={s.chevron} aria-hidden="true">▾</span>
              </summary>
              <MatchupDetails matchup={g} dateLabel={info.date_label} splitSeason={info.split_season} />
            </details>
          </>
        );
      })() : busy ? <p className="muted section">Loading a game…</p> : null}
    </div>
  );
}
