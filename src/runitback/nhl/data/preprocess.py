"""NHL generator training data: every game replayed through the rules engine, one row per event.

Input: the `nhl_pbp_full` release files (`raw/pbp_<season>.parquet`, `runitback nhl download`) and
the schedule (`raw/games_<season>.parquet`, game dates and final scores). Each game goes through the
H4 replay (`engine.replay`): `game_events` turns the release rows into engine outcomes and
`repair_events` fills the feed's gaps. The events are then applied to the engine in order, with the
penalties of each stoppage labeled by their effect on the strength (`label_penalties`), exactly as
`replay_game` does. Every row holds the engine's state *before* the event (what the generator sees)
and the event (what it predicts):

- the clock: an event's `elapsed` is measured against the engine's own clock, clamped to what the
  engine allows (`elapsed_range`: 0 outside live play, the rest of the period for a period end), so
  the engine's clock follows the data's and every elapsed target is legal;
- legality: the masks the sampler will use, from the engine itself (`legal_types`, `legal_sides`,
  `legal_results`), stored as bit masks: `type_mask` over PLAY_TYPES, `side_mask` over
  (home, away, none) given the real type, `result_mask` over RESULTS given the real type. A real
  event the engine calls illegal (well under 1%) has its own bit added, so it stays trainable, and
  `legal` is False;
- the score is re-synced to the data after each goal (as in `replay_game`): an engine miss is
  counted once instead of shifting every later state.

A game is kept when the engine finished it (`over`) with the data's final score and at least
MIN_LEGAL of its events legal; the rest are counted in the summary and dropped.

Outputs (in the output directory, `data/nhl/plays` by default):
- plays.parquet    one row per event: game_id, idx, state before it, play_type, side (-1 none,
                   0 home, 1 away), result, elapsed, masks, legal
- games.parquet    one row per kept game: game_id (the NHL's, as a string), season (the year it
                   ends), game_type (REG / POST), game_date, home, away (current franchise codes),
                   home_final, away_final (a shootout win counts one goal), periods, shootout,
                   n_events, legal share, xg_h / xg_a (the release's expected goals per team, no
                   shootout; NaN where the release has none)
- ratings.parquet  leak-free pre-game team ratings (`ratings.py`)
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.replay import STRENGTH, game_events, label_penalties, repair_events, split_games
from ..engine.rules import (
    AWAY,
    HOME,
    GameState,
    Outcome,
    apply_outcome,
    elapsed_range,
    legal_results,
    legal_sides,
    legal_types,
    new_game,
    period_length,
    shootout_shooter,
)
from ..vocab import MAX_ELAPSED, PLAY_TYPES, PT, RESULTS, RS

log = logging.getLogger(__name__)

MIN_LEGAL = 0.97
SIDES = [HOME, AWAY, None]           # side index 2 = no team
SIDE_IDX = {HOME: 0, AWAY: 1, None: 2}
RAW_COLUMNS = [
    "event_type", "secondary_type", "event_team_type", "period", "period_type", "period_seconds",
    "home_score", "away_score", "penalty_severity", "penalty_minutes", "home_skaters", "away_skaters",
    "home_goalie", "away_goalie", "game_id", "season", "season_type", "home_abbr", "away_abbr",
    "event_idx", "reason", "event_player_1_id", "home_on_1_id", "away_on_1_id", "xg",
]
STATE_COLUMNS = ["period", "psec", "phase", "score_h", "score_a", "sk_h", "sk_a", "en_h", "en_a",
                 "pen_h", "pen_a", "pen_left_h", "pen_left_a", "so_shots_h", "so_shots_a", "so_goals_h",
                 "so_goals_a", "so_next"]


def bits(idx) -> int:
    out = 0
    for i in idx:
        out |= 1 << int(i)
    return out


def state_row(s: GameState) -> dict:
    sk = s.skaters
    left = [min((p[1] for p in s.serving(side)), default=0) for side in (HOME, AWAY)]
    nxt = shootout_shooter(s)
    return dict(period=s.period, psec=s.psec, phase=s.phase, score_h=s.score[HOME], score_a=s.score[AWAY],
                sk_h=sk[HOME], sk_a=sk[AWAY], en_h=s.empty_net[HOME], en_a=s.empty_net[AWAY],
                pen_h=len(s.serving(HOME)), pen_a=len(s.serving(AWAY)), pen_left_h=left[HOME],
                pen_left_a=left[AWAY], so_shots_h=s.so_shots[HOME], so_shots_a=s.so_shots[AWAY],
                so_goals_h=s.so_goals[HOME], so_goals_a=s.so_goals[AWAY],
                so_next=SIDE_IDX[nxt] if nxt is not None else 2)


def masks(s: GameState, o: Outcome) -> tuple[int, int, int, int, int, bool]:
    """Bit masks of what the engine allows in state `s` (types; sides and results given the real
    type), the legal elapsed range for it, and whether `o` itself is legal."""
    types = legal_types(s)
    tm = bits(PT[t] for t in types)
    sides = legal_sides(s, o.play_type)
    sm = bits(SIDE_IDX[x] for x in sides)
    rm = bits(RS[r] for r in legal_results(s, o.play_type))
    lo, hi = elapsed_range(s, o.play_type)
    legal = o.play_type in types and o.side in sides and o.result in legal_results(s, o.play_type)
    return tm, sm, rm, lo, hi, legal


def game_plays(meta: dict, events: list) -> tuple[list[dict], GameState]:
    """Apply a game's (repaired) events to the engine. Returns one row per event (the state before
    it, the event, the engine's masks) and the final state."""
    s = new_game(meta["home"], meta["away"], meta["season"], meta["game_type"])
    rows: list[dict] = []
    labeled: set[int] = set()
    for i, e in enumerate(events):
        if e.play_type == "penalty" and id(e) not in labeled:
            group = [x for x in events[i:] if x.play_type == "penalty" and x.period == e.period
                     and x.t == e.t and x.result in STRENGTH]
            label_penalties(group, s)
            labeled.update(id(x) for x in events[i:] if x.play_type == "penalty"
                           and x.period == e.period and x.t == e.t)
        # elapsed against the engine's clock (seconds into the period), within the engine's range
        clock = period_length(s.period, s.game_type) - s.psec
        lo, hi = elapsed_range(s, e.play_type)
        elapsed = int(min(max(e.t - clock if e.period == s.period else 0, lo), hi))
        o = Outcome(e.play_type, e.result, e.side, elapsed)
        tm, sm, rm, lo, hi, legal = masks(s, o)
        rows.append(dict(state_row(s), idx=i, play_type=o.play_type, side=SIDE_IDX[o.side], result=o.result,
                         elapsed=min(elapsed, MAX_ELAPSED), type_mask=tm | (1 << PT[o.play_type]),
                         side_mask=sm | (1 << SIDE_IDX[o.side]), result_mask=rm | (1 << RS[o.result]),
                         el_lo=min(lo, elapsed), el_hi=max(min(hi, MAX_ELAPSED), elapsed), legal=legal))
        n = apply_outcome(s, o)
        if n.over:
            return rows, n
        if e.score is not None and n.phase != 3:  # re-sync the score (not in the shootout)
            n.score = list(e.score)
        s = n
    return rows, s


def season_plays(path: Path, schedule: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every game of one release file -> (plays, games, summary)."""
    import pyarrow.parquet as pq
    cols = [c for c in RAW_COLUMNS if c in set(pq.read_schema(path).names)]
    df = pd.read_parquet(path, columns=cols)
    sched = schedule.set_index(schedule.game_id.astype(str)) if schedule is not None and len(schedule) else None
    plays, games = [], []
    info = dict(games=0, kept=0, unparsed=0, shootout_unordered=0, not_over=0, wrong_final=0, illegal=0,
                events=0, legal_events=0)
    for g in split_games(df):
        res = game_events(g)
        if res is None:
            info["unparsed"] += 1
            continue
        meta, events = res
        info["games"] += 1
        if meta.get("shootout_unordered"):
            info["shootout_unordered"] += 1
            continue
        events, _ = repair_events(meta, events)
        rows, final = game_plays(meta, events)
        legal = float(np.mean([r["legal"] for r in rows])) if rows else 0.0
        info["events"] += len(rows)
        info["legal_events"] += int(sum(r["legal"] for r in rows))
        row = sched.loc[meta["game_id"]] if sched is not None and meta["game_id"] in sched.index else None
        date = str(row.game_date)[:10] if row is not None else None
        data_final = ((int(row.home_final), int(row.away_final))
                      if row is not None and pd.notna(row.home_final) and pd.notna(row.away_final)
                      else (meta["home_goals"], meta["away_goals"]))
        if not final.over:
            info["not_over"] += 1
            continue
        so = final.phase == 3
        if row is not None and pd.notna(row.home_final):
            ok = tuple(final.score) == data_final
        elif so:  # no schedule: the release's goals before the shootout, plus one for the winner
            ok = meta["home_goals"] == meta["away_goals"] and sorted(final.score) == \
                [meta["home_goals"], meta["home_goals"] + 1]
        else:
            ok = tuple(final.score) == data_final
        if not ok:
            info["wrong_final"] += 1
            continue
        if legal < MIN_LEGAL:
            info["illegal"] += 1
            continue
        xg = {}
        if "xg" in g:
            reg = g[(g.period_type != "SHOOTOUT") & g.event_type.isin(("GOAL", "SHOT", "MISSED_SHOT"))]
            for side, name in ((HOME, "home"), (AWAY, "away")):
                v = reg.loc[reg.event_team_type == name, "xg"]
                xg[side] = float(v.sum()) if v.notna().any() else np.nan
        games.append(dict(game_id=meta["game_id"], season=meta["season"], game_type=meta["game_type"],
                          game_date=date, home=meta["home"], away=meta["away"], home_final=final.score[HOME],
                          away_final=final.score[AWAY], periods=min(final.period, 4) if not so else 4,
                          shootout=bool(so), n_events=len(rows), legal=legal,
                          xg_h=xg.get(HOME, np.nan), xg_a=xg.get(AWAY, np.nan)))
        for r in rows:
            r["game_id"] = meta["game_id"]
        plays += rows
        info["kept"] += 1
    info["legal_share"] = round(info["legal_events"] / max(info["events"], 1), 5)
    plays_df = pd.DataFrame(plays)
    games_df = pd.DataFrame(games)
    if len(games_df):
        season = int(games_df.season.mode().iloc[0])
        if games_df.game_date.isna().any():
            # no schedule: order by game id within the season (ids are assigned in schedule order)
            games_df["game_date"] = games_df.game_date.fillna(f"{season - 1}-10-01")
        info["season"] = season
    return plays_df, games_df, info


def _season_job(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path, sched_path = args
    schedule = pd.read_parquet(sched_path) if sched_path is not None and Path(sched_path).exists() else None
    return season_plays(Path(path), schedule)


PLAY_DTYPES = dict(idx="int16", period="int8", psec="int16", phase="int8", score_h="int8", score_a="int8",
                   sk_h="int8", sk_a="int8", en_h="bool", en_a="bool", pen_h="int8", pen_a="int8",
                   pen_left_h="int16", pen_left_a="int16", so_shots_h="int8", so_shots_a="int8",
                   so_goals_h="int8", so_goals_a="int8", so_next="int8", side="int8", elapsed="int16",
                   type_mask="int32", side_mask="int8", result_mask="int64", el_lo="int16", el_hi="int16",
                   legal="bool")


def preprocess(seasons, raw_dir: Path, out_dir: Path, workers: int = 1) -> dict:
    """plays.parquet, games.parquet and ratings.parquet for the seasons with release data."""
    from .ratings import build_ratings
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in seasons:
        path = Path(raw_dir) / f"pbp_{s}.parquet"
        if not path.exists():
            log.warning("season %s: no play-by-play (runitback nhl download)", s)
            continue
        jobs.append((str(path), str(Path(raw_dir) / f"games_{s}.parquet")))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_season_job, jobs))
    else:
        results = [_season_job(j) for j in jobs]
    plays = [p for p, _, _ in results if len(p)]
    games = [g for _, g, _ in results if len(g)]
    if not games:
        raise FileNotFoundError(f"no NHL play-by-play under {raw_dir}")
    games = pd.concat(games, ignore_index=True).drop_duplicates("game_id")
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    plays = pd.concat(plays, ignore_index=True)
    plays = plays[plays.game_id.isin(games.game_id)]
    plays = plays.drop_duplicates(["game_id", "idx"]).astype(PLAY_DTYPES)
    cols = ["game_id", "idx", *STATE_COLUMNS, "play_type", "side", "result", "elapsed", "type_mask",
            "side_mask", "result_mask", "el_lo", "el_hi", "legal"]
    plays[cols].to_parquet(out_dir / "plays.parquet", index=False)
    games.to_parquet(out_dir / "games.parquet", index=False)
    build_ratings(games, out_dir)
    summary = {int(r[2]["season"]): {k: v for k, v in r[2].items() if k != "season"}
               for r in results if "season" in r[2]}
    summary["total"] = dict(games=int(len(games)), events=int(len(plays)),
                            legal_share=round(float(plays.legal.mean()), 5),
                            max_events=int(games.n_events.max()),
                            events_p999=int(games.n_events.quantile(0.999)))
    return summary


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    data_dir = Path(data_dir)
    games = pd.read_parquet(data_dir / "games.parquet")
    plays = pd.read_parquet(data_dir / "plays.parquet")
    r = data_dir / "ratings.parquet"
    return games, plays, pd.read_parquet(r) if r.exists() else None


__all__ = ["preprocess", "season_plays", "game_plays", "state_row", "masks", "load", "SIDES", "SIDE_IDX",
           "PLAY_TYPES", "RESULTS"]
