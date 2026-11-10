"""MLB projected lineups: each team-game's starting pitcher and batting order, from what is known
before first pitch.

Three pieces (``docs/multisport/PLAN.md``, M3):

- **Starting pitcher.** ``sp="probable"`` (the default) is the feed's probable pitcher, as the
  sportsml loader (``runitback.wager.mlb``) uses. **The leak:** for finished games the Stats API
  reports the pitcher who started as the probable (they agree in about 99.8% of games), so a late
  scratch counts as known. Starters are announced days ahead and scratches are rare, so the leak
  is small, but it is a leak. ``sp="rotation"`` is the leak-free alternative: the pitcher who
  started the team's game ``ROTATION`` (5) team-games ago this season, if he has not appeared for
  another team since and is rested; a game without a probable falls back to it as well.
- **Batting order** (``lineup="previous"``): the team's most recent starting lineup **against a
  starter of the same throwing hand** as this game's (projected) opposing starter, in a game with
  the same designated-hitter rule. The lookups, in order (``source``):
  ``prev_hand`` (this season, same hand and DH rule), ``prev`` (this season, same DH rule),
  ``prev_season_hand`` / ``prev_season`` (the same from earlier seasons, at the start of a
  season). Then the copied lineup is brought up to date with what is known before the game:

  - a batter who has since appeared for another team (traded, released) is replaced by the
    player who most recently started at his fielding position for the team (``filled``), or left
    vacant (``player_id`` missing) when there is none;
  - the pitcher's spot (no DH) gets this game's projected starter; when the copied lineup's DH
    rule differs from this game's (only the last fallbacks), the DH and pitcher swap places.

  Only earlier games are read: the team's own games before this one (a doubleheader's first game
  counts for the second), and other teams' games on earlier dates (who has moved).
- **Posted lineup** (``lineup="posted"``): MLB teams post their batting order a few hours before
  first pitch. The Stats API keeps no history of what was posted, so the historical stand-in is
  the game's actual starting nine (the box score's batting order). The leak: a late scratch
  after the lineup was posted counts as known. For upcoming games (:func:`live_lineups`) the
  posted lineup is the one listed now, when the team has posted it, else the projection.

The **DH rule** of a game is known before it: universal in 2020 and from 2022; before that, only
in American League parks (:func:`uses_dh`, from the home team).

Output (``projected_lineups.parquet`` / ``posted_lineups.parquet`` in the processed directory),
one row per (game, side, role, spot): ``game_pk, season, game_type, date, side, team_id, opp_id,
role ("SP" | "B"), spot (1-9 for batters; the SP's batting spot when he bats, else 0),
position (fielding code: 1 P .. 9 RF, 10 DH), player_id, hand (bats for B, throws for SP),
opp_sp_id, opp_sp_throws, dh, source, sp_source, ref_game_pk`` (the game the lineup was copied
from).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPOTS = 9
ROTATION = 5          # sp="rotation": the starter of the team's game this many team-games ago
MIN_REST_DAYS = 4     # ... if he has not pitched in the last MIN_REST_DAYS - 1 days
PITCHER, DH = "1", "10"
# American League team ids (Stats API); Houston moved from the NL in 2013.
AL_TEAMS = {108, 110, 111, 114, 116, 118, 133, 136, 139, 140, 141, 142, 145, 147}
HOU, HOU_AL_FROM = 117, 2013

COLUMNS = ["game_pk", "season", "game_type", "date", "side", "team_id", "opp_id", "role", "spot",
           "position", "player_id", "hand", "opp_sp_id", "opp_sp_throws", "dh", "source", "sp_source",
           "ref_game_pk"]


def uses_dh(season: int, home_id: int) -> bool:
    """Whether a game is played with the designated hitter: every game in 2020 and from 2022,
    before that only in American League parks (postseason included)."""
    if season == 2020 or season >= 2022:
        return True
    return int(home_id) in AL_TEAMS or (int(home_id) == HOU and season >= HOU_AL_FROM)


def throw_hand(code) -> str | None:
    """A pitcher's throwing hand for platoon purposes: ``L`` or ``R`` (switch pitchers count as R)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    return "L" if code == "L" else "R"


# ---------------------------------------------------------------------------------- load

def load(processed: Path, seasons: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """``games``, ``lineups`` and ``players`` of the seasons in ``processed`` (all, or ``seasons``)."""
    processed = Path(processed)
    found = sorted(int(p.stem.split("_")[1]) for p in processed.glob("games_*.parquet"))
    seasons = [s for s in found if seasons is None or s in set(seasons)]
    if not seasons:
        raise FileNotFoundError(f"no games_<season>.parquet in {processed} (run `runitback mlb process`)")
    out = {}
    for name in ("games", "lineups", "players"):
        frames = [pd.read_parquet(processed / f"{name}_{s}.parquet") for s in seasons
                  if (processed / f"{name}_{s}.parquet").exists()]
        out[name] = pd.concat(frames, ignore_index=True)
    out["players"] = out["players"].drop_duplicates("player_id", keep="last")
    return out


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per side), in the order they were played: ``game_pk, season,
    game_type, date, start, side, team_id, opp_id, home_id, probable_sp, opp_probable_sp, sp,
    opp_sp, dh`` (the rule, :func:`uses_dh`)."""
    g = games.copy()
    g["start"] = pd.to_datetime(g["start"], utc=True, errors="coerce")
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "game_pk": g.game_pk.astype("int64"), "season": g.season.astype(int), "game_type": g.game_type,
            "date": g.date.astype(str), "start": g.start,
            "game_number": g.get("game_number", pd.Series(1, index=g.index)).fillna(1).astype(int),
            "side": side, "team_id": g[f"{side}_id"].astype("int64"), "opp_id": g[f"{other}_id"].astype("int64"),
            "home_id": g.home_id.astype("int64"),
            "probable_sp": g[f"{side}_probable_sp"], "opp_probable_sp": g[f"{other}_probable_sp"],
            "sp": g[f"{side}_sp"], "opp_sp": g[f"{other}_sp"]}))
    tg = pd.concat(rows, ignore_index=True)
    tg["dh"] = [uses_dh(s, h) for s, h in zip(tg.season, tg.home_id)]
    return tg.sort_values(["date", "start", "game_number", "game_pk", "side"], kind="stable") \
        .reset_index(drop=True)


def starting_lineups(lineups: pd.DataFrame) -> dict[tuple[int, str], list[tuple[int, int, str]]]:
    """(game_pk, side) -> the starting nine as [(spot, player_id, start_position)] by spot."""
    st = lineups[lineups.starter.astype(bool)].sort_values(["game_pk", "side", "spot"], kind="stable")
    out: dict[tuple[int, str], list] = defaultdict(list)
    for pk, side, spot, pid, pos in zip(st.game_pk, st.side, st.spot, st.player_id, st.start_position):
        out[(int(pk), side)].append((int(spot), int(pid), str(pos)))
    return dict(out)


# ------------------------------------------------------------------------------ projection

class _History:
    """What is known before each game, updated as games are played (see :func:`projected_lineups`)."""

    def __init__(self, absent: int | None = None):
        self.absent = absent
        self.recent = defaultdict(list)      # team -> [set of players who batted], one per game
        self.last = {}                       # (team, key...) -> (game_pk, lineup)
        self.pos_starters = defaultdict(list)  # (team, season, position) -> [player ids], latest last
        self.sp_order = defaultdict(list)    # (team, season) -> [starter ids], latest last
        self.last_team = {}                  # player -> team of his latest appearance (earlier dates)
        self.last_pitched = {}               # pitcher -> date of his latest appearance (earlier dates)
        self.pending_team = {}
        self.pending_pitched = {}

    def lookup(self, team: int, season: int, hand: str | None, dh: bool):
        tries = [("prev_hand", (team, season, hand, dh)), ("prev", (team, season, dh)),
                 ("prev_season_hand", (team, hand, dh)), ("prev_season", (team, dh)),
                 ("prev_season", (team,))]
        for source, key in tries:
            if (hand is None and source.endswith("hand")) or key not in self.last:
                continue
            pk, lineup, src_dh = self.last[key]
            return source, pk, lineup, src_dh
        return "none", None, None, None

    def record(self, row: dict, lineup, opp_hand: str | None, dh_used: bool, pitchers: list[int],
               batted: set[int] = frozenset()):
        """After a game: its starting nine (faced a starter throwing ``opp_hand``, DH used or not)
        and its pitchers in order."""
        team, season = row["team_id"], row["season"]
        entry = (row["game_pk"], lineup, dh_used)
        keys = [(team, season, dh_used), (team, dh_used), (team,)]
        if opp_hand is not None:
            keys += [(team, season, opp_hand, dh_used), (team, opp_hand, dh_used)]
        for key in keys:
            self.last[key] = entry
        for spot, pid, pos in lineup:
            if pos not in (PITCHER,):
                lst = self.pos_starters[(team, season, pos)]
                lst.append(pid)
                del lst[:-20]
            self.pending_team[pid] = team
        rec = self.recent[team]
        rec.append(set(batted) | {pid for _, pid, _ in lineup})
        del rec[:-20]
        if pitchers:
            self.sp_order[(team, season)].append(pitchers[0])
        for pid in pitchers:
            self.pending_team[pid] = team
            self.pending_pitched[pid] = row["date"]

    def end_of_date(self):
        self.last_team.update(self.pending_team)
        self.last_pitched.update(self.pending_pitched)
        self.pending_team.clear()
        self.pending_pitched.clear()

    def with_team(self, pid: int, team: int) -> bool:
        return self.last_team.get(pid, team) == team

    def available(self, pid: int, team: int) -> bool:
        """Still with the team and, with ``absent``, batted in one of its last ``absent`` games."""
        if not self.with_team(pid, team):
            return False
        rec = self.recent.get(team, [])
        if self.absent is None or len(rec) < self.absent:
            return True
        return any(pid in g for g in rec[-self.absent:])

    def rotation_sp(self, team: int, season: int, date: str) -> int | None:
        order = self.sp_order.get((team, season), [])
        if len(order) < ROTATION:
            return None
        cand = order[-ROTATION]
        last = self.last_pitched.get(cand)
        rested = last is None or (pd.Timestamp(date) - pd.Timestamp(last)).days >= MIN_REST_DAYS
        return cand if self.with_team(cand, team) and rested else None

    def fill(self, team: int, season: int, pos: str, taken: set[int]) -> int | None:
        for pid in reversed(self.pos_starters.get((team, season, pos), [])):
            if pid not in taken and self.available(pid, team):
                return pid
        return None


def _adapt(lineup, src_dh: bool, dh: bool, sp: int | None, hist: _History, team: int, season: int):
    """A copied lineup brought up to date: movers replaced, the pitcher's spot given to ``sp``, and
    the DH / pitcher swapped when the DH rule differs. Returns [(spot, player_id | None, position,
    filled)]."""
    out = []
    taken = {pid for _, pid, _ in lineup}
    for spot, pid, pos in lineup:
        if pos == PITCHER:
            if dh:   # the copied game had no DH, this one has: the pitcher's spot goes to a DH
                new = hist.fill(team, season, DH, taken)
                out.append((spot, new, DH, True))
                if new is not None:
                    taken.add(new)
            else:
                out.append((spot, sp, PITCHER, False))
            continue
        if pos == DH and not dh:   # the copied game had a DH, this one has not
            out.append((spot, sp, PITCHER, False))
            continue
        if hist.available(pid, team):
            out.append((spot, pid, pos, False))
            continue
        new = hist.fill(team, season, pos, taken)
        out.append((spot, new, pos, True))
        if new is not None:
            taken.add(new)
    return out


def projected_lineups(games: pd.DataFrame, lineups: pd.DataFrame, players: pd.DataFrame,
                      lineup: str = "previous", sp: str = "probable",
                      seasons: set[int] | None = None, absent: int | None = None) -> pd.DataFrame:
    """Projected (``lineup="previous"``) or posted (``"posted"``) lineups of every team-game in
    ``games`` (or those of ``seasons``), with the probable (``sp="probable"``) or rotation
    (``"rotation"``) starting pitcher. See the module docstring."""
    if lineup not in ("previous", "posted") or sp not in ("probable", "rotation"):
        raise ValueError(f"lineup={lineup!r}, sp={sp!r}")
    tg = team_games(games)
    starting = starting_lineups(lineups)
    pitchers = (lineups.dropna(subset=["pitcher_order"]).sort_values("pitcher_order")
                .groupby(["game_pk", "side"]).player_id.agg(list).to_dict())
    throws = {int(k): v for k, v in zip(players.player_id, players.throws.map(throw_hand))}
    bats = {int(k): v for k, v in zip(players.player_id, players.bats)}
    hist = _History(absent)
    batted = lineups.dropna(subset=["batting_order"]).groupby(["game_pk", "side"]).player_id \
        .agg(lambda v: set(int(p) for p in v)).to_dict()
    out: list[dict] = []
    cur_date = None
    recs = tg.to_dict("records")
    i = 0
    while i < len(recs):
        pair = [recs[i]]
        while i + len(pair) < len(recs) and recs[i + len(pair)]["game_pk"] == pair[0]["game_pk"]:
            pair.append(recs[i + len(pair)])
        i += len(pair)
        if len(pair) != 2:
            continue   # a game listed with one side only
        x0 = pair[0]
        if x0["date"] != cur_date:
            hist.end_of_date()
            cur_date = x0["date"]
        # the projected starters of both sides first: each side's lineup depends on the other's
        proj = {}
        for x in pair:
            prob = None if pd.isna(x["probable_sp"]) else int(x["probable_sp"])
            if sp == "probable" and prob is not None:
                proj[x["side"]] = (prob, "probable")
            else:
                rot = hist.rotation_sp(x["team_id"], x["season"], x["date"])
                # the first games of a season have no rotation yet: their starters (opening day's
                # are named weeks ahead) come from the probables
                proj[x["side"]] = (rot, "rotation") if rot is not None or prob is None else (prob, "probable")
        for x, y in (pair, pair[::-1]):
            if seasons is not None and x["season"] not in seasons:
                continue
            opp_sp = proj[y["side"]][0]
            opp_hand = throws.get(opp_sp) if opp_sp is not None else None
            _emit(out, x, lineup, *proj[x["side"]], opp_sp, opp_hand,
                  starting.get((x["game_pk"], x["side"]), []), hist, throws, bats)
        for x in pair:   # then what this game tells the later ones
            actual = starting.get((x["game_pk"], x["side"]), [])
            if not actual:
                continue
            opp_sp = None if pd.isna(x["opp_sp"]) else int(x["opp_sp"])
            hist.record(x, actual, throws.get(opp_sp), any(pos == DH for _, _, pos in actual),
                        [int(p) for p in pitchers.get((x["game_pk"], x["side"]), [])],
                        batted.get((x["game_pk"], x["side"]), set()))
    return _types(pd.DataFrame(out, columns=COLUMNS))


def _types(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("player_id", "opp_sp_id", "ref_game_pk"):
        df[c] = pd.array(df[c], dtype="Int64")
    df["spot"] = df["spot"].astype("int16")
    df["dh"] = df["dh"].astype(bool)
    return df.reset_index(drop=True)


def _emit(out, x, lineup, sp_id, sp_source, opp_sp, opp_hand, actual, hist, throws, bats):
    base = dict(game_pk=x["game_pk"], season=x["season"], game_type=x["game_type"], date=x["date"],
                side=x["side"], team_id=x["team_id"], opp_id=x["opp_id"], opp_sp_id=opp_sp,
                opp_sp_throws=opp_hand, dh=x["dh"], sp_source=sp_source)
    if lineup == "posted":
        source, ref = ("posted", x["game_pk"]) if actual else ("none", None)
        bat = [(spot, sp_id if pos == PITCHER else pid, pos, False) for spot, pid, pos in actual]
    else:
        source, ref, copied, src_dh = hist.lookup(x["team_id"], x["season"], opp_hand, x["dh"])
        bat = _adapt(copied, src_dh, x["dh"], sp_id, hist, x["team_id"], x["season"]) if copied else []
    sp_spot = next((s for s, _, pos, _ in bat if pos == PITCHER), 0)
    out.append(base | dict(role="SP", spot=sp_spot, position=PITCHER, player_id=sp_id,
                           hand=throws.get(sp_id), source=source, ref_game_pk=ref))
    for spot, pid, pos, filled in bat:
        hand = throws.get(pid) if pos == PITCHER else bats.get(pid)
        out.append(base | dict(role="B", spot=spot, position=pos, player_id=pid, hand=hand,
                               source=("vacant" if pid is None else "filled") if filled else source,
                               ref_game_pk=ref))


# ------------------------------------------------------------------------------------- evaluation

def evaluate(proj: pd.DataFrame, games: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Per season, the projection against what happened, next to the naive baseline of the team's
    previous game's lineup as played (``prev_*``), over the team-games that have a previous game:

    - ``sp``: share of team-games whose projected starter started;
    - ``batters``: share of the actual starting batters (pitchers batting excluded) in the
      projected lineup; ``spots``: share in the right batting spot; ``all``: share of team-games
      with every batter right (spots ignored);
    - ``hand_match``: share of team-games whose lineup came from a game against the same hand
      (``source == "prev_hand"``)."""
    tg = team_games(games)
    starting = starting_lineups(lineups)
    tg["prev"] = tg.groupby("team_id").game_pk.shift(1)
    actual_sp = {(r.game_pk, r.side): r.sp for r in tg.itertuples(index=False)}
    prev_of = {(r.game_pk, r.side): r.prev for r in tg.itertuples(index=False)}
    side_of = {(int(pk), int(t)): s for pk, t, s in zip(tg.game_pk, tg.team_id, tg.side)}

    def bats(lineup):
        return {(spot, pid) for spot, pid, pos in lineup if pos != PITCHER}

    rows = defaultdict(lambda: defaultdict(float))
    for (pk, side, season), d in proj.groupby(["game_pk", "side", "season"], sort=False):
        act = bats(starting.get((pk, side), []))
        prev = prev_of[(pk, side)]
        team = int(d.team_id.iloc[0])
        if not act or pd.isna(prev) or (int(prev), team) not in side_of:
            continue   # compared on team-games with a previous game in the data only
        r = rows[int(season)]
        r["team_games"] += 1
        sp_row = d[d.role == "SP"]
        r["sp"] += float(len(sp_row) and pd.notna(sp_row.player_id.iloc[0])
                         and sp_row.player_id.iloc[0] == actual_sp[(pk, side)])
        b = d[(d.role == "B") & (d.position != PITCHER)].dropna(subset=["player_id"])
        pr = {(int(s), int(p)) for s, p in zip(b.spot, b.player_id)}
        ids, act_ids = {p for _, p in pr}, {p for _, p in act}
        r["batters"] += len(ids & act_ids) / len(act_ids)
        r["spots"] += len(pr & act) / len(act)
        r["all"] += float(act_ids <= ids)
        r["hand_match"] += float((d.source == "prev_hand").any())
        pv = bats(starting.get((int(prev), side_of[(int(prev), team)]), []))
        pv_ids = {p for _, p in pv}
        r["prev_batters"] += len(pv_ids & act_ids) / len(act_ids)
        r["prev_spots"] += len(pv & act) / len(act)
        r["prev_all"] += float(act_ids <= pv_ids)
    out = {}
    for season, r in sorted(rows.items()):
        n = n_prev = r["team_games"]
        out[season] = dict(team_games=int(n), sp=r["sp"] / n, batters=r["batters"] / n, spots=r["spots"] / n,
                           all=r["all"] / n, hand_match=r["hand_match"] / n,
                           prev_batters=r["prev_batters"] / n_prev, prev_spots=r["prev_spots"] / n_prev,
                           prev_all=r["prev_all"] / n_prev)
    return pd.DataFrame(out).T.rename_axis("season")


# ------------------------------------------------------------------------------- live (upcoming)

def project_live(games: pd.DataFrame, lineups: pd.DataFrame, players: pd.DataFrame,
                 upcoming: list[dict], sp: str = "probable") -> pd.DataFrame:
    """Lineups of upcoming games (``statsapi.probables`` rows: one per team-game, with
    ``probable_sp_id`` and the posted ``lineup``) from the finished ``games`` before them: the
    probable starter (else the rotation's), and the posted lineup when the team has posted nine
    (``source == "posted"``, positions from the players' primary positions), else the projection
    from the previous lineups. Games without both teams are skipped."""
    up = pd.DataFrame(upcoming)
    if up.empty:
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for pk, d in up.groupby("game_pk", sort=False):
        d = d.set_index("side")
        if not {"home", "away"} <= set(d.index):
            continue
        h, a = d.loc["home"], d.loc["away"]
        season = int(str(h.date)[:4])
        rows.append(dict(game_pk=int(pk), season=season, game_type=h.game_type, date=str(h.date),
                         start=f"{h.date}T23:59:00Z", doubleheader=None, game_number=1, state=h.state,
                         final=False, home_id=int(h.team_id), away_id=int(a.team_id),
                         home_probable_sp=h.probable_sp_id, away_probable_sp=a.probable_sp_id,
                         home_sp=None, away_sp=None))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    new = pd.DataFrame(rows)
    allg = pd.concat([games, new], ignore_index=True)
    proj = projected_lineups(allg, lineups, players, lineup="previous", sp=sp,
                             seasons=set(new.season))
    proj = proj[proj.game_pk.isin(set(new.game_pk))].copy()
    posted = {(int(r["game_pk"]), r["side"]): [int(p) for p in r.get("lineup") or []] for r in upcoming}
    position = dict(zip(players.player_id, players.position.map(_position_code)))
    bats = dict(zip(players.player_id, players.bats))
    throws = {int(k): v for k, v in zip(players.player_id, players.throws.map(throw_hand))}
    out = []
    for (pk, side), d in proj.groupby(["game_pk", "side"], sort=False):
        ids = posted.get((int(pk), side), [])
        if len(ids) != SPOTS:
            out.append(d)
            continue
        sp_row = d[d.role == "SP"].iloc[0].to_dict()
        base = {k: sp_row[k] for k in COLUMNS if k not in ("role", "spot", "position", "player_id", "hand",
                                                           "source", "ref_game_pk")}
        sp_id = sp_row["player_id"]
        bat = []
        for spot, pid in enumerate(ids, start=1):
            pos = PITCHER if pd.notna(sp_id) and pid == sp_id and not base["dh"] else position.get(pid)
            bat.append(base | dict(role="B", spot=spot, position=pos, player_id=pid,
                                   hand=throws.get(pid) if pos == PITCHER else bats.get(pid),
                                   source="posted", ref_game_pk=int(pk)))
        sp_spot = next((b["spot"] for b in bat if b["position"] == PITCHER), 0)
        sp_row.update(spot=sp_spot, source="posted", ref_game_pk=int(pk))
        out.append(pd.DataFrame([sp_row] + bat, columns=COLUMNS))
    return _types(pd.concat(out, ignore_index=True))


# primaryPosition abbreviation -> fielding code
POSITION_CODES = {"P": "1", "C": "2", "1B": "3", "2B": "4", "3B": "5", "SS": "6", "LF": "7", "CF": "8",
                  "RF": "9", "DH": "10", "TWP": "10", "OF": "8"}


def _position_code(abbr) -> str | None:
    return POSITION_CODES.get(abbr) if isinstance(abbr, str) else None


def live_lineups(processed: Path, dates: list[str], sp: str = "probable", client=None) -> pd.DataFrame:
    """:func:`project_live` for the not-yet-final games on ``dates`` (``YYYY-MM-DD``), from the
    Stats API's probables and posted lineups as listed now."""
    from . import statsapi
    data = load(processed)
    upcoming = [r for day in dates for r in statsapi.probables(day, client)
                if r["state"] not in statsapi.NOT_PLAYED and r["state"] not in ("Final", "Game Over",
                                                                                  "Completed Early")]
    return project_live(data["games"], data["lineups"], data["players"], upcoming, sp=sp)


# ------------------------------------------------------------------------------------------ build

def build_lineups(processed: Path, seasons: list[int] | None = None, sp: str = "probable",
                  live_dates: list[str] | None = None) -> dict:
    """Write ``projected_lineups.parquet`` (previous lineup vs hand) and ``posted_lineups.parquet``
    (the posted stand-in) for every game in ``processed`` (or ``seasons``; earlier seasons there
    are still read as history), and with ``live_dates`` ``projected_lineups_upcoming.parquet``.
    Returns :func:`evaluate`'s summary per season."""
    processed = Path(processed)
    data = load(processed)
    want = None if seasons is None else set(seasons)
    proj = projected_lineups(data["games"], data["lineups"], data["players"], sp=sp, seasons=want)
    proj.to_parquet(processed / "projected_lineups.parquet", index=False)
    posted = projected_lineups(data["games"], data["lineups"], data["players"], lineup="posted", sp=sp,
                               seasons=want)
    posted.to_parquet(processed / "posted_lineups.parquet", index=False)
    ev = evaluate(proj, data["games"], data["lineups"])
    log.info("projected lineups: %d rows, %d team-games\n%s", len(proj),
             proj[["game_pk", "side"]].drop_duplicates().shape[0], ev.round(4).to_string())
    if live_dates:
        up = live_lineups(processed, live_dates, sp=sp)
        up.to_parquet(processed / "projected_lineups_upcoming.parquet", index=False)
        log.info("upcoming: %d games", up.game_pk.nunique())
    return {int(s): {k: round(float(v), 4) for k, v in r.items()} for s, r in ev.iterrows()}
