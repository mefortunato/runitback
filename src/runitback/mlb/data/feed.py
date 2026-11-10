"""Parse one MLB Stats API live feed into plate appearances, lineups, substitutions and starters.

The live feed lists every play (``liveData.plays.allPlays``): a plate appearance, or a base-running
play that ended a half inning with the batter still at the plate (caught stealing, pickoff). Each
play carries its pitches and actions (``playEvents``, substitutions included) and its runner
movements (``runners``), each tied to the event it happened on (``playIndex``).

**Base / out state is replayed, not read.** Starting from empty bases and no outs each half
inning (plus the automatic runner on second base in extra innings, regular season 2020+), the
events and runner movements are applied in order. That gives three states per play:

- ``start``: when the plate appearance began;
- ``pre``: just before the event that ended it (after any steals, wild pitches or pickoffs
  during it);
- ``end``: after it (``outs == 3`` when it ended the half inning).

The replay is checked against what the feed states, and the counts of disagreements are returned
in ``checks``: every movement's starting base must hold that runner (``base_mismatch``), the
replayed bases after each play must match the feed's ``postOnFirst/Second/Third``
(``post_mismatch``), outs must match ``count.outs`` (``outs_mismatch``) and runs must match the
score change (``runs_mismatch``). On real feeds all four are zero or nearly so (tests/mlb).

**Defense.** Each team's fielders by position (``fielder_2`` .. ``fielder_9``, Statcast's
numbering) start from the box score's starting positions and follow the defensive substitutions
and switches; the pitcher is the play's pitcher. Every fielding credit (putout, assist, error) is
checked against it (``fielder_checks`` / ``fielder_mismatch``).
"""
from __future__ import annotations

from collections import defaultdict

BASES = ("1B", "2B", "3B")
# Statcast fielder numbering: 2 catcher .. 9 right field.
FIELD_POSITIONS = tuple(str(p) for p in range(2, 10))

# Plays that are plate appearances: the Stats API's eventTypes with plateAppearance=true, plus
# grounded_into_triple_play (listed there as false, but a batter's own result).
PA_EVENTS = {
    "single", "double", "triple", "home_run", "double_play", "field_error", "field_out",
    "fielders_choice", "fielders_choice_out", "force_out", "grounded_into_double_play",
    "grounded_into_triple_play", "strikeout", "strike_out", "strikeout_double_play",
    "strikeout_triple_play", "triple_play", "sac_fly", "catcher_interf", "batter_interference",
    "fan_interference", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play", "walk",
    "intent_walk", "hit_by_pitch",
}

# eventType -> coarse result (the outcome classes a PA-level simulator draws from).
RESULTS = {
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "walk": "BB", "intent_walk": "IBB", "hit_by_pitch": "HBP",
    "strikeout": "K", "strike_out": "K", "strikeout_double_play": "K", "strikeout_triple_play": "K",
    "field_error": "ROE", "catcher_interf": "INT", "fan_interference": "INT",
    "sac_bunt": "SH", "sac_bunt_double_play": "SH", "sac_fly": "SF", "sac_fly_double_play": "SF",
}  # everything else in PA_EVENTS is an out (or fielder's choice) on a ball in play: "OUT"

# Substitution events (playEvents with isSubstitution) and what they change.
SUB_EVENTS = {"offensive_substitution", "defensive_substitution", "defensive_switch",
              "pitching_substitution"}
PINCH_RUNNER = "12"


def result_class(event_type: str) -> str:
    return RESULTS.get(event_type, "OUT")


def bases_code(bases: dict) -> int:
    """Occupied bases as a bit mask: 1 first, 2 second, 4 third (0-7)."""
    return sum(1 << i for i, b in enumerate(BASES) if bases.get(b) is not None)


def ghost_runner_rule(season: int, game_type: str) -> bool:
    """Extra innings start with a runner on second: regular season, 2020 onward."""
    return game_type == "R" and season >= 2020


def _spot(code) -> int | None:
    """Box score ``battingOrder`` code ``"601"`` (6th spot, first substitute) -> 6."""
    try:
        return int(code) // 100
    except (TypeError, ValueError):
        return None


def _id(obj) -> int | None:
    return (obj or {}).get("id")


class _Half:
    """Replayed state of one half inning."""

    def __init__(self):
        self.bases: dict[str, int | None] = {b: None for b in BASES}
        self.outs = 0

    def snapshot(self) -> dict:
        return {"bases": dict(self.bases), "outs": self.outs}


def _state_cols(prefix: str, state: dict) -> dict:
    bases = state["bases"]
    return {f"{prefix}_bases": bases_code(bases), f"{prefix}_outs": state["outs"],
            f"{prefix}_on_1b": bases["1B"], f"{prefix}_on_2b": bases["2B"],
            f"{prefix}_on_3b": bases["3B"]}


def parse_feed(feed: dict) -> dict:
    """One game's feed -> ``{"game": dict, "plays": [dict], "lineups": [dict],
    "substitutions": [dict], "players": [dict], "checks": dict}``.

    ``plays`` has every play (``is_pa`` marks plate appearances) so that runs scored on
    base-running plays still count toward run expectancy. Ids are MLB person / team ids.
    """
    gd = feed.get("gameData") or {}
    live = feed.get("liveData") or {}
    game_info = gd.get("game") or {}
    game_pk = int(feed.get("gamePk") or game_info.get("pk"))
    season = int(game_info.get("season"))
    game_type = game_info.get("type")
    box = (live.get("boxscore") or {}).get("teams") or {}
    linescore = live.get("linescore") or {}
    scheduled = int(linescore.get("scheduledInnings") or 9)
    teams = {side: _id((gd.get("teams") or {}).get(side)) for side in ("home", "away")}
    ghost_rule = ghost_runner_rule(season, game_type)

    # players: side, batting order code, starting position
    side_of: dict[int, str] = {}
    order_of: dict[int, str] = {}
    lineups = []
    for side in ("home", "away"):
        team_box = box.get(side) or {}
        pitchers = team_box.get("pitchers") or []
        for key, p in (team_box.get("players") or {}).items():
            pid = _id(p.get("person"))
            if pid is None:
                continue
            side_of[pid] = side
            order = p.get("battingOrder")
            if order:
                order_of[pid] = order
            positions = [a.get("code") for a in p.get("allPositions") or []]
            if not order and pid not in pitchers:
                continue  # did not play
            stats = p.get("stats") or {}
            lineups.append({
                "game_pk": game_pk, "side": side, "team_id": teams[side], "player_id": pid,
                "batting_order": int(order) if order else None, "spot": _spot(order),
                "starter": bool(order) and int(order) % 100 == 0,
                "start_position": positions[0] if positions else None,
                "positions": ",".join(c for c in positions if c),
                "is_sp": bool(pitchers) and pitchers[0] == pid,
                "pitcher_order": pitchers.index(pid) if pid in pitchers else None,
                "box_pa": (stats.get("batting") or {}).get("plateAppearances"),
                "box_bf": (stats.get("pitching") or {}).get("battersFaced"),
            })

    # defense: position -> player, per side, from the starters' first positions
    fielders: dict[str, dict[str, int | None]] = {s: {p: None for p in FIELD_POSITIONS}
                                                  for s in ("home", "away")}
    for row in lineups:
        if row["starter"] and row["start_position"] in FIELD_POSITIONS:
            fielders[row["side"]][row["start_position"]] = row["player_id"]

    probable = gd.get("probablePitchers") or {}
    starting = {s: ((box.get(s) or {}).get("pitchers") or [None])[0] for s in ("home", "away")}
    status = gd.get("status") or {}
    venue = gd.get("venue") or {}
    dt = gd.get("datetime") or {}

    checks = defaultdict(int)
    plays, subs = [], []
    half: _Half | None = None
    half_key = None
    prev_score = {"home": 0, "away": 0}

    def move(state: _Half, group: list[dict], counts: dict) -> int:
        """Apply one event's runner movements at once; returns runs scored."""
        per_runner: dict[int, list[dict]] = {}
        for r in group:
            rid = _id((r.get("details") or {}).get("runner"))
            per_runner.setdefault(rid, []).append(r)
        # every runner's first start base must hold him before the event
        for rid, moves in per_runner.items():
            start = (moves[0].get("movement") or {}).get("start")
            if start in BASES and state.bases.get(start) != rid:
                counts["base_mismatch"] += 1
                for b in BASES:  # repair: trust the movement
                    if state.bases[b] == rid:
                        state.bases[b] = None
        for rid, moves in per_runner.items():  # everyone leaves at once, then everyone arrives
            start = (moves[0].get("movement") or {}).get("start")
            if start in BASES:
                state.bases[start] = None
        runs = 0
        for rid, moves in per_runner.items():
            last = moves[-1].get("movement") or {}
            if any((m.get("movement") or {}).get("isOut") for m in moves):
                state.outs += 1
            elif last.get("end") == "score":
                runs += 1
            elif last.get("end") in BASES:
                state.bases[last["end"]] = rid
        return runs

    for play in (live.get("plays") or {}).get("allPlays") or []:
        about = play.get("about") or {}
        result = play.get("result") or {}
        matchup = play.get("matchup") or {}
        inning = int(about.get("inning") or 0)
        top = bool(about.get("isTopInning"))
        bat_side, fld_side = ("away", "home") if top else ("home", "away")
        batter, pitcher = _id(matchup.get("batter")), _id(matchup.get("pitcher"))
        events = play.get("playEvents") or []
        runners = play.get("runners") or []

        new_half = (inning, top) != half_key
        if new_half:
            half_key, half = (inning, top), _Half()
            if ghost_rule and inning > scheduled:
                # the player batting just before the leadoff hitter's spot
                spot = _spot(order_of.get(batter))
                if spot is not None:
                    want = (spot - 2) % 9 + 1
                    lineup_now = _current_lineup(lineups, bat_side, plays, subs, want)
                    half.bases["2B"] = lineup_now
                    checks["ghost_runners"] += 1
        start = half.snapshot()

        # batter's result event: his own first movement from the plate
        batter_moves = [r for r in runners if _id((r.get("details") or {}).get("runner")) == batter
                        and (r.get("movement") or {}).get("start") is None]
        indices = [e.get("index", i) for i, e in enumerate(events)]
        result_index = min((r.get("details") or {}).get("playIndex", indices[-1] if indices else 0)
                           for r in batter_moves) if batter_moves else (indices[-1] if indices else 0)
        groups: dict[int, list[dict]] = defaultdict(list)
        for r in runners:
            groups[(r.get("details") or {}).get("playIndex", result_index)].append(r)

        pre = None
        fielders_at_result = None
        runs_pre = runs_result = 0
        by_index = {e.get("index", i): e for i, e in enumerate(events)}
        for idx in sorted(set(by_index) | set(groups)):
            e = by_index.get(idx)
            if e is not None and e.get("isSubstitution"):
                etype = (e.get("details") or {}).get("eventType")
                pid, rep = _id(e.get("player")), _id(e.get("replacedPlayer"))
                pos = (e.get("position") or {}).get("code")
                side = side_of.get(pid, fld_side if etype != "offensive_substitution" else bat_side)
                subs.append({
                    "game_pk": game_pk, "at_bat_index": about.get("atBatIndex"), "event_index": idx,
                    "inning": inning, "half": "top" if top else "bottom", "outs": half.outs,
                    "side": side, "team_id": teams.get(side), "event_type": etype,
                    "player_id": pid, "replaced_id": rep, "position": pos,
                    "batting_order": int(e["battingOrder"]) if e.get("battingOrder") else None,
                    "description": (e.get("details") or {}).get("description"),
                })
                if pos == PINCH_RUNNER and rep is not None:
                    for b in BASES:
                        if half.bases[b] == rep:
                            half.bases[b] = pid
                if side in fielders and pid is not None:
                    field = fielders[side]
                    if pos in FIELD_POSITIONS or pos in ("1", "10"):
                        for p in FIELD_POSITIONS:  # leaves his old position
                            if field[p] == pid:
                                field[p] = None
                    if pos in FIELD_POSITIONS:
                        field[pos] = pid
            if idx == result_index and pre is None:
                pre = half.snapshot()
                fielders_at_result = dict(fielders[fld_side])
            if idx in groups:
                scored = move(half, groups[idx], checks)
                if idx == result_index:
                    runs_result += scored
                elif pre is None:
                    runs_pre += scored
                else:
                    runs_result += scored
        if pre is None:
            pre = half.snapshot()
            fielders_at_result = dict(fielders[fld_side])

        # checks against what the feed states
        end_outs = int((play.get("count") or {}).get("outs") or 0)
        if min(half.outs, 3) != end_outs:
            checks["outs_mismatch"] += 1
            half.outs = end_outs
        if half.outs < 3:
            post = {"1B": _id(matchup.get("postOnFirst")), "2B": _id(matchup.get("postOnSecond")),
                    "3B": _id(matchup.get("postOnThird"))}
            if post != half.bases:
                checks["post_mismatch"] += 1
                half.bases = post  # trust the feed going forward
        else:
            half.bases = {b: None for b in BASES}
        score = {"home": result.get("homeScore"), "away": result.get("awayScore")}
        if score["home"] is not None and score["away"] is not None:
            delta = (score[bat_side] - prev_score[bat_side]) + (score[fld_side] - prev_score[fld_side])
            if delta != runs_pre + runs_result:
                checks["runs_mismatch"] += 1
            score_before = dict(prev_score)
            prev_score = score
        else:
            score_before = dict(prev_score)
        end = half.snapshot()

        # fielding credits vs replayed defense
        for r in runners:
            for c in r.get("credits") or []:
                pos = (c.get("position") or {}).get("code")
                if pos in FIELD_POSITIONS:
                    checks["fielder_checks"] += 1
                    if fielders[fld_side].get(pos) != _id(c.get("player")):
                        checks["fielder_mismatch"] += 1

        event_type = result.get("eventType")
        is_pa = event_type in PA_EVENTS and bool(about.get("isComplete", True))
        pitches = sum(1 for e in events if e.get("isPitch"))
        row = {
            "game_pk": game_pk, "season": season, "game_type": game_type,
            "at_bat_index": about.get("atBatIndex"), "inning": inning,
            "half": "top" if top else "bottom", "bat_team_id": teams[bat_side],
            "fld_team_id": teams[fld_side], "bat_home": int(not top),
            "batter_id": batter, "bat_side": (matchup.get("batSide") or {}).get("code"),
            "spot": _spot(order_of.get(batter)),
            "pitcher_id": pitcher, "pitch_hand": (matchup.get("pitchHand") or {}).get("code"),
            "is_pa": is_pa, "event_type": event_type,
            "result": result_class(event_type) if is_pa else None,
            "balls": (play.get("count") or {}).get("balls"),
            "strikes": (play.get("count") or {}).get("strikes"), "pitches": pitches,
            "rbi": result.get("rbi"),
            "bat_score": score_before[bat_side], "fld_score": score_before[fld_side],
            **_state_cols("start", start), **_state_cols("pre", pre), **_state_cols("end", end),
            "runs_pre": runs_pre, "runs": runs_result,
            **{f"fielder_{p}": fielders_at_result.get(p) for p in FIELD_POSITIONS},
        }
        plays.append(row)

    # runs from each play's start to the end of its half inning, for run expectancy
    _runs_to_end_of_half(plays)

    players = []
    for key, p in (gd.get("players") or {}).items():
        players.append({
            "player_id": p.get("id"), "name": p.get("fullName"),
            "bats": (p.get("batSide") or {}).get("code"), "throws": (p.get("pitchHand") or {}).get("code"),
            "position": (p.get("primaryPosition") or {}).get("abbreviation"),
            "birth_date": p.get("birthDate"),
        })

    final_score = {s: ((linescore.get("teams") or {}).get(s) or {}).get("runs") for s in ("home", "away")}
    game = {
        "game_pk": game_pk, "season": season, "game_type": game_type,
        "date": dt.get("officialDate"), "start": dt.get("dateTime"),
        "doubleheader": game_info.get("doubleHeader"), "game_number": game_info.get("gameNumber"),
        "state": status.get("detailedState"), "final": status.get("abstractGameState") == "Final",
        "venue_id": venue.get("id"), "scheduled_innings": scheduled,
        "innings": linescore.get("currentInning"),
        "home_id": teams["home"], "away_id": teams["away"],
        "home_runs": final_score["home"], "away_runs": final_score["away"],
        "home_probable_sp": _id(probable.get("home")), "away_probable_sp": _id(probable.get("away")),
        "home_sp": starting["home"], "away_sp": starting["away"],
        "pa": sum(p["is_pa"] for p in plays),
    }
    checks["plays"] = len(plays)
    checks["pa"] = game["pa"]
    # PA count vs the box score's plate appearances
    box_pa = sum((r["box_pa"] or 0) for r in lineups)
    checks["box_pa_mismatch"] = abs(box_pa - game["pa"]) if box_pa else 0
    return {"game": game, "plays": plays, "lineups": lineups, "substitutions": subs,
            "players": players, "checks": dict(checks)}


def _current_lineup(lineups: list[dict], side: str, plays: list[dict], subs: list[dict],
                    spot: int) -> int | None:
    """Who holds batting ``spot`` for ``side`` now: the last substitute into it so far, else the
    starter."""
    holder = next((r["player_id"] for r in lineups
                   if r["side"] == side and r["spot"] == spot and r["starter"]), None)
    for s in subs:
        if s["side"] == side and s["batting_order"] is not None and s["batting_order"] // 100 == spot:
            holder = s["player_id"]
    return holder


def _runs_to_end_of_half(plays: list[dict]) -> None:
    """Adds ``runs_rest`` (runs from the play's start through the end of its half inning) and
    ``half_complete`` (the half inning ended with three outs, not a walk-off or a stopped game)."""
    i = len(plays)
    while i > 0:
        j = i - 1
        key = (plays[j]["game_pk"], plays[j]["inning"], plays[j]["half"])
        k = j
        while k > 0 and (plays[k - 1]["game_pk"], plays[k - 1]["inning"], plays[k - 1]["half"]) == key:
            k -= 1
        complete = plays[j]["end_outs"] >= 3
        total = 0
        for m in range(j, k - 1, -1):
            total += plays[m]["runs_pre"] + plays[m]["runs"]
            plays[m]["runs_rest"] = total
            plays[m]["half_complete"] = complete
        i = k
