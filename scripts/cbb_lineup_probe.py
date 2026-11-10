"""C1 probe: is there on-floor lineup data for college basketball?

Two sources (docs/multisport/PLAN.md, work package C1):

``espn``  Per season and gender, how many games in sportsdataverse's ESPN play-by-play
          have substitution events, and how many have enough of them (both teams, at least
          ``--min-subs`` events each) to rebuild the five on the floor. One parquet file per
          season is downloaded, summarized and deleted. Writes ``espn_seasons.csv`` and
          ``espn_games.csv`` (per-game flags, used by ``cbbd``).

``cbbd``  CollegeBasketballData (``CBBD_API_KEY``, men's only). For sampled dates, every game
          of the day (``/games``) against its plays (``/plays/date``), each of which carries the
          ten players CBBD says are on the floor (``onFloor``). A game counts as having lineups
          when at least 90% of its plays list exactly five per team, each team uses at least
          three distinct fives (a real rotation, not starters copied all game), and the player
          credited with a play (other than a substitution) is one of the ten on at least 95% of
          plays. Joined to
          ``espn_games.csv`` by ESPN game id, it shows whether CBBD covers games ESPN has no
          substitutions for. Writes ``cbbd_games.csv``. Uses two API calls per date.

Outputs go to ``--out`` (never commit them). Usage::

    python scripts/cbb_lineup_probe.py espn --out OUT --first 2006 --last 2026
    python scripts/cbb_lineup_probe.py cbbd --out OUT --first 2010 --last 2027
    python scripts/cbb_lineup_probe.py report --out OUT
"""

from __future__ import annotations

import argparse
import datetime
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PBP = {
    "mens": RELEASES + "/espn_mens_college_basketball_pbp/play_by_play_{season}.parquet",
    "womens": RELEASES + "/espn_womens_college_basketball_pbp/play_by_play_{season}.parquet",
}
CBBD = "https://api.collegebasketballdata.com"
COLUMNS = ["game_id", "type_text", "text", "team_id", "athlete_id_1", "athlete_id_2",
           "home_team_id", "away_team_id", "game_date"]


# --------------------------------------------------------------------------- ESPN


def espn_games(frame: pd.DataFrame, min_subs: int) -> pd.DataFrame:
    """One row per game: substitution events per side and whether they can rebuild lineups."""
    games = frame.groupby("game_id").agg(
        home_team_id=("home_team_id", "first"),
        away_team_id=("away_team_id", "first"),
        game_date=("game_date", "first"),
        plays=("type_text", "size"),
    )
    subs = frame[frame["type_text"].eq("Substitution")]
    side = subs["team_id"].eq(subs["home_team_id"]).map({True: "home_subs", False: "away_subs"})
    counts = subs.groupby(["game_id", side]).size().unstack(fill_value=0)
    games = games.join(counts).fillna({"home_subs": 0, "away_subs": 0})
    for column in ("home_subs", "away_subs"):
        if column not in games:
            games[column] = 0
    games["two_ids"] = subs.groupby("game_id")["athlete_id_2"].apply(lambda s: s.notna().mean())
    games["has_subs"] = (games["home_subs"] + games["away_subs"]) > 0
    games["rebuildable"] = games[["home_subs", "away_subs"]].min(axis=1) >= min_subs
    return games.reset_index()


def run_espn(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seasons, per_game = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for gender in args.genders:
            for season in range(args.first, args.last + 1):
                path = Path(tmp) / f"{gender}_{season}.parquet"
                response = requests.get(PBP[gender].format(season=season), timeout=600)
                if response.status_code != 200:
                    print(f"{gender} {season}: HTTP {response.status_code}, skipped")
                    continue
                path.write_bytes(response.content)
                frame = pd.read_parquet(path, columns=COLUMNS)
                path.unlink()
                games = espn_games(frame, args.min_subs)
                games["gender"], games["season"] = gender, season
                per_game.append(games)
                row = {
                    "gender": gender,
                    "season": season,
                    "games": len(games),
                    "with_subs": int(games["has_subs"].sum()),
                    "rebuildable": int(games["rebuildable"].sum()),
                    "subs_per_game": float(
                        (games["home_subs"] + games["away_subs"])[games["has_subs"]].mean()
                    ),
                    "sub_events_two_ids": float(games.loc[games["has_subs"], "two_ids"].mean()),
                }
                row["with_subs_pct"] = 100 * row["with_subs"] / row["games"]
                row["rebuildable_pct"] = 100 * row["rebuildable"] / row["games"]
                seasons.append(row)
                print(row, flush=True)
    pd.DataFrame(seasons).to_csv(out / "espn_seasons.csv", index=False)
    pd.concat(per_game).to_csv(out / "espn_games.csv", index=False)


# --------------------------------------------------------------------------- CBBD


def sample_dates(season: int) -> list[datetime.date]:
    """Saturdays near December 7 and February 10 of a season (ending year), or the latest
    Saturday before today for a season in progress."""
    def saturday(day: datetime.date) -> datetime.date:
        return day - datetime.timedelta(days=(day.weekday() - 5) % 7)

    today = datetime.date.today()
    dates = [saturday(datetime.date(season - 1, 12, 7)), saturday(datetime.date(season, 2, 10))]
    done = [d for d in dates if d < today]
    if not done and datetime.date(season - 1, 11, 4) < today:
        done = [saturday(today - datetime.timedelta(days=1))]
    return done


def _get(path: str, key: str, **params) -> tuple[list[dict], str | None]:
    """One CBBD request; a day of plays is ~80 MB, so a transfer cut off midway is retried."""
    for attempt in range(4):
        try:
            response = requests.get(
                f"{CBBD}/{path}", headers={"Authorization": f"Bearer {key}"}, params=params,
                timeout=600,
            )
            response.raise_for_status()
            return response.json(), response.headers.get("x-calllimit-remaining")
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _markdown(frame: pd.DataFrame) -> str:
    """A pipe table (no ``tabulate`` dependency)."""
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def cbbd_game(plays: list[dict]) -> dict:
    """Lineup quality of one game's CBBD plays."""
    five, credited, with_participant, subs = 0, 0, 0, 0
    fives = defaultdict(set)
    for play in plays:
        on = play.get("onFloor") or []
        teams = defaultdict(list)
        for player in on:
            teams[player.get("team")].append(player["id"])
        if len(teams) == 2 and all(len(ids) == 5 for ids in teams.values()):
            five += 1
            for team, ids in teams.items():
                fives[team].add(frozenset(ids))
        if play.get("playType") == "Substitution":
            subs += 1
            continue  # the player is coming on or going off
        participants = play.get("participants") or []
        if participants:
            with_participant += 1
            if participants[0]["id"] in {p["id"] for p in on}:
                credited += 1
    n = len(plays)
    distinct = [len(v) for v in fives.values()]
    return {
        "plays": n,
        "subs": subs,
        "five_share": five / n if n else 0.0,
        "min_distinct_fives": min(distinct) if len(distinct) == 2 else 0,
        "credited_on_floor": credited / with_participant if with_participant else 0.0,
    }


def run_cbbd(args) -> None:
    key = os.environ.get("CBBD_API_KEY")
    if not key:
        raise SystemExit("CBBD_API_KEY is not set")
    out = Path(args.out)
    rows = []
    for season in range(args.first, args.last + 1):
        for day in sample_dates(season):
            start = datetime.datetime.combine(day, datetime.time(0), datetime.timezone.utc)
            games, _ = _get(
                "games", key,
                startDateRange=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                endDateRange=(start + datetime.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            plays, remaining = _get("plays/date", key, date=day.isoformat())
            by_game = defaultdict(list)
            for play in plays:
                by_game[play["gameId"]].append(play)
            for game in games:
                if str(game.get("status", "")).lower() != "final":
                    continue
                row = {
                    "season": season,
                    "date": day.isoformat(),
                    "cbbd_game_id": game["id"],
                    "game_id": game.get("sourceId"),
                    "home": game.get("homeTeam"),
                    "away": game.get("awayTeam"),
                    "home_conference": game.get("homeConference"),
                    "away_conference": game.get("awayConference"),
                } | cbbd_game(by_game.get(game["id"], []))
                rows.append(row)
            frame = pd.DataFrame([r for r in rows if r["date"] == day.isoformat()])
            print(f"{season} {day}: {len(frame)} final games, "
                  f"{(frame['plays'] > 0).sum()} with plays, calls left {remaining}", flush=True)
    games = pd.DataFrame(rows)
    games["has_lineups"] = (
        (games["five_share"] >= 0.9)
        & (games["min_distinct_fives"] >= 3)
        & (games["credited_on_floor"] >= 0.95)
    )
    games.to_csv(out / "cbbd_games.csv", index=False)


# --------------------------------------------------------------------------- report


def run_report(args) -> None:
    out = Path(args.out)
    seasons = pd.read_csv(out / "espn_seasons.csv")
    for gender, frame in seasons.groupby("gender"):
        print(f"\nESPN {gender}")
        print(_markdown(frame[["season", "games", "with_subs_pct", "rebuildable_pct",
                               "subs_per_game", "sub_events_two_ids"]].round(1)))
    path = out / "cbbd_games.csv"
    if not path.exists():
        return
    cbbd = pd.read_csv(path, dtype={"game_id": "string"})
    espn = pd.read_csv(out / "espn_games.csv", dtype={"game_id": "string"})
    espn = espn[espn["gender"].eq("mens")][["game_id", "has_subs", "rebuildable"]]
    cbbd = cbbd.merge(espn, on="game_id", how="left")
    cbbd["espn_rebuildable"] = cbbd["rebuildable"].astype("boolean").fillna(False)
    cbbd["either"] = cbbd["has_lineups"] | cbbd["espn_rebuildable"]
    cbbd["d1_both"] = cbbd["home_conference"].notna() & cbbd["away_conference"].notna()
    table = cbbd.groupby("season").agg(
        games=("cbbd_game_id", "size"),
        with_plays=("plays", lambda s: 100 * (s > 0).mean()),
        cbbd_lineups=("has_lineups", lambda s: 100 * s.mean()),
        espn_rebuildable=("espn_rebuildable", lambda s: 100 * s.mean()),
        either=("either", lambda s: 100 * s.mean()),
        five_share=("five_share", "median"),
        credited=("credited_on_floor", "median"),
    )
    print("\nCBBD (sampled dates, final games; % of games)")
    print(_markdown(table.round(3).reset_index()))
    lacking = cbbd[~cbbd["espn_rebuildable"]]
    print(f"\nCBBD lineups on games ESPN can't rebuild: {lacking['has_lineups'].mean():.1%} "
          f"of {len(lacking)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    espn = sub.add_parser("espn")
    espn.add_argument("--out", required=True)
    espn.add_argument("--first", type=int, default=2006)
    espn.add_argument("--last", type=int, default=2026)
    espn.add_argument("--genders", nargs="+", default=["mens", "womens"])
    espn.add_argument("--min-subs", type=int, default=10)
    cbbd = sub.add_parser("cbbd")
    cbbd.add_argument("--out", required=True)
    cbbd.add_argument("--first", type=int, default=2010)
    cbbd.add_argument("--last", type=int, default=2027)
    report = sub.add_parser("report")
    report.add_argument("--out", required=True)
    args = parser.parse_args()
    {"espn": run_espn, "cbbd": run_cbbd, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
