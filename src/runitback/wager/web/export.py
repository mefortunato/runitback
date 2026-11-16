"""``runitback wager web export``: every page's data as static JSON files.

The web app is a static site with no server: each page's data is
precomputed here from the snapshots ``runitback wager web build`` writes, and the
challenge's random pick and scoring run in the browser. ``refresh=True``
re-downloads results first, so a daily run updates team state, upcoming
predictions and rankings without retraining.

Writes, under ``<output_dir>/data/``:

* ``sports.json``: each sport's info (label, model, backtest summary);
* ``<sport>/upcoming.json``: ``{"default": ..., "by_date": {"<season>-<date>": ...}}``;
* ``<sport>/games.json``: ``{game_id: game view}`` for every upcoming game;
* ``<sport>/lineups/<game_id>.json``: an upcoming game's lineup editor (both
  projected lineups and the model's points per point of lineup value,
  :mod:`runitback.wager.web.lineups`), for a model with player values;
* ``<sport>/recent.json``: ``{"dates": [{season, date, name}], "by_date":
  {"<season>-<date>": [game]}}``, the last played dates' games with their
  pre-game predictions and results (:meth:`Snapshot.recent`), and
  ``<sport>/recent_games.json``, those games' views (like ``games.json``);
* ``<sport>/challenge/<n>.json``: ``[{"game": ..., "answers": {market: ...}}]``
  per pool game playable on some market, each answer without the
  pick-dependent ``pick`` / ``correct`` / ``units``, in shards listed by
  ``<sport>/challenge/index.json`` (see :func:`write_challenge`);
* ``<sport>/players.json`` (with ``player_values``): this season's players ranked by value within
  each position group (:mod:`runitback.wager.web.players`);
* ``<sport>/performance.json`` and ``<sport>/rankings.json``;
* ``<sport>/map.json``: the latest season's teams in 3-D before each
  ranking date (:meth:`Snapshot.team_map_view`; ``null`` for a model
  without team embeddings);
* ``<sport>/odds.json`` (with ``odds``): current sportsbook lines for the
  upcoming games (:mod:`runitback.wager.web.odds`);
* ``challenge-keys.json``: ``{sport: key}``, what each sport's challenge
  files were exported from (:func:`challenge_key`), so the next export can
  reuse them instead of exporting them again (``challenge_from``).

Team objects are written once per file (``"teams"``) and referenced as
``{"$team": key}``; floats are rounded to 6 decimals.
"""

from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import json
import logging
import pathlib
import shutil
import urllib.request

import pandas as pd

from ..core.process import process
from ..core.sport import get_sport
from ..odds.providers import provider_for
from .odds import load_previous, refresh_odds
from .players import player_ranks
from .snapshot import Snapshot
from .views import BUILTIN_VIEWS

logger = logging.getLogger(__name__)


def download_games(sport: str) -> pd.DataFrame:
    """The sport's games, freshly downloaded and processed."""
    return process(sport, get_sport(sport).download())


def load_snapshots(
    data_dir: str | pathlib.Path,
    refresh: bool = False,
    sports: list[str] | None = None,
    require_refresh: bool = True,
) -> dict[str, Snapshot]:
    """One snapshot per sport subdirectory of ``data_dir`` (only ``sports``
    when given).

    With ``refresh``, each sport's games are downloaded first and the
    snapshot is built once from them (building one runs the model over the
    season, so loading the built games and then refreshing would do that
    work twice). A sport whose download fails (e.g. its API's rate limit)
    keeps its built games, so one source cannot hold back the other
    sports; with ``require_refresh``, loading fails when every download
    does (an export of a single sport turns it off, and :func:`merge`
    checks the sports together).
    """
    snapshots, errors = {}, []
    for path in sorted(pathlib.Path(data_dir).iterdir()):
        if not (path / "meta.json").exists():
            continue
        sport = json.loads((path / "meta.json").read_text())["sport"]
        if sports is not None and sport not in sports:
            continue
        games = None
        if refresh:
            try:
                games = download_games(sport)
            except Exception as error:
                # a GitHub Actions annotation, plain text elsewhere
                print(
                    f"::warning::{sport} refresh failed, keeping the built data: "
                    f"{error}"
                )
                errors.append(error)
        snapshot = Snapshot.load(path, games=games)
        snapshots[snapshot.view.sport] = snapshot
        logger.info("loaded %s from %s", snapshot.view.sport, path)
    if require_refresh and errors and len(errors) == len(snapshots):
        raise errors[0]
    return dict(sorted(snapshots.items(), key=lambda kv: _nav_order(kv[0])))


def _nav_order(sport: str) -> int:
    """The site's nav order: built-in sports as listed, any others after."""
    order = {sport: i for i, sport in enumerate(BUILTIN_VIEWS)}
    return order.get(sport, len(order))


def _compact(value, teams: dict):
    if isinstance(value, dict):
        key = value.get("team")
        if isinstance(key, str) and key in teams and value == teams[key]:
            return {"$team": key}
        return {k: _compact(v, teams) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v, teams) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _write(path: pathlib.Path, data, teams: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"teams": teams, "data": _compact(data, teams)}
    path.write_text(json.dumps(payload, separators=(",", ":")))


def write_challenge(
    path: pathlib.Path, challenge: list[dict], teams: dict, shard_size: int = 100
) -> None:
    """The challenge pool as ``challenge/index.json`` and shards of
    ``shard_size`` games, ``challenge/<n>.json``.

    The page picks a game from the index (``[game_id, shard, markets]`` per
    game) and fetches only that game's shard, instead of a pool file that
    reaches tens of MB for college football. Each shard carries just the
    teams its games reference.
    """
    directory = path / "challenge"
    if directory.exists():
        for old in directory.glob("*.json"):
            old.unlink()
    (path / "challenge.json").unlink(missing_ok=True)  # single-file exports
    index = []
    for shard, start in enumerate(range(0, len(challenge), shard_size)):
        entries = challenge[start : start + shard_size]
        used = {e["game"][side]["team"] for e in entries for side in ("home", "away")}
        _write(
            directory / f"{shard}.json",
            entries,
            {key: team for key, team in teams.items() if key in used},
        )
        index += [[e["game"]["game_id"], shard, list(e["answers"])] for e in entries]
    _write(directory / "index.json", {"shard_size": shard_size, "games": index}, {})


CHALLENGE_KEYS = "challenge-keys.json"


@functools.cache
def _code_digest() -> str:
    """A digest of the runitback.wager package's source: an exporter change
    exports the challenge again."""
    package = pathlib.Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def challenge_key(sport_dir: str | pathlib.Path) -> str:
    """What a sport's challenge files are exported from: its built data
    (pool, games, teams, meta) and the exporter's code.

    The challenge only shows backtest games of complete seasons, with team
    summaries from the games before each one, so the daily refresh of the
    current season does not change it; a new build (the Train workflow) or
    a code change does. A source's revision of an old game reaches the
    challenge with the next build.
    """
    path = pathlib.Path(sport_dir)
    digest = hashlib.sha256(_code_digest().encode())
    for name in ("meta.json", "pool.csv", "games.csv", "teams.json"):
        file = path / name
        digest.update(name.encode())
        digest.update(file.read_bytes() if file.exists() else b"-")
    return digest.hexdigest()


def _fetch(source: str, name: str) -> bytes:
    """``name`` under a ``data/`` directory or a site's ``data/`` URL."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(f"{source.rstrip('/')}/{name}", timeout=30) as r:
            return r.read()
    return (pathlib.Path(source) / name).read_bytes()


def previous_challenge_keys(source: str | None) -> dict[str, str]:
    """The ``challenge-keys.json`` of a previous export, ``{}`` without one."""
    if not source:
        return {}
    try:
        return json.loads(_fetch(source, CHALLENGE_KEYS))
    except Exception as error:  # nothing to reuse: export the challenge
        logger.warning("no previous challenge keys from %s: %s", source, error)
        return {}


def reuse_challenge(source: str, sport: str, path: pathlib.Path) -> int | None:
    """Copy a previous export's ``<sport>/challenge/`` files into ``path``
    (the sport's output directory): the number of challenge games, or None
    when they cannot all be fetched (the caller exports them instead)."""
    try:
        index = _fetch(source, f"{sport}/challenge/index.json")
        games = json.loads(index)["data"]["games"]
        names = [f"{shard}.json" for shard in sorted({g[1] for g in games})]
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            shards = list(
                pool.map(lambda n: _fetch(source, f"{sport}/challenge/{n}"), names)
            )
    except Exception as error:
        logger.warning("could not reuse the %s challenge: %s", sport, error)
        return None
    directory = path / "challenge"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    (path / "challenge.json").unlink(missing_ok=True)  # single-file exports
    for name, data in zip(names, shards):
        (directory / name).write_bytes(data)
    (directory / "index.json").write_bytes(index)
    return len(games)


def export(
    data_dir: str = "webdata",
    output_dir: str = "web/static",
    refresh: bool = False,
    odds: dict[str, str] | None = None,
    odds_history: str | None = None,
    challenge_from: str | None = None,
    sports: list[str] | None = None,
    require_refresh: bool = True,
    player_values: dict[str, str] | None = None,
    data_root: str = "data",
) -> None:
    """Write every sport's page data as static JSON.

    Args:
        data_dir: Built web data (``runitback wager web build``).
        output_dir: Writes ``<output_dir>/data/``.
        refresh: Re-download each sport's results first (team state is
            refreshed with the saved model; nothing is retrained). A failed
            download fails the export, so the last published site stays.
        odds: Fetch current lines per sport from a provider, e.g.
            ``{nfl: oddsapi, cfb: espn}`` (The Odds API needs
            ``ODDS_API_KEY``; ESPN no key). Skipped when no game kicks off within 8
            days; a failed fetch keeps the previous lines and does not fail
            the export.
        odds_history: The previous ``data/`` (a directory or the deployed
            site's URL) whose ``<sport>/odds.json`` line history to extend.
        challenge_from: A previous ``data/`` (a directory or the deployed
            site's URL) whose challenge files to copy for each sport whose
            :func:`challenge_key` is unchanged, instead of exporting them
            again (the slowest part of an export).
        sports: Export only these sports (e.g. one per CI job, combined
            with :func:`merge`); default every sport in ``data_dir``.
        require_refresh: With ``refresh``, fail when every sport's download
            fails (off for a single sport's export: :func:`merge` checks).
        player_values: Each sport's player-values directory
            (``runitback <sport> player-values``), e.g. ``{nfl: player_values/nfl}``:
            exports its players page. Its processed data is read from
            ``<data_root>/<sport>/processed``.
        data_root: Where the sports' downloaded and processed data live.
    """
    keys = {}
    for path in pathlib.Path(data_dir).iterdir():
        if (path / "meta.json").exists():
            sport = json.loads((path / "meta.json").read_text())["sport"]
            if sports is None or sport in sports:
                keys[sport] = challenge_key(path)
    previous_keys = previous_challenge_keys(challenge_from)
    snapshots = load_snapshots(
        data_dir, refresh=refresh, sports=sports, require_refresh=require_refresh
    )
    out = pathlib.Path(output_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    ranks = {
        sport: payload
        for sport, directory in (player_values or {}).items()
        if sport in snapshots
        and (
            payload := player_ranks(
                sport, directory, pathlib.Path(data_root) / sport / "processed"
            )
        )
    }
    (out / "sports.json").write_text(
        json.dumps(
            [
                {**snap.info(), **({"has_players": True} if sport in ranks else {})}
                for sport, snap in snapshots.items()
            ]
        )
    )
    for sport, snap in snapshots.items():
        # the static page's CSP blocks external images: badges fall back to
        # the team-colored chip instead of making requests that fail
        teams = {key: {**team, "logo": None} for key, team in snap.teams.items()}
        snap.teams = teams

        default = snap.upcoming()
        by_date = {
            f"{default['season']}-{date}": snap.upcoming(default["season"], date)
            for date in default["dates"]
        }
        _write(
            out / sport / "upcoming.json",
            {"default": default, "by_date": by_date},
            teams,
        )

        if odds and sport in odds:
            cards = [g for week in by_date.values() for g in week["games"]]
            payload = refresh_odds(
                provider_for(sport, odds[sport]),
                cards,
                load_previous(odds_history, sport),
            )
            if payload:
                (out / sport).mkdir(parents=True, exist_ok=True)
                (out / sport / "odds.json").write_text(
                    json.dumps(payload, separators=(",", ":"), default=float)
                )
                print(
                    f"odds {sport}: {len(payload['games'])} games from "
                    f"{payload['provider']} (requests left: {payload['remaining']})"
                )

        game_ids = {g["game_id"] for week in by_date.values() for g in week["games"]}
        games = {game_id: snap.game(game_id) for game_id in sorted(game_ids)}
        # each game's lineup editor in its own file: only that game's page loads it
        for game_id, view in games.items():
            lineups = view.pop("lineups", None) if view else None
            if lineups:
                _write(out / sport / "lineups" / f"{game_id}.json", lineups, {})
        _write(out / sport / "games.json", games, teams)

        recent_dates = snap.recent_dates()
        recent = {
            f"{d['season']}-{d['date']}": snap.recent(d["season"], d["date"])["games"]
            for d in recent_dates
        }
        _write(
            out / sport / "recent.json",
            {"dates": recent_dates, "by_date": recent},
            teams,
        )
        recent_ids = sorted({g["game_id"] for day in recent.values() for g in day})
        _write(
            out / sport / "recent_games.json",
            {game_id: snap.game(game_id) for game_id in recent_ids},
            teams,
        )

        reused = None
        if challenge_from and previous_keys.get(sport) == keys.get(sport):
            reused = reuse_challenge(challenge_from, sport, out / sport)
        if reused is None:
            challenge = [
                {"game": snap.challenge_game(game_id), "answers": answers}
                for game_id in snap.pool.index
                if (answers := snap.challenge_answers(game_id))
            ]
            write_challenge(out / sport, challenge, teams)
        _write(out / sport / "performance.json", snap.performance(), teams)
        _write(out / sport / "rankings.json", snap.rankings(), teams)
        if sport in ranks:
            _write(out / sport / "players.json", ranks[sport], {})
        _write(out / sport / "map.json", snap.team_map_view(), teams)
        print(
            f"exported {sport}: {len(by_date)} dates, {len(games)} games, "
            f"{len(recent_ids)} recent games, "
            f"{len(challenge) if reused is None else reused} challenge games"
            f"{' (reused)' if reused is not None else ''} -> {out / sport}"
        )
    (out / CHALLENGE_KEYS).write_text(
        json.dumps({sport: keys[sport] for sport in snapshots if sport in keys})
    )


def merge(
    parts_dir: str = "parts",
    output_dir: str = "web/static",
    require_refresh: bool = False,
) -> None:
    """Combine exports of some sports each (``export --sports``) into one.

    Args:
        parts_dir: Holds one exported ``data/`` directory per subdirectory.
        output_dir: Writes ``<output_dir>/data/``: each sport's files, and
            ``sports.json`` and ``challenge-keys.json`` across the parts.
        require_refresh: Fail unless some sport was refreshed (``export
            --refresh``), as an export of every sport at once does: a part
            whose download failed kept its built data.
    """
    out = pathlib.Path(output_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    infos, keys = {}, {}
    for part in sorted(p for p in pathlib.Path(parts_dir).iterdir() if p.is_dir()):
        for info in json.loads((part / "sports.json").read_text()):
            sport = info["sport"]
            if sport in infos:
                raise ValueError(f"{sport} is in more than one part ({part})")
            infos[sport] = info
            shutil.copytree(part / sport, out / sport, dirs_exist_ok=True)
        if (part / CHALLENGE_KEYS).exists():
            keys |= json.loads((part / CHALLENGE_KEYS).read_text())
    if not infos:
        raise ValueError(f"no exported sports under {parts_dir}")
    if require_refresh and not any(i.get("refreshed_at") for i in infos.values()):
        raise RuntimeError("no sport was refreshed: every download failed")
    order = sorted(infos, key=_nav_order)
    (out / "sports.json").write_text(json.dumps([infos[s] for s in order]))
    (out / CHALLENGE_KEYS).write_text(
        json.dumps({s: keys[s] for s in order if s in keys})
    )
    print(f"merged {', '.join(order)} -> {out}")
