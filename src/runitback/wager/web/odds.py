"""Current sportsbook lines for the site's upcoming games (``odds.json``).

The models do not take lines as inputs, so fresh odds only change how the
predictions compare with the market: the browser reprices every pick from
these lines (``web/src/lib/pricing.ts``). One file per sport:

``{"updated", "provider", "remaining", "games": {game_id: {...}}}`` where
each game has every book's line (``books``), the median line
(``consensus``), the best price on each side (``best``), the opening line
(``open``) and the consensus over time (``history``), all from the home
team's side as the site shows it (``spread`` is the home team's expected
margin).

History carries over between refreshes: pass the previous ``odds.json``
(the deployed site's) and its points are kept for games still upcoming.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import pathlib
import urllib.request

import numpy as np
import pandas as pd

from ..backtest import payout
from ..odds.providers import Provider

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 200
LINE_COLUMNS = [
    "spread", "spread_home_odds", "spread_away_odds", "ml_home", "ml_away",
    "total", "over_odds", "under_odds",
]  # fmt: skip


def _num(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _kickoff(card: dict) -> pd.Timestamp | None:
    """A card's kickoff as UTC (cards carry US Eastern "YYYY-MM-DD HH:MM")."""
    if not card.get("kickoff"):
        return None
    stamp = pd.Timestamp(card["kickoff"])
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("US/Eastern")
    return stamp.tz_convert("UTC")


def match_games(lines: pd.DataFrame, cards: list[dict]) -> pd.DataFrame:
    """``lines`` with ``game_id`` and oriented to the card's home team.

    A line matches the upcoming game between the same two teams (either way
    round: providers sometimes list a neutral-site game's teams the other
    way) kicking off nearest its ``commence`` time, within 3 days. Teams
    match on their keys or, for sports without a sportsbook name map
    (college basketball's 360 schools), on their display names. Lines from
    the other side are flipped: spread negated, prices swapped.
    """

    def side(team: dict) -> set:
        return {team["team"], team.get("name")} - {None}

    rows = []
    for _, line in lines.iterrows():
        best, gap = None, pd.Timedelta(days=3)
        for card in cards:
            home, away = side(card["home"]), side(card["away"])
            straight = line["home"] in home and line["away"] in away
            if not (straight or (line["home"] in away and line["away"] in home)):
                continue
            kickoff = _kickoff(card)
            delta = (
                abs(kickoff - line["commence"])
                if kickoff is not None and not pd.isna(line["commence"])
                else pd.Timedelta(0)
            )
            if delta <= gap:
                best, gap = card, delta
        if best is None:
            continue
        row = line.to_dict()
        if line["home"] not in side(best["home"]):
            row |= {
                "spread": -line["spread"] if not pd.isna(line["spread"]) else np.nan,
                "spread_open": -line["spread_open"]
                if not pd.isna(line["spread_open"])
                else np.nan,
                "spread_home_odds": line["spread_away_odds"],
                "spread_away_odds": line["spread_home_odds"],
                "ml_home": line["ml_away"],
                "ml_away": line["ml_home"],
            }
        row["game_id"] = best["game_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def _best(books: list[dict], side: str) -> dict | None:
    """The best price on one side: the most points (then the best payout)."""

    def key(book):
        if side in ("home", "away") and "points" not in book:
            return payout(book["odds"])
        # home covers below its spread, away above; over wants a low total
        points = {"home": -1, "away": 1, "over": -1, "under": 1}[side] * book["points"]
        return (points, payout(book["odds"] if book["odds"] is not None else -110))

    offers = [b for b in books if b is not None]
    return max(offers, key=key) if offers else None


def game_odds(rows: pd.DataFrame) -> dict:
    """One game's books, consensus, best prices and opening line."""
    books = [
        {"book": row["book"], **{c: _num(row[c]) for c in LINE_COLUMNS}}
        for _, row in rows.iterrows()
    ]
    consensus = {c: _num(rows[c].median()) for c in LINE_COLUMNS}

    def offers(points, odds):
        out = []
        for b in books:
            if points is not None and b[points] is None:
                continue
            if points is None and b[odds] is None:
                continue
            offer = {"book": b["book"], "odds": b[odds]}
            if points is not None:
                offer["points"] = b[points]
            out.append(offer)
        return out

    best = {
        "spread": {
            "home": _best(offers("spread", "spread_home_odds"), "home"),
            "away": _best(offers("spread", "spread_away_odds"), "away"),
        },
        "moneyline": {
            "home": _best(offers(None, "ml_home"), "home"),
            "away": _best(offers(None, "ml_away"), "away"),
        },
        "total": {
            "over": _best(offers("total", "over_odds"), "over"),
            "under": _best(offers("total", "under_odds"), "under"),
        },
    }
    opening = {
        "spread": _num(rows["spread_open"].median()),
        "total": _num(rows["total_open"].median()),
    }
    return {"books": books, "consensus": consensus, "best": best, "open": opening}


def build_odds(
    lines: pd.DataFrame,
    cards: list[dict],
    previous: dict | None = None,
    now: str | None = None,
    provider: str | None = None,
    remaining: int | None = None,
) -> dict:
    """The ``odds.json`` payload for one sport (see the module docstring)."""
    now = now or datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    upcoming = {card["game_id"] for card in cards}
    old = (previous or {}).get("games", {})
    games = {}
    matched = match_games(lines, cards) if len(lines) else pd.DataFrame()
    for game_id, rows in matched.groupby("game_id") if len(matched) else []:
        game = game_odds(rows)
        c = game["consensus"]
        point = {
            "t": now,
            **{k: c[k] for k in ("spread", "total", "ml_home", "ml_away")},
        }
        history = [p for p in old.get(game_id, {}).get("history", [])]
        history = (history + [point])[-HISTORY_LIMIT:]
        # the first line seen opens, unless the provider reports its own
        first = history[0]
        game["open"] = {
            "spread": game["open"]["spread"]
            if game["open"]["spread"] is not None
            else first.get("spread"),
            "total": game["open"]["total"]
            if game["open"]["total"] is not None
            else first.get("total"),
        }
        games[game_id] = {"updated": now, **game, "history": history}
    # games this refresh missed keep their last lines while still upcoming
    for game_id, game in old.items():
        if game_id in upcoming and game_id not in games:
            games[game_id] = game
    return {
        "updated": now if len(matched) else (previous or {}).get("updated"),
        "provider": provider,
        "remaining": remaining,
        "games": games,
    }


def load_previous(source: str | None, sport: str) -> dict | None:
    """The last published ``<sport>/odds.json`` from a directory or a site
    URL (its ``data/`` folder); None if there is none yet."""
    if not source:
        return None
    try:
        if source.startswith(("http://", "https://")):
            url = f"{source.rstrip('/')}/{sport}/odds.json"
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read())
        path = pathlib.Path(source) / sport / "odds.json"
        return json.loads(path.read_text()) if path.exists() else None
    except Exception as error:  # a missing history is not worth failing a build
        logger.warning("no previous odds for %s from %s: %s", sport, source, error)
        return None


def refresh_odds(
    provider: Provider,
    cards: list[dict],
    previous: dict | None = None,
    window_days: float = 8,
    now: datetime.datetime | None = None,
) -> dict | None:
    """Fetch and build a sport's odds, or keep the previous ones when there
    is nothing to fetch for (no game within ``window_days``, which keeps a
    small quota for the season) or the fetch fails (odds are optional; the
    site still builds)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    soon = [
        c
        for c in cards
        if (k := _kickoff(c)) is not None
        and k <= pd.Timestamp(now) + pd.Timedelta(days=window_days)
    ]
    if not soon:
        logger.info("no games within %s days: odds not fetched", window_days)
        return previous
    try:
        lines = provider.fetch()
    except Exception as error:
        logger.warning("odds fetch from %s failed: %s", provider.name, error)
        return previous
    return build_odds(
        lines,
        cards,
        previous,
        now=now.isoformat(timespec="seconds"),
        provider=provider.name,
        remaining=provider.remaining,
    )
