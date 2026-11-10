"""Code hashes whose backtests are the same as an older code's.

A change to hashed source (:func:`runitback.wager.web.build.code_hash`) gives
every sport new backtest keys, even when it changes the results of only
some sports. Mapping a sport's new hash to the old one here lets a build
reuse the seasons stored under the old keys (and store them again under
the new ones) instead of backtesting them again.

The map is keyed by the new hash, so it applies only to exactly that
source: any later change to hashed code makes it inert. This module is
under ``web/`` and so not hashed itself. Add an entry only when the
change leaves that sport's walk-forward results identical.
"""

from __future__ import annotations

EQUIVALENT_CODE: dict[str, tuple[str, ...]] = {
    # Weekly power rankings for college basketball (``ranking_every``),
    # and the main model backtesting the season it validated on (under keys
    # of its own): every other sport's seasons from a backtest model are
    # produced as before.
    # nfl
    "086f4177a50b208cd7ed856a2e578f66ce09c37ada43721ba024303b9cb6f458": (
        "9f8bdd5b6fe3111ee009e08c0e54bedd127ad0c59cb42c0e9d8465eb08db684d",
    ),
    # cfb
    "e3b4384602cf56a0d1937ce838fd379a2fc24d7ab8d8e539dee9d293cf5b4748": (
        "3fcdd4044659df88f9f195100a1708ef40794e8b8b059056df1de39151b5cea9",
    ),
    # nba
    "b951c2e2523bfd7644d897bc8de62ce7cfb30e63dcb640849c5e3c848c070dab": (
        "e0e6dc1367a80f3c528254c80fcd0a741f377f4966e0cdc4abc9b7e49c100ac6",
    ),
    # wnba
    "2efa07541d3349901651aa6098e009ecc5b5324c3bc830048ca373aea124f29f": (
        "5f3572bf56b9d982c22886186910b1f2bb5e9a70ed57aeef20499266ec33af2b",
    ),
    # nhl
    "910500b668ce68f236413adfb184cdb5e8fb4d7c92e8648a239a11f03e0083f5": (
        "f1b5541a229ceaf439d50e8b4336daa2ce3d42312979c7ac431b115126a9fc6e",
    ),
    # mlb
    "c6b81294866cf1e201f8fb96cee17ab6691da011000a10ad0d5200f00f2904a3": (
        "8158c871cd70aabecfa85bd55019e1c18a2e199ae98a4557db53f62e95c2342c",
    ),
    # S4c (mlb), S4d (cfb) and S4a (nba, wnba, cbb, wcbb): sports ported from sportsml and added
    # to BUILTIN_SPORTS (core/sport.py); nothing the NFL trains on or predicts with changed. The
    # NFL code after S4d, after S4c, and before either.
    # nfl
    "5a7e94f366292c2a6bf12774a323fb5098f78465e6b47b1c3a9ee44ddf7aa4d4": (
        "174f47a238ad1c9bdf44a97bffe1ce5d61065a2ed14e1bfa0532855708513074",
        "f297abba0effffa148a4e535957011d0747e2d1b34e55127f705e8045dd3e028",
        "d51c92ffb1132c3abdc0a5ece6a07f8f881b88a35ae694c7b4ff5b1e73246922",
    ),
}
