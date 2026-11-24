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
    # The transformer's Elo tokens use Sport.elo (models/transformer/data.py ``strength``); the
    # ``epl`` sport, its efficiency function and its three-outcome (draws) mode (GameSchema.draws;
    # core/metrics.py, core/evaluation.calibrate, core/predictor.py, backtest.py, experiments.py,
    # tune.py and staking.py's 1X2 market), EloParams' football rule (models/baselines.py) and cold start
    # (Sport.cold_start, the transformer's drop_older_than / eff2 / draws market tokens, TransformerModel's
    # distribution ranges, rps_weight, experiments' `family`), every
    # new branch of which is off for a schema without draws, for EloParams' defaults and for a Sport
    # without cold_start. The only
    # production transformers that read the Elo tokens are cfb, cbb and wcbb (``opponent_strength``),
    # and none of them sets Sport.elo; nhl, mlb and wnba set it but do not use those tokens. So no
    # existing sport's walk-forward results changed (tests/wager/test_outcome_identity.py pins the
    # two-outcome outputs). The code of main after #110 and #111, #110, and before; regenerate with
    # scripts/equivalent_code.py after any later change to hashed code.
    # nfl
    "e0700a1b3bddcd9b777d629cde6cf60c247588bfc45ecd33d7e7e28f6af1a578": (
        "85e0396c4305482d5f903225c2d6c9e9ce03cfaea557ead52466581c2113b300",
    ),
    # cfb
    "5080b7bb61c9deb06bdef57d9425c875236bb1904ebe5debadb7b858c223e8a0": (
        "0df622ddd4b4f59d2c71f8e843dcabee2c1b0abd7fb9390eeacbad591df6ae6e",
    ),
    # nba
    "576d183d998ac8161309e2596652e61daf244d69f1f9aa21c623806d33a39485": (
        "f7a55e00c54bbfadb186faac858a3adfdf892fdec2e92dec284b544aaaa76693",
    ),
    # cbb
    "4d68bd949b29e0ac0c980c7fb82cca65f08aaf492b17eefeb568445a5ac8f7a8": (
        "95fe4dd6de56ca91ffcda43e97a9cb45e016c8e89107f47bc14e35dcf8d73272",
    ),
    # wcbb
    "8a2bea9229d7c3048f43bc83ac66d721b48b76d7cec651f963ee9fbff4aa43c8": (
        "f1edc337306155a719f9b63f0191da27d1b8d5090140e70c01101103ed76303a",
    ),
    # wnba
    "d576b284d69dbd721d357ba2d7b4da84e5a34fedf963e59df86be06d015a79a2": (
        "08fc830245c03bdd9d5af9c7e9ebbb2724516675ef5179503db2d9893d53a16d",
    ),
    # nhl
    "8ec7cdf1555f27accdf8b2b18cc9662f838c28b02aa0c3ef31f41d95aa80fbcc": (
        "8af32119569361b8600b0a8c304720903bea243105051a92fe364cb2f1417ae7",
    ),
    # mlb
    "25ed226f87b44bf320a4dc38d9e097e833332d3c323a5c6e6ff7a8c3dc483f29": (
        "5bbf73ec765cd9d807dadb298084807a66db70646526d7a1dd4086840ff45e99",
    ),
}
