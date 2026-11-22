from ..core.efficiency import epl_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

# Football Elo (EloParams(mov="football"): World Football Elo's goal-difference multiplier, so a
# draw moves ratings), tuned by scripts/epl_elo_tune.py on 2006-2016 with 2001-2005 as burn-in, all
# before the first test season (2022): the setting with the best ranked probability score of the
# rating gap's ordered probit (RPS 0.19577 against the NFL settings' 0.19663 on the same games);
# points_per_elo is the least-squares slope (1/202). The surface is flat: k 15 and regress 0.05 /
# 0.1 are within 0.0001 of each other, and so is a new team's start 50-150 below the mean (100 is
# used). The best home edge is 80 on 2006-2016, 2006-2021 and 2006-2021 without 2020-21
# (experiments/epl/README.md). A club the model has no state for (promoted, or back after years away)
# is predicted from an empty history (``cold_start``), the transformer is trained on such games
# (``min_history`` 0) and a team's history stops two seasons back (``drop_older_than``), as the
# state it is served from does. No other GNN or transformer settings yet.
SPORT = Sport(
    name="epl",
    schema=SCHEMA,
    download=download,
    efficiency=epl_efficiency,
    cold_start=True,
    transformer={"min_history": 0, "drop_older_than": 2},
    elo={
        "k": 15.0,
        "home": 80.0,
        "regress": 0.05,
        "points_per_elo": 1 / 200,
        "mov": "football",
        "new_team": 1405.0,
    },
)
