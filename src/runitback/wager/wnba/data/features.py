import dataclasses

from ...cbb.data.features import OPP_STATS_COLUMNS, STATS_COLUMNS
from ...nba.data.features import SCHEMA as NBA_SCHEMA

# The NBA's ESPN box score without the stats ESPN only has since 2023
# (cbb.data.features.RECENT_STATS), and the NBA's market and features.
SCHEMA = dataclasses.replace(NBA_SCHEMA, stats=tuple(STATS_COLUMNS + OPP_STATS_COLUMNS))
