from ..core.efficiency import epl_efficiency
from ..core.sport import Sport
from .data.download import download
from .data.features import SCHEMA

# Untuned: no Elo / GNN / transformer settings yet (the games table is the
# ingest; the draw, a third of the games, is not modeled by the two-way
# moneyline columns).
SPORT = Sport(
    name="epl",
    schema=SCHEMA,
    download=download,
    efficiency=epl_efficiency,
)
