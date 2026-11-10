import dataclasses

from ...cbb.data.teams import load_women_teams
from .cbb import VIEW as MEN

VIEW = dataclasses.replace(
    MEN,
    sport="wcbb",
    label="WCBB",
    load_teams=load_women_teams,
)
