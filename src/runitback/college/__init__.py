"""College basketball (men's ``cbb`` and women's ``wcbb``) player data, values and rotations.

Work package C3 of ``docs/multisport/PLAN.md``: layer B (player values) and layer C (projected
rotations) of the NFL player model for both genders. See ``experiments/cbb/README.md``.

- :mod:`.data`: ESPN player box scores -> ``player_games.parquet`` / ``players.parquet``.
- :mod:`.lineups`: each game's projected rotation (recent minutes minus players who didn't play).
- :mod:`.box_values`: minutes-weighted box-score values, refit weekly (both genders).
- :mod:`.cbbd_lineups`: men's possessions with the ten on the floor from CollegeBasketballData.
- :mod:`.player_values`: men's adjusted plus-minus shrunk toward the box values, and the per-game
  ``pv.parquet`` for the wager model's lineup terms.
"""
