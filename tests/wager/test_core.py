import numpy as np
import pandas as pd
import pytest

from runitback.wager.core.metrics import antisymmetrize, evaluate, win_probability
from runitback.wager.core.process import process
from runitback.wager.core.schema import mirror, played, upcoming, validate
from runitback.wager.core.splits import SeasonSplit, split_seasons

from .conftest import TOY_SCHEMA


def test_mirror_swaps_paired_and_negates_location():
    row = pd.DataFrame(
        {"team": ["A"], "opp": ["B"], "location": [1], "rest": [7], "opp_rest": [4]}
    )
    out = mirror(row, TOY_SCHEMA).iloc[0]
    assert (out.team, out.opp, out.location, out.rest, out.opp_rest) == (
        "B",
        "A",
        -1,
        4,
        7,
    )
    assert mirror(mirror(row, TOY_SCHEMA), TOY_SCHEMA).equals(row)


def test_played_and_upcoming_partition(toy_games):
    assert len(played(toy_games, TOY_SCHEMA)) + len(
        upcoming(toy_games, TOY_SCHEMA)
    ) == len(toy_games)
    assert upcoming(toy_games, TOY_SCHEMA)["season"].eq(2024).all()


def test_validate_reports_missing_columns(toy_games):
    with pytest.raises(ValueError, match="rest"):
        validate(toy_games.drop(columns=["rest"]), TOY_SCHEMA)


def test_antisymmetrize_pairs_rows_by_game():
    rows = pd.DataFrame({"game_id": ["g1", "g1", "g2"]})
    out = antisymmetrize(rows, [4.0, 2.0, 5.0], "game_id")
    np.testing.assert_allclose(out, [1.0, -1.0, 5.0])


def test_split_defaults_and_overlap():
    split = split_seasons([2020, 2021, 2022, 2023])
    assert split == SeasonSplit(train=[2020, 2021], val=[2022], test=[2023])
    assert split_seasons([2020, 2021], val=[], test=[]).train == [2020, 2021]
    with pytest.raises(ValueError, match="overlap"):
        split_seasons([2020, 2021], train=[2020, 2021], test=[2021])


def test_evaluate_excludes_ties_from_winner_metrics():
    m = evaluate(y=[3, -3, 0], margin=[1, -1, 2], win_prob=[0.9, 0.1, 0.8])
    assert m["accuracy"] == 1.0
    assert m["brier"] == pytest.approx(0.01)


def test_win_probability_is_symmetric():
    p = win_probability([-3.0, 0.0, 3.0], sigma=13.0)
    assert p[1] == 0.5 and p[0] == pytest.approx(1 - p[2])


def test_process_keeps_upcoming_and_drops_unpaired(toy_games):
    games = pd.concat([toy_games, toy_games.iloc[[0]].assign(game_id="orphan")])
    out = process("toy", games, output_file=None)
    assert "orphan" not in set(out.game_id)
    assert out.margin.isna().sum() == toy_games.margin.isna().sum()
