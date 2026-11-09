import pandas as pd


def process_odds(odds, odds_map):
    """Consensus (mean over bookmakers) lines per game, home team's view.

    ``spread_line`` is the home team's handicap as books quote it (-3.0 means
    home favored by 3), i.e. the opposite sign of nflverse's ``spread_line``.
    """
    df = odds_to_dataframe(odds).dropna(subset=["last_update"])
    df = consensus_odds(df)
    df = df.replace({"home_team": odds_map, "away_team": odds_map})

    col_renamer = {
        col: col.removeprefix("home_") for col in df.columns if "home_" in col
    }
    col_renamer.update(
        {col: col.replace("away_", "opp_") for col in df.columns if "away_" in col}
    )

    df = df.rename(columns=col_renamer)

    return df


def odds_to_dataframe(odds):
    df = (
        pd.DataFrame(odds)
        .explode("bookmakers")
        .reset_index(drop=True)
        .dropna(subset=["bookmakers"])
    )
    bookmakers = (
        pd.DataFrame(df["bookmakers"].tolist())
        .rename(columns={"key": "bookmaker"})
        .drop(columns=["title", "last_update"])
    )
    df = pd.concat([df.drop(columns=["bookmakers"]), bookmakers], axis=1)

    df = df.explode("markets").reset_index(drop=True).dropna(subset=["markets"])
    markets = pd.DataFrame(df["markets"].tolist()).rename(columns={"key": "market"})
    df = pd.concat([df.drop(columns=["markets"]), markets], axis=1)

    df = pd.concat(
        [df.drop(columns=["outcomes"]), df.apply(format_row, axis=1)], axis=1
    )

    df["market"] = df["market"].replace(
        {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
    )

    return df


def consensus_odds(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "away_moneyline",
        "away_spread_odds",
        "home_moneyline",
        "home_spread_odds",
        "over_odds",
        "spread_line",
        "total_line",
        "under_odds",
    ]
    df = (
        df.groupby(["commence_time", "home_team", "away_team", "market"])[
            [col for col in cols if col in df]
        ]
        .mean()
        .reset_index()
        .assign(last_update=df["last_update"].max())
    )

    dfs = [
        df[df["market"] == market]
        .drop(columns=["market", "last_update"])
        .set_index(["home_team", "away_team"])
        .dropna(how="all", axis=1)
        for market in ["moneyline", "spread", "total"]
        if df["market"].notnull().sum() > 0
    ]

    df = (
        pd.concat(dfs, axis=1).reset_index().assign(last_update=df["last_update"].max())
    )

    df = df.loc[:, ~df.columns.duplicated()].dropna()

    df = df[
        (
            pd.to_datetime(df["commence_time"], utc=True)
            - pd.to_datetime(df["last_update"], utc=True)
        ).dt.days
        < 7
    ]

    return df


def format_row(row):
    if row["market"] == "h2h":
        return pd.Series(format_h2h(row))
    elif row["market"] == "spreads":
        return pd.Series(format_spreads(row))
    elif row["market"] == "totals":
        return pd.Series(format_totals(row))
    else:
        return pd.Series()


def format_totals(row):
    res = {}
    for val in row["outcomes"]:
        res[f"{val['name'].lower()}_odds"] = val["price"]
        res["total_line"] = val["point"]
    return res


def format_spreads(row):
    res = {}
    for outcome in row["outcomes"]:
        if outcome["name"] == row["home_team"]:
            res["home_spread_odds"] = outcome["price"]
            res["spread_line"] = outcome["point"]
        if outcome["name"] == row["away_team"]:
            res["away_spread_odds"] = outcome["price"]
    return res


def format_h2h(row):
    res = {}
    for outcome in row["outcomes"]:
        if outcome["name"] == row["home_team"]:
            res["home_moneyline"] = outcome["price"]
        if outcome["name"] == row["away_team"]:
            res["away_moneyline"] = outcome["price"]
    return res
