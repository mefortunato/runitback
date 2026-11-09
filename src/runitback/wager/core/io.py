import pandas as pd

# Annotation for tables: a DataFrame, or a CSV path (what the CLI passes).
Table = str | pd.DataFrame


def read_table(table: Table) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table
    return pd.read_csv(table, low_memory=False)
