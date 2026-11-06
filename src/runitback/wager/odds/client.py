import datetime
import os
import zoneinfo

import httpx


class OddsAPIClient(object):
    """Minimal client for https://the-odds-api.com (key: ``ODDS_API_KEY``)."""

    def __init__(self, api_key: str = None, version: int = 4):
        self.host = f"https://api.the-odds-api.com/v{version}"
        if api_key is None:
            api_key = os.getenv("ODDS_API_KEY")
        self.api_key = api_key

    def get(self, path: str, params: dict | None = None):
        params = {**(params or {}), "api_key": self.api_key}
        resp = httpx.get(f"{self.host}/{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def sports(self):
        return self.get("sports")

    def odds(
        self,
        sport: str,
        date: datetime.datetime | str = None,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        convert_timezone: str = "US/Eastern",
    ):
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        path = f"sports/{sport}/odds"
        if date is not None:
            path = f"historical/{path}"
            if isinstance(date, str):
                date = datetime.datetime.fromisoformat(date)
            params["date"] = date.strftime("%Y-%m-%dT%H:%M:%SZ")
        odds = self.get(path, params)
        if date is not None:
            odds = odds["data"]
        tz = zoneinfo.ZoneInfo(convert_timezone)
        for odd in odds:
            dt = datetime.datetime.fromisoformat(odd["commence_time"])
            odd["commence_time"] = dt.astimezone(tz).isoformat()
        return odds
