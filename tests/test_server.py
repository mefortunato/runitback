"""The combined server: predictions data under /data and the web app's per-sport pages."""
import importlib

from fastapi.testclient import TestClient


def _app(tmp_path, monkeypatch, predictions=True):
    web = tmp_path / "dist"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<html>app</html>")
    monkeypatch.setenv("RUNITBACK_WEB", str(web))
    if predictions:
        data = tmp_path / "predictions"
        (data / "nfl").mkdir(parents=True)
        (data / "sports.json").write_text('[{"sport": "nfl"}]')
        monkeypatch.setenv("RUNITBACK_PREDICTIONS", str(data))
    else:
        monkeypatch.delenv("RUNITBACK_PREDICTIONS", raising=False)
    import runitback.server as server
    return TestClient(importlib.reload(server).app)


def test_predictions_data_and_sport_pages(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch)
    r = c.get("/data/sports.json")
    assert r.json() == [{"sport": "nfl"}]
    assert r.headers["cache-control"] == "no-cache"
    assert c.get("/data/nfl/upcoming.json").status_code == 404   # not exported: the page says so
    for page in ("/nfl", "/nfl/rankings", "/nfl/games/2025_01_KC_BAL", "/nfl/generator", "/nba"):
        assert c.get(page).text == "<html>app</html>"


def test_without_exported_predictions_data_paths_404(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch, predictions=False)
    assert c.get("/data/sports.json").status_code == 404
    assert c.get("/nfl").text == "<html>app</html>"


def test_every_ported_sport_has_pages(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch)
    data = tmp_path / "predictions"
    (data / "nba").mkdir()
    (data / "nba" / "upcoming.json").write_text('{"teams": {}, "data": {}}')
    assert c.get("/data/nba/upcoming.json").json() == {"teams": {}, "data": {}}
    for sport in ("nfl", "cfb", "nba", "cbb", "wnba", "wcbb", "nhl", "mlb"):
        for page in (f"/{sport}", f"/{sport}/recent", f"/{sport}/rankings", f"/{sport}/games/401"):
            assert c.get(page).text == "<html>app</html>"
        assert c.get(f"/api/{sport}/nope").status_code == 404


def test_sports_listing_matches_the_exports_and_the_web_app(tmp_path, monkeypatch):
    """/api/sports, the predictions export's sports and web/src/sports.ts list the same sports in
    the same order; only the NFL and NBA have generators."""
    import re
    from pathlib import Path

    from runitback.wager.web.views import BUILTIN_VIEWS

    listed = _app(tmp_path, monkeypatch).get("/api/sports").json()["sports"]
    assert [s["id"] for s in listed] == list(BUILTIN_VIEWS)
    assert [s["id"] for s in listed if s["generator"]] == ["nfl", "nba"]
    assert not any(s["model_available"] for s in listed if not s["generator"])
    web = (Path(__file__).resolve().parents[1] / "web" / "src" / "sports.ts").read_text()
    entries = re.findall(r'\{ id: "(\w+)", label: "(\w+)", generator: (true|false) \}', web)
    assert [(i, g == "true") for i, _, g in entries] == [(s["id"], s["generator"]) for s in listed]
    assert [label for _, label, _ in entries] == [s["name"] for s in listed]
