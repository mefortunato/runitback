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
    assert c.get("/data/sports.json").json() == [{"sport": "nfl"}]
    assert c.get("/data/nfl/upcoming.json").status_code == 404   # not exported: the page says so
    for page in ("/nfl", "/nfl/rankings", "/nfl/games/2025_01_KC_BAL", "/nfl/generator", "/nba"):
        assert c.get(page).text == "<html>app</html>"


def test_without_exported_predictions_data_paths_404(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch, predictions=False)
    assert c.get("/data/sports.json").status_code == 404
    assert c.get("/nfl").text == "<html>app</html>"
