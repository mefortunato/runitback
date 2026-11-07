"""Does a foundation's passer head rank QBs? Its predictions from each QB's embedding vs his actual
rates (2016-24, >= 300 dropbacks).

    python scripts/passer_probe.py runs/po/fold2025/model.pt
"""
import sys
import numpy as np, pandas as pd, torch
from pathlib import Path
from runitback.nfl.sim.sampler import Simulator
from runitback.nfl.evaluate import load_lineup_data
from runitback.nfl.vocab import RESULTS, RS, PT, idx_to_yards, N_YARDS

ck = Path(sys.argv[1]); DATA = Path("data/nfl/processed")
sim = Simulator.load(ck); m = sim.model.eval(); ld = load_lineup_data(sim, DATA)
plays = pd.read_parquet(DATA / "plays.parquet"); games = pd.read_parquet(DATA / "games.parquet")
pers = pd.read_parquet(DATA / "personnel.parquet"); players = pd.read_parquet(DATA / "players.parquet")
print(players.columns.tolist()[:12])
grp = players.set_index("player_id")["group"] if "group" in players else None
p = plays[plays.play_type == "pass"].merge(pers, on=["game_id", "play_id"]).merge(games[["game_id", "season"]], on="game_id")
p = p[(p.season >= 2016) & (p.season <= 2024)]
ids = np.where(p.pos == 0, p.home_ids, p.away_ids)
qbs = [next((x for x in row if grp.get(x) == "QB"), None) for row in ids]
p = p.assign(qb=qbs).dropna(subset=["qb"])
act = p.groupby("qb").agg(n=("result", "size"), comp=("result", lambda r: (r == "complete").mean()),
                          sack=("result", lambda r: (r == "sack").mean()),
                          intc=("result", lambda r: r.isin(["interception", "int"]).mean()), yds=("yards", "mean"))
act = act[act.n >= 300]
yv = torch.tensor([idx_to_yards(i) for i in range(N_YARDS)], dtype=torch.float32)
rows = []
with torch.no_grad():
    for qb, a in act.iterrows():
        tok = ld.pvocab.idx.get(qb)
        if not tok: continue
        e = m.lineup.player(torch.tensor([tok]))
        pr = torch.softmax(m.qb_res(e), -1)[0]
        cm = RS["complete"]
        py = torch.softmax(m.qb_yds(e + m.e_res(torch.tensor([cm]))), -1)[0]
        rows.append(dict(qb=qb, name=players.set_index("player_id").name.get(qb), n=a.n, comp=a.comp, p_comp=float(pr[cm]),
                         sack=a.sack, p_sack=float(pr[RS["sack"]]) if "sack" in RS else np.nan,
                         yds=a.yds, p_yds_complete=float((py * yv).sum()), norm=float(e.norm())))
df = pd.DataFrame(rows)
print(RESULTS)
print(len(df), "QBs with >=300 dropbacks 2016-24 and an embedding")
for c, pc in (("comp", "p_comp"), ("sack", "p_sack")):
    print(f"corr actual {c} vs head: {df[c].corr(df[pc]):.2f}")
names = ["00-0033873", "00-0034857", "00-0033662", "00-0032156", "00-0034955"]
print(df[df.qb.isin(names)][["name", "n", "comp", "p_comp", "sack", "p_sack", "p_yds_complete", "norm"]].to_string())
print(df.sort_values("p_comp")[["name", "n", "comp", "p_comp"]].iloc[[0, 1, 2, -3, -2, -1]].to_string())
