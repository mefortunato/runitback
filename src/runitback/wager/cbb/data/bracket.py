import random

import networkx as nx
import numpy as np
import pandas as pd


class Bracket(nx.DiGraph):
    def __init__(self, slots, seeds):
        super().__init__()

        for idx, row in slots.iterrows():
            self.add_edge(row.StrongSeed, row.Slot)
            self.add_edge(row.WeakSeed, row.Slot)
        for idx, row in seeds.iterrows():
            self.nodes[row.Seed]["team_id"] = row.TeamID

        self.team_seed_map = seeds.set_index("TeamID")["Seed"]

        self.championship = [node for node, degree in self.out_degree() if degree == 0]
        if len(self.championship) != 1:
            raise ValueError(
                f"only 1 final championship game should exist {self.championship}"
            )
        self.championship = self.championship[0]

        self.games = sorted(
            [
                node
                for node, data in self.nodes(data=True)
                if data.get("team_id") is None
            ],
            key=lambda x: nx.shortest_path_length(self, x, self.championship),
            reverse=True,
        )

    def reset(self):
        for node, degree in self.degree():
            if degree > 1:
                if self.nodes[node].get("team_id"):
                    del self.nodes[node]["team_id"]
                if self.nodes[node].get("seed"):
                    del self.nodes[node]["seed"]

    def simulate(self, predictor):
        for game in self.games:
            predecessors = list(self.predecessors(game))
            team_ids = [self.nodes[pred]["team_id"] for pred in predecessors]
            winner = predictor(*team_ids)
            self.nodes[game]["team_id"] = winner
            self.nodes[game]["seed"] = self.team_seed_map[winner]
        return self.nodes[self.championship]

    def simulate_random(self):
        return self.simulate(lambda x, y: random.choice([x, y]))


class ProbabilityPredictor:
    def __init__(self, probabilities: pd.DataFrame):
        self.probabilities = probabilities

    def __call__(self, team_x, team_y):
        prob = self.probabilities.loc[team_x, team_y]
        return team_x if np.random.random() < prob else team_y
