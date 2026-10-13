"""
class_features.py  -  champion CLASS matchups (run from 02_features/)

Exact champion-vs-champion lane pairs (f_lane_matchup) are rare, so most values sit near 0.
Pooling champions into classes gives each pair hundreds of games.
Classes from the League of Legends Wiki (wiki.leagueoflegends.com/en-us/Champion_classes),
stored in champion_classes.csv: 13 subclasses inside 7 classes. A champion can have 2 subclasses
(e.g. Jhin = Marksman + Catcher); then every combination is used and averaged.

For each game, using only games BEFORE it (values from blue's side):
  f_class_lane  : sum over the 5 lanes of (win rate of my SUBCLASS vs the enemy SUBCLASS in that
                  lane, last 365 days, shrunk to 50 % with 10 pseudo-games) - 0.5
                  e.g. "Juggernaut top vs Skirmisher top"
  f_class_cross : mean over all 25 blue x red champion pairs (any lanes) of the CLASS-vs-CLASS
                  win rate (365 days, shrunk, 10 pseudo-games) - 0.5
                  catches cross-lane counters, e.g. Tank jungler vs Mage mid, engage vs poke
Draft is known only ~10 min before the game (same caveat as draft_features.py).
Output: class_features.csv
"""
import glob
import re
from collections import defaultdict, deque
from itertools import product

import numpy as np
import pandas as pd

import build_features as bf

SPAN, A = pd.Timedelta(days=365), 10
POS = ["top", "jng", "mid", "bot", "sup"]
COLS = ["gameid", "league", "date", "side", "position", "champion", "result"]


def key(name):
    return re.sub(r"[^a-z]", "", str(name).lower())


class RunningWindow:
    """Wins / games per key over a sliding time window, with running sums (fast for big pools)."""
    def __init__(self, span):
        self.span = span
        self.h = defaultdict(deque)
        self.s = defaultdict(lambda: [0.0, 0.0])        # key -> [weight, wins]

    def _trim(self, k, now):
        h, s = self.h[k], self.s[k]
        while h and h[0][0] < now - self.span:
            _, w, y = h.popleft()
            s[0] -= w
            s[1] -= w * y

    def rate(self, k, now):
        self._trim(k, now)
        n, w = self.s[k]
        return (w + A * 0.5) / (n + A)

    def add(self, k, now, y, w=1.0):
        self.h[k].append((now, w, y))
        self.s[k][0] += w
        self.s[k][1] += w * y


def main():
    cc = pd.read_csv("champion_classes.csv")
    SUB = {key(r.champion): r.subclasses.split("|") for r in cc.itertuples()}
    CLS = {key(r.champion): r.classes.split("|") for r in cc.itertuples()}

    parts = []
    for f in sorted(glob.glob("../01_data/*LoL_esports_match_data*.csv")):
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        parts.append(d[d["position"].isin(POS)])
    d = pd.concat(parts, ignore_index=True)
    d["league"] = d["league"].replace(bf.LEAGUE_MAP)
    d = d[d["league"].isin(bf.LEAGUES) & d["champion"].notna()].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["k"] = d["champion"].map(key)
    missing = sorted(set(d.loc[~d["k"].isin(SUB), "champion"]))
    print(f"champions without a class: {missing if missing else 'none'}")
    d = d[d.groupby("gameid")["k"].transform("size") == 10].sort_values(["date", "gameid"])

    lane, cross = RunningWindow(SPAN), RunningWindow(SPAN)
    rows = []
    for gid, g in d.groupby("gameid", sort=False):
        now = g["date"].iloc[0]
        B = g[g["side"] == "Blue"].set_index("position")
        R = g[g["side"] == "Red"].set_index("position")
        if len(B) != 5 or len(R) != 5 or not all(k in SUB for k in g["k"]):
            continue
        # ---- score (before update)
        f_lane = 0.0
        for pos in POS:
            pairs = list(product(SUB[B.at[pos, "k"]], SUB[R.at[pos, "k"]]))
            f_lane += np.mean([lane.rate((pos, a, b), now) for a, b in pairs]) - 0.5
        cr = []
        for cb, cr_ in product(B["k"], R["k"]):
            pairs = list(product(CLS[cb], CLS[cr_]))
            cr.append(np.mean([cross.rate((a, b), now) for a, b in pairs]))
        rows.append({"gameid": gid, "f_class_lane": f_lane, "f_class_cross": np.mean(cr) - 0.5})
        # ---- update AFTER scoring; both perspectives, multi-class champions split weight
        bw = int(B["result"].iloc[0])
        for pos in POS:
            pairs = list(product(SUB[B.at[pos, "k"]], SUB[R.at[pos, "k"]]))
            for a, b in pairs:
                lane.add((pos, a, b), now, bw, 1 / len(pairs))
                lane.add((pos, b, a), now, 1 - bw, 1 / len(pairs))
        for cb, cr_ in product(B["k"], R["k"]):
            pairs = list(product(CLS[cb], CLS[cr_]))
            for a, b in pairs:
                cross.add((a, b), now, bw, 1 / len(pairs))
                cross.add((b, a), now, 1 - bw, 1 / len(pairs))
    out = pd.DataFrame(rows)
    out.to_csv("class_features.csv", index=False)
    print(f"Saved class_features.csv: {len(out)} games")
    print(out.drop(columns="gameid").describe().round(4).to_string())


if __name__ == "__main__":
    main()
