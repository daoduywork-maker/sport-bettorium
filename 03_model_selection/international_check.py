"""
international_check.py  -  how good is the model at international events (MSI, Worlds, First Stand, EWC)?

Problem: each league is a closed system, so a 1600-Elo LCK team and a 1600-Elo LCS team are NOT equal.
We rebuild the features with international games included, and add a REGION Elo: one rating per
home league, updated only by cross-region international games (K = 32).
Walk-forward per event: the LR is trained on every game (domestic + earlier internationals) played
before the event starts, then scores that event. Run from 03_model_selection/.
"""
import json, sys
from collections import Counter
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "../02_features")
import build_features as bf

INTL = {"MSI", "WLDs", "FST", "EWC"}
bf.LEAGUES = bf.LEAGUES | INTL
C = json.load(open("results/final_models.json"))["logloss"]["hyperparams"]["C"]

raw = bf.load("../01_data")
raw, stats, edges = bf.add_edges(raw)
past = bf.rolling_past(raw, stats, edges)
past["elo"] = bf.add_elo(raw)
games = bf.to_game_rows(raw, past, stats)

# home league of each team per year = its most common domestic league
dom = raw[~raw.league.isin(INTL)]
home = {(t, y): Counter(g).most_common(1)[0][0] for (t, y), g in dom.groupby(["team", "year"])["league"]}
side_team = raw.set_index(["gameid", "side"])["team"]
games["blue_home"] = [home.get((side_team[(g, "Blue")], y)) for g, y in zip(games.gameid, games.year)]
games["red_home"] = [home.get((side_team[(g, "Red")], y)) for g, y in zip(games.gameid, games.year)]

# region Elo, updated only on cross-region international games
reg, pre = {}, []
for _, r in games.sort_values("date").iterrows():
    hb, hr = r.blue_home, r.red_home
    rb, rr = reg.get(hb, 1500.0), reg.get(hr, 1500.0)
    pre.append((r.gameid, rb - rr if hb and hr else 0.0))
    if r.league in INTL and hb and hr and hb != hr:
        e = 1 / (1 + 10 ** ((rr - rb) / 400))
        reg[hb], reg[hr] = rb + 32 * (r.blue_win - e), rr - 32 * (r.blue_win - e)
games = games.merge(pd.DataFrame(pre, columns=["gameid", "f_region_elo"]), on="gameid")
games["cross_region"] = games.blue_home != games.red_home

F = ["f_elo", "f_dragon_edge", "f_grub_edge", "f_wcpm_edge", "f_firsttower", "f_vspm_edge"]
games[F] = games[F].fillna(0)
games = games.dropna(subset=["f_elo", "f_dragon_edge"])
intl = games[games.league.isin(INTL)]
rows = []
for (lg, yr), ev in intl.groupby(["league", "year"]):
    tr = games[(games.date >= "2022-01-01") & (games.date < ev.date.min())]
    if len(tr) < 1000:
        continue
    for name, feats in [("no region info", F), ("+ region Elo", F + ["f_region_elo"])]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[feats], tr.blue_win)
        p = m.predict_proba(ev[feats])[:, 1]
        for subset, mask in [("all", ev.index == ev.index), ("cross-region", ev.cross_region.values)]:
            if mask.sum():
                rows.append({"event": f"{lg} {yr}", "model": name, "games": int(mask.sum()), "subset": subset,
                             "logloss": log_loss(ev.blue_win[mask], p[mask], labels=[0, 1]),
                             "acc": accuracy_score(ev.blue_win[mask], p[mask] >= .5)})
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.pivot_table(index=["event", "games", "subset"], columns="model", values=["logloss", "acc"]).round(3).to_string())
tot = r.groupby(["subset", "model"]).apply(lambda t: pd.Series({"games": t.games.sum(),
      "logloss": np.average(t.logloss, weights=t.games), "acc": np.average(t.acc, weights=t.games)}))
print("\nPOOLED over events\n", tot.round(4).to_string())
print("\nregion Elo now:", {k: round(v) for k, v in sorted(reg.items(), key=lambda x: -x[1])})
dm = games[(~games.league.isin(INTL)) & (games.date >= "2026-01-01")]
print(f"\nfor comparison, domestic 2026 games: {len(dm)}")
r.to_csv("results/international_check.csv", index=False)
