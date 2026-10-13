"""
draft_vs_market.py  -  is the draft model sharper than Pinnacle's map-1 price AFTER the draft?
Map-1 winner (market 1847) for matched series in Jun-Oct 2026.
Market price taken at several cutoffs before Oracle's game-1 timestamp
(-60 min = before draft, -2 min ~ draft finished). Model: player-Elo LR with / without draft
features, trained on games before 2026-06-01. Run from 05_betting_simulation/.
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import simulate as sim

fm = json.load(open("../03_model_selection/results/final_models.json"))["logloss"]
C = fm["hyperparams"]["C"]
df = pd.read_csv("../02_features/features.csv", parse_dates=["date"])
for f in ["roster_features.csv", "draft_features.csv"]:
    x = pd.read_csv("../02_features/" + f)
    df = df.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left")
PE = [f for f in fm["features"] if f != "f_elo"] + ["f_player_elo"]
DR = PE + ["f_champ_wr", "f_comfort_games", "f_comfort_wr"]
df[DR] = df[DR].fillna(0)
tr = df[df.date < "2026-06-01"]
preds = {}
for name, F in [("model pre-draft", PE), ("model + draft", DR)]:
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
    preds[name] = dict(zip(df.gameid, m.predict_proba(df[F])[:, 1]))

m = pd.read_csv("results/matching.csv", parse_dates=["date"])
t1 = pd.read_csv("../04_odds/odds_series.csv").drop_duplicates("fixtureId").set_index("fixtureId")["team1"]
m = m[m.market == 1847].drop_duplicates("fixtureId")
bt = df[(df.date >= "2026-06-01") & (df.game == 1)][["gameid", "date", "blue_team", "red_team", "blue_win"]]
rows = []
for _, s in m.iterrows():
    g = bt[(bt.blue_team == s.team_a) & (bt.red_team == s.team_b) & ((bt.date - s.date.tz_localize(None)).abs() < pd.Timedelta(minutes=1))]
    if len(g) != 1:
        continue
    g = g.iloc[0]
    row = {"y": g.blue_win, **{k: v[g.gameid] for k, v in preds.items()}}
    for mins in [60, 20, 10, 2]:
        pr = sim.closing_from_raw(s.fixtureId, "1847", s.date - pd.Timedelta(minutes=mins))
        if pr is not None:
            oa, ob = pr if sim.same_team(s.team_a, t1[s.fixtureId]) else (pr[1], pr[0])
            row[f"market -{mins}min"] = (1 / oa) / (1 / oa + 1 / ob)
    rows.append(row)
r = pd.DataFrame(rows).dropna()
ll = lambda p: -np.mean(r.y * np.log(p) + (1 - r.y) * np.log(1 - p))
print(f"{len(r)} map-1 games with all prices")
for c in [c for c in r if c != "y"]:
    print(f"  {c:18s} log-loss {ll(r[c]):.4f}")
