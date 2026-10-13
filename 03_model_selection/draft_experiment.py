"""
draft_experiment.py  -  which extra features add information? (run from 03_model_selection/)
Static LR (log-loss-branch C) trained on 2022-2025, log-loss on 2026 by period.
Pre-draft additions: early-game stats. Draft additions: champion meta, per-patch strength,
player comfort, lane matchups, synergy, champion-class matchups. Also reports the non-LPL subset (where early stats exist).
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

fm = json.load(open("results/final_models.json"))["logloss"]
BASE, C = fm["features"], fm["hyperparams"]["C"]
df = pd.read_csv("../02_features/features.csv", parse_dates=["date"])
for f in ["roster_features.csv", "draft_features.csv", "early_features.csv", "class_features.csv"]:
    x = pd.read_csv("../02_features/" + f)
    df = df.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left")
PE = [f for f in BASE if f != "f_elo"] + ["f_player_elo"]
EARLY = ["f_gd10", "f_gd15", "f_xpd15", "f_csd15", "f_early_available"]
D1 = ["f_champ_wr", "f_comfort_games", "f_comfort_wr"]
DALL = D1 + ["f_champ_wr_patch", "f_lane_matchup", "f_synergy"]
VARIANTS = {
    "A  pre-draft: player-Elo model": PE,
    "B  A + early-game stats": PE + EARLY,
    "C  A + draft (meta, comfort)": PE + D1,
    "D  C + per-patch strength": PE + D1 + ["f_champ_wr_patch"],
    "E  C + lane matchups": PE + D1 + ["f_lane_matchup"],
    "F  C + synergy": PE + D1 + ["f_synergy"],
    "G  A + all draft": PE + DALL,
    "H  everything (A + early + all draft)": PE + EARLY + DALL,
    "I  C + class lane matchup": PE + D1 + ["f_class_lane"],
    "J  C + class cross matchup": PE + D1 + ["f_class_cross"],
    "K  G + both class matchups": PE + DALL + ["f_class_lane", "f_class_cross"],
    "L  G with class lane instead of exact lane": PE + [f for f in DALL if f != "f_lane_matchup"] + ["f_class_lane"]}
allf = sorted(set(sum(VARIANTS.values(), [])))
df[allf] = df[allf].fillna(0)
tr, te = df[(df.date >= "2022-01-01") & (df.date < "2026-01-01")], df[df.date >= "2026-01-01"]
periods = {"Jan-Mar": te.date < "2026-04-01", "Apr-May": (te.date >= "2026-04-01") & (te.date < "2026-06-01"),
           "Jun-Oct": te.date >= "2026-06-01", "all 2026": te.date.notna(), "non-LPL 2026": te.league != "LPL"}
rows, coefs = [], {}
for name, F in VARIANTS.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
    p = m.predict_proba(te[F])[:, 1]
    rows.append({"variant": name, **{k: log_loss(te.blue_win[v], p[v]) for k, v in periods.items()},
                 "acc 2026": accuracy_score(te.blue_win, p >= .5)})
    coefs[name] = {k: round(v, 3) for k, v in zip(F, m[-1].coef_[0])}
r = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(r.round(4).to_string(index=False))
print("\ncoefficients per SD, K:", coefs["K  G + both class matchups"])
r.to_csv("results/draft_experiment.csv", index=False)
