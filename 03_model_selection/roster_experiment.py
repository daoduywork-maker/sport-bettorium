"""
roster_experiment.py  -  do roster-aware features help, especially early in the year?
Static LR (log-loss-branch C), trained on 2022-2025, scored by month on 2026.
Variants: baseline features | + roster changes & familiarity | team Elo -> player Elo | both.
Run from 03_model_selection/ after ../02_features/roster_features.py.
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
ro = pd.read_csv("../02_features/roster_features.csv")
df = df.merge(ro[["gameid", "f_roster_changes", "f_familiarity", "f_player_elo"]], on="gameid", how="left")
ROST = ["f_roster_changes", "f_familiarity"]
noelo = [f for f in BASE if f != "f_elo"]
VARIANTS = {"baseline (team Elo)": BASE,
            "+ roster changes/familiarity": BASE + ROST,
            "player Elo instead of team Elo": noelo + ["f_player_elo"],
            "player Elo + roster": noelo + ["f_player_elo"] + ROST,
            "team + player Elo + roster": BASE + ["f_player_elo"] + ROST}
allf = sorted(set(sum(VARIANTS.values(), [])))
df[allf] = df[allf].fillna(0)
tr, te = df[(df.date >= "2022-01-01") & (df.date < "2026-01-01")], df[df.date >= "2026-01-01"].copy()
periods = {"Jan-Mar": te.date < "2026-04-01", "Apr-May": (te.date >= "2026-04-01") & (te.date < "2026-06-01"),
           "Jun-Oct (bet)": te.date >= "2026-06-01", "all 2026": te.date.notna()}
rows = []
for name, F in VARIANTS.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
    p = m.predict_proba(te[F])[:, 1]
    row = {"variant": name}
    for per, mask in periods.items():
        row[per] = log_loss(te.blue_win[mask], p[mask])
    row["acc all 2026"] = accuracy_score(te.blue_win, p >= .5)
    rows.append(row)
    if name == "player Elo + roster":
        coef = dict(zip(F, m[-1].coef_[0]))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Log-loss on 2026 games (lower = better); model trained on 2022-2025 only\n")
print(r.round(4).to_string(index=False))
print("\ncoefficients (per SD), 'player Elo + roster':", {k: round(v, 3) for k, v in coef.items()})
r.to_csv("results/roster_experiment.csv", index=False)
