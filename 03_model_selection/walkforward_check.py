"""
walkforward_check.py  -  is a model trained on last year going stale?
Static      : train once on 2022-2025, predict every month of 2026.
Walk-forward: retrain at the start of every month on ALL games before it
              (time-decay weights, half-life 90 days), predict that month only.
Model: log-loss-branch LR (features + C from results/final_models.json). Run from 03_model_selection/.
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

fm = json.load(open("results/final_models.json"))["logloss"]
F, C, TAU = fm["features"], fm["hyperparams"]["C"], 90
df = pd.read_csv("../02_features/features.csv", parse_dates=["date"]).sort_values("date")
df[F] = df[F].fillna(0)
df["month"] = df.date.dt.to_period("M")

def fit(train, tau=None):
    age = (train.date.max() - train.date).dt.days
    w = np.ones(len(train)) if tau is None else 0.5 ** (age / tau)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return m.fit(train[F], train.blue_win, logisticregression__sample_weight=w)

static = fit(df[(df.date >= "2022-01-01") & (df.date < "2026-01-01")])
rows = []
for mo in sorted(df[df.date >= "2026-01-01"].month.unique()):
    te = df[df.month == mo]
    past = df[df.date < te.date.min()]
    wf_eq, wf_tau = fit(past), fit(past, TAU)
    rows.append({"month": str(mo), "games": len(te),
                 "static": log_loss(te.blue_win, static.predict_proba(te[F])[:, 1]),
                 "walkfwd": log_loss(te.blue_win, wf_eq.predict_proba(te[F])[:, 1]),
                 "walkfwd_tau90": log_loss(te.blue_win, wf_tau.predict_proba(te[F])[:, 1])})
r = pd.DataFrame(rows)
tot = {c: np.average(r[c], weights=r.games) for c in ["static", "walkfwd", "walkfwd_tau90"]}
print(r.round(4).to_string(index=False))
print("weighted mean:", {k: round(v, 4) for k, v in tot.items()})
bet = r[r.month >= "2026-06"]
print("betting period (Jun-Oct):", {c: round(np.average(bet[c], weights=bet.games), 4) for c in ["static", "walkfwd", "walkfwd_tau90"]})
r.to_csv("results/walkforward_check.csv", index=False)
