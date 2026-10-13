"""
recency_experiment.py  -  does weighting recent matches more improve the probabilities?

Three knobs, all "recent counts more":
  A. feature averaging : plain mean of last 10 games  vs  exponential moving average (EWMA)
                         with half-life h games (weight of a game h games ago = 1/2)
  B. Elo update speed  : K = 16 / 24 / 40  (bigger K = rating reacts faster to recent results)
  C. training weights  : every training game weighted  w = 0.5 ** (age_days / tau)
                         (tau = half-life in days; recent games matter more when fitting the LR)

Model: logistic regression on the log-loss branch's features, C from the pipeline.
Selection: log-loss on the TEST months (Jan-May 2026); the betting period (Jun-Oct 2026) is
reported only for the chosen setting, so it stays an honest out-of-sample check.

Run from 03_model_selection/:  python recency_experiment.py   (~1-2 min)
Output: results/recency_grid.csv
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "../02_features")
import build_features as bf  # noqa: E402

HALFLIVES = [None, 3, 5, 10, 20]          # None = plain mean of last 10 (current setup)
ELO_KS = [16, 24, 40]
TAUS = [None, 90, 180, 365]               # None = all training games equal
FEATS = json.load(open("results/final_models.json"))["logloss"]["features"]
C = json.load(open("results/final_models.json"))["logloss"]["hyperparams"]["C"]


def rolling(df, stats, edges, halflife):
    e = edges.copy()
    e["team"] = df["team"].values
    g = e.groupby("team")
    out = pd.DataFrame(index=df.index)
    for c in stats:
        if halflife is None:
            f = lambda s: s.shift(1).rolling(bf.WINDOW, min_periods=bf.MIN_GAMES).mean()
        else:
            f = lambda s: s.shift(1).ewm(halflife=halflife, min_periods=bf.MIN_GAMES).mean()
        out[c] = g[c].transform(f)
    out["n_past_games"] = g.cumcount()
    return out


def main():
    print("loading raw data ...")
    df = bf.load("../01_data")
    df, stats, edges = bf.add_edges(df)
    elos = {}
    for k in ELO_KS:
        bf.ELO_K = k
        elos[k] = bf.add_elo(df)
        print(f"  Elo K={k} done")

    rows = []
    for h in HALFLIVES:
        past = rolling(df, stats, edges, h)
        for k in ELO_KS:
            past["elo"] = elos[k]
            games = bf.to_game_rows(df, past, stats)
            core = [c for c in games if c.startswith("f_") and c != "f_grub_edge"]
            games = games.dropna(subset=core)
            games[FEATS] = games[FEATS].fillna(0)
            tr = games[(games.date >= "2022-01-01") & (games.date < "2026-01-01")]
            te = games[(games.date >= "2026-01-01") & (games.date < "2026-06-01")]
            bt = games[games.date >= "2026-06-01"]
            for tau in TAUS:
                age = (tr.date.max() - tr.date).dt.days
                w = np.ones(len(tr)) if tau is None else 0.5 ** (age / tau)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                m.fit(tr[FEATS], tr.blue_win, logisticregression__sample_weight=w)
                pt = m.predict_proba(te[FEATS])[:, 1]
                # betting period: refit on train + test, like the pipeline does
                full = pd.concat([tr, te])
                age_f = (full.date.max() - full.date).dt.days
                w_f = np.ones(len(full)) if tau is None else 0.5 ** (age_f / tau)
                m.fit(full[FEATS], full.blue_win, logisticregression__sample_weight=w_f)
                pb = m.predict_proba(bt[FEATS])[:, 1]
                rows.append({"halflife": "last10 mean" if h is None else h, "elo_K": k,
                             "tau_days": "equal" if tau is None else tau,
                             "test_logloss": log_loss(te.blue_win, pt),
                             "test_acc": accuracy_score(te.blue_win, pt >= .5),
                             "bet_logloss": log_loss(bt.blue_win, pb),
                             "bet_acc": accuracy_score(bt.blue_win, pb >= .5)})
        print(f"  half-life {h} done")

    res = pd.DataFrame(rows).sort_values("test_logloss")
    res.to_csv("results/recency_grid.csv", index=False)
    base = res[(res.halflife == "last10 mean") & (res.elo_K == 24) & (res.tau_days == "equal")].iloc[0]
    best = res.iloc[0]
    pd.set_option("display.width", 200)
    print("\nTOP 10 by test log-loss")
    print(res.head(10).round(4).to_string(index=False))
    print("\nEFFECT OF EACH KNOB ALONE (others at current setting), test log-loss")
    for knob, cur in [("halflife", ("elo_K", 24, "tau_days", "equal")),
                      ("elo_K", ("halflife", "last10 mean", "tau_days", "equal")),
                      ("tau_days", ("halflife", "last10 mean", "elo_K", 24))]:
        t = res[(res[cur[0]] == cur[1]) & (res[cur[2]] == cur[3])].sort_values(knob, key=lambda s: s.astype(str))
        print(f"  {knob:9s}: " + "  ".join(f"{v}->{l:.4f}" for v, l in zip(t[knob], t.test_logloss)))
    print(f"\ncurrent setup : test {base.test_logloss:.4f}  bet-period logloss {base.bet_logloss:.4f}  acc {base.bet_acc:.3f}")
    print(f"best on test  : test {best.test_logloss:.4f}  bet-period logloss {best.bet_logloss:.4f}  acc {best.bet_acc:.3f}"
          f"   (half-life {best.halflife}, K {best.elo_K}, tau {best.tau_days})")
    print("Pinnacle series log-loss for reference: 0.575 (series, not games: not directly comparable)")


if __name__ == "__main__":
    main()
