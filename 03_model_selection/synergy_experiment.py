"""
synergy_experiment.py  -  which champion pairs matter? (run from 03_model_selection/)
Same setup as draft_experiment.py: static LR (log-loss-branch C), train 2022-2025, log-loss on 2026.
Base = player-Elo model + champion meta + comfort (variant C there). Each pair added alone, then together.
Output: results/synergy_experiment.csv
"""
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

fm = json.load(open("results/final_models.json"))["logloss"]
BASE, C = fm["features"], fm["hyperparams"]["C"]
df = pd.read_csv("../02_features/features.csv", parse_dates=["date"])
for f in ["roster_features.csv", "draft_features.csv", "synergy_features.csv"]:
    x = pd.read_csv("../02_features/" + f)
    df = df.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left")
C0 = [f for f in BASE if f != "f_elo"] + ["f_player_elo", "f_champ_wr", "f_comfort_games", "f_comfort_wr"]
PAIRS = ["f_syn_bot_sup", "f_syn_jng_mid", "f_syn_jng_sup", "f_syn_mid_sup", "f_syn_jng_top"]
TRIO = ["f_syn_trio", "f_syn_trio_cls"]
V = {"base (meta + comfort)": C0, "+ old f_synergy (3 duos averaged)": C0 + ["f_synergy"]}
V.update({f"+ {p[6:]}": C0 + [p] for p in PAIRS + TRIO})
V["+ all 5 pairs"] = C0 + PAIRS
V["+ all 5 pairs + trios"] = C0 + PAIRS + TRIO
V["+ all + lane matchup + per-patch"] = C0 + PAIRS + TRIO + ["f_lane_matchup", "f_champ_wr_patch"]
allf = sorted(set(sum(V.values(), [])))
df[allf] = df[allf].fillna(0)
tr, te = df[(df.date >= "2022-01-01") & (df.date < "2026-01-01")], df[df.date >= "2026-01-01"]
per = {"Jan-Mar": te.date < "2026-04-01", "Apr-May": (te.date >= "2026-04-01") & (te.date < "2026-06-01"),
       "Jun-Oct": te.date >= "2026-06-01", "all 2026": te.date.notna()}
rows, coefs = [], {}
for name, F in V.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
    p = m.predict_proba(te[F])[:, 1]
    rows.append({"variant": name, **{k: log_loss(te.blue_win[v], p[v]) for k, v in per.items()}})
    coefs[name] = dict(zip(F, m[-1].coef_[0].round(3)))
r = pd.DataFrame(rows)
r["vs base"] = r["all 2026"] - r["all 2026"].iloc[0]
pd.set_option("display.width", 200)
print(r.round(4).to_string(index=False))
print("\ncoefficients per SD (all pairs + trios):",
      {k: v for k, v in coefs["+ all 5 pairs + trios"].items() if "syn" in k})
r.to_csv("results/synergy_experiment.csv", index=False)
