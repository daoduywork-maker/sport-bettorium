"""
experience_experiment.py  -  does player experience add information? (run from 03_model_selection/)
Same setup as draft_experiment.py: static LR (log-loss-branch C), train 2022-2025, log-loss on 2026.
Two bases: the pipeline's log-loss model (team Elo) and the pre-draft player-Elo model.
Experience is known before the draft, so it is a PRE-MATCH feature.
Output: results/experience_experiment.csv
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
for f in ["roster_features.csv", "experience_features.csv"]:
    x = pd.read_csv("../02_features/" + f)
    df = df.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left")
PE = [f for f in BASE if f != "f_elo"] + ["f_player_elo"]
EXP = ["f_exp_games", "f_exp_major", "f_exp_playoffs", "f_exp_intl", "f_exp_years", "f_rookies"]
V = {}
for bname, B in [("team Elo", BASE), ("player Elo", PE)]:
    V[f"{bname}: base"] = B
    for e in EXP:
        V[f"{bname}: + {e[2:]}"] = B + [e]
    V[f"{bname}: + all experience"] = B + EXP
V["player Elo: + all experience + roster"] = PE + EXP + ["f_roster_changes", "f_familiarity"]
allf = sorted(set(sum(V.values(), [])))
print("missing values before fill:", df[allf].isna().sum()[lambda s: s > 0].to_dict())
df[allf] = df[allf].fillna(0)
tr, te = df[(df.date >= "2022-01-01") & (df.date < "2026-01-01")], df[df.date >= "2026-01-01"]
per = {"Jan-Mar": te.date < "2026-04-01", "Apr-May": (te.date >= "2026-04-01") & (te.date < "2026-06-01"),
       "Jun-Oct": te.date >= "2026-06-01", "all 2026": te.date.notna()}
rows, coefs = [], {}
for name, F in V.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
    p = m.predict_proba(te[F])[:, 1]
    base = name.split(":")[0] + ": base"
    rows.append({"variant": name, **{k: log_loss(te.blue_win[v], p[v]) for k, v in per.items()}})
    coefs[name] = dict(zip(F, m[-1].coef_[0].round(3)))
r = pd.DataFrame(rows)
r["vs base"] = r["all 2026"] - r.groupby(r.variant.str.split(":").str[0])["all 2026"].transform("first")
pd.set_option("display.width", 200)
print(r.round(4).to_string(index=False))
print("\ncoefficients per SD (player Elo + all experience):",
      {k: v for k, v in coefs["player Elo: + all experience"].items() if "exp" in k or "rook" in k or "elo" in k})
print("\ncorrelation with Elo:")
print(df[["f_elo", "f_player_elo"] + EXP].corr().loc[EXP, ["f_elo", "f_player_elo"]].round(2).to_string())
r.to_csv("results/experience_experiment.csv", index=False)
