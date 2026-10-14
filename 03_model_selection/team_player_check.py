"""
team_player_check.py  -  is the league skewed toward a few teams / players, and does the model (or the market) misjudge them?
(run from 03_model_selection/)

Walk-forward pre-draft model M3 (player Elo + experience, log-loss-branch C), test years 2024, 2025, 2026, train 2022..Y-1.
Residual of a team in a game = (team won) - (model probability that it wins). Persistent positive residual = the model
keeps underrating that team/player. Persistence is checked by comparing 2024-25 with 2026 (split-half).
"""
import glob, json, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "../02_features")
import build_features as bf

fm = json.load(open("results/final_models.json"))["logloss"]
BASE, C = fm["features"], fm["hyperparams"]["C"]
df = pd.read_csv("../02_features/features.csv", parse_dates=["date"]).sort_values("date")
for f in ["roster_features.csv", "experience_features.csv"]:
    x = pd.read_csv("../02_features/" + f)
    df = df.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left")
EXP = ["f_exp_games", "f_exp_major", "f_exp_playoffs", "f_exp_intl", "f_exp_years", "f_rookies"]
F = [f for f in BASE if f != "f_elo"] + ["f_player_elo"] + EXP
df[F] = df[F].fillna(0)
out = []
for Y in (2024, 2025, 2026):
    tr, te = df[(df.date >= "2022-01-01") & (df.date < f"{Y}-01-01")], df[df.date.dt.year == Y].copy()
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
    te["p"] = m.predict_proba(te[F])[:, 1]
    out.append(te)
G = pd.concat(out)
G["r"] = G.blue_win - G.p
print(f"{len(G)} games 2024-2026\n")

# ---------- team table (from each team's own perspective)
tb = G[["gameid", "date", "league", "blue_team", "blue_win", "p"]].rename(columns={"blue_team": "team", "blue_win": "won", "p": "pw"})
tr_ = G[["gameid", "date", "league", "red_team", "blue_win", "p"]].rename(columns={"red_team": "team"})
tr_["won"], tr_["pw"] = 1 - tr_.pop("blue_win"), 1 - tr_.pop("p")
T = pd.concat([tb, tr_]); T["r"] = T.won - T.pw; T["period"] = np.where(T.date < "2026-01-01", "24-25", "26")
agg = T.groupby("team").agg(n=("won", "size"), win=("won", "mean"), pw=("pw", "mean"), res=("r", "mean"),
                            z=("r", lambda s: s.sum() / np.sqrt((T.loc[s.index, 'pw'] * (1 - T.loc[s.index, 'pw'])).sum())))
agg = agg[agg.n >= 40].sort_values("win", ascending=False)
print(f"SKEW: {len(agg)} teams with >= 40 games. Win rate of the 10 best teams: {agg.win.head(10).mean():.1%}; of the 10 worst: {agg.win.tail(10).mean():.1%}")
share = T.groupby("team").size().sort_values(ascending=False)
print(f"the 10 most-played teams are {share.head(10).sum()/share.sum():.0%} of all team-games\n")
print("Model residual by team: positive = team wins MORE than the model says (model underrates)")
print("most underrated:\n", agg.sort_values("z", ascending=False).head(6).round(3).to_string())
print("most overrated:\n", agg.sort_values("z").head(6).round(3).to_string())
print(f"teams with |z| > 2: {(agg.z.abs() > 2).sum()} of {len(agg)} (chance alone: about {0.046*len(agg):.1f})")
a, b = T[T.period == "24-25"].groupby("team").r.agg(["mean", "size"]), T[T.period == "26"].groupby("team").r.agg(["mean", "size"])
j = a.join(b, lsuffix="_a", rsuffix="_b").query("size_a >= 40 and size_b >= 15")
rho, pv = stats.spearmanr(j.mean_a, j.mean_b)
print(f"\nTEAM persistence (residual 2024-25 vs 2026, {len(j)} teams): Spearman {rho:+.2f} (p = {pv:.2f})")

# ---------- player residuals
parts = []
for f in sorted(glob.glob("../01_data/*LoL_esports_match_data*.csv")):
    d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ["gameid", "side", "position", "playerid", "playername"])
    parts.append(d[d.position.isin(["top", "jng", "mid", "bot", "sup"])])
P = pd.concat(parts); P["pid"] = P.playerid.fillna(P.playername).astype(str)
P = P.merge(G[["gameid", "date", "blue_win", "p"]], on="gameid")
P["won"] = np.where(P.side == "Blue", P.blue_win, 1 - P.blue_win); P["pw"] = np.where(P.side == "Blue", P.p, 1 - P.p)
P["r"] = P.won - P.pw; P["period"] = np.where(P.date < "2026-01-01", "24-25", "26")
pa = P.groupby("pid").agg(name=("playername", "last"), n=("won", "size"), win=("won", "mean"), res=("r", "mean"),
                          z=("r", lambda s: s.sum() / np.sqrt((P.loc[s.index, 'pw'] * (1 - P.loc[s.index, 'pw'])).sum())))
pa = pa[pa.n >= 60]
print(f"\nPLAYERS with >= 60 games: {len(pa)}; |z| > 2: {(pa.z.abs() > 2).sum()} (chance alone: about {0.046*len(pa):.0f})")
print("most underrated by the model:\n", pa.sort_values("z", ascending=False).head(6)[["name", "n", "win", "res", "z"]].round(3).to_string(index=False))
print("most overrated:\n", pa.sort_values("z").head(6)[["name", "n", "win", "res", "z"]].round(3).to_string(index=False))
a, b = P[P.period == "24-25"].groupby("pid").r.agg(["mean", "size"]), P[P.period == "26"].groupby("pid").r.agg(["mean", "size"])
j = a.join(b, lsuffix="_a", rsuffix="_b").query("size_a >= 60 and size_b >= 20")
rho, pv = stats.spearmanr(j.mean_a, j.mean_b)
print(f"PLAYER persistence (residual 2024-25 vs 2026, {len(j)} players): Spearman {rho:+.2f} (p = {pv:.2f})")
top = j.sort_values("mean_a", ascending=False)
q = len(top) // 5
print(f"players the model underrated most in 2024-25 (top fifth): residual in 2026 = {top.head(q).mean_b.mean():+.3f}; bottom fifth: {top.tail(q).mean_b.mean():+.3f}")

G.to_csv("results/team_player_residuals.csv", index=False)
