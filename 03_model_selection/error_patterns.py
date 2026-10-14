"""
error_patterns.py  -  when is our model wrong, and is there anything predictable about it?  (run from 03_model_selection/
after team_player_check.py and early_split_check.py)

Model = walk-forward pre-draft M3 (player Elo + experience), 6,602 games 2024-2026, residual r = blue_win - p.
 1. anatomy: confident misses, wrong rate by confidence
 2. scan: Spearman of every candidate variable with the signed residual (Bonferroni-corrected)
 3. second stage: can ANY variable fix the model out of sample? (logit p + candidates, train earlier years, test later)
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
G = pd.read_csv("results/team_player_residuals.csv", parse_dates=["date"])
for f in ["draft_features.csv", "early_features.csv", "class_features.csv", "synergy_features.csv"]:
    x = pd.read_csv("../02_features/" + f); G = G.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left", suffixes=("", "_dup"))
G = G[[c for c in G if not c.endswith("_dup")]]
E = pd.read_csv("results/early_split_losses.csv", usecols=["gameid", "weeks", "fresh"]); G = G.merge(E, on="gameid", how="left")

# ---- history-based variables from the full schedule (features.csv)
S = pd.read_csv("../02_features/features.csv", parse_dates=["date"], usecols=["gameid", "date", "blue_team", "red_team", "blue_win", "game"]).sort_values(["date", "gameid"])
last, streak, recent, h2h, lastpair, rows = {}, {}, {}, {}, {}, []
for r in S.itertuples():
    b, rd, d = r.blue_team, r.red_team, r.date
    rest = lambda t: min((d - last[t]).days, 30) if t in last else 30
    n14 = lambda t: sum(1 for x in recent.get(t, []) if (d - x).days <= 14)
    pair = frozenset((b, rd)); hh = h2h.get(pair, {"n": 0, b: 0, rd: 0})
    lp = lastpair.get(pair); prev = 0
    if lp and (d - lp[0]).days <= 1: prev = 1 if lp[1] == b else -1
    rows.append({"gameid": r.gameid, "rest_diff": rest(b) - rest(rd), "streak_diff": np.clip(streak.get(b, 0), -5, 5) - np.clip(streak.get(rd, 0), -5, 5),
                 "load14_diff": n14(b) - n14(rd), "h2h": (hh.get(b, 0) - hh.get(rd, 0)) / (hh["n"] + 4), "prev_game_winner": prev})
    for t, w in ((b, r.blue_win), (rd, 1 - r.blue_win)):
        s0 = streak.get(t, 0); streak[t] = (s0 + 1 if s0 > 0 else 1) if w else (s0 - 1 if s0 < 0 else -1)
        last[t] = d; recent.setdefault(t, []).append(d)
    hh = dict(hh); hh["n"] += 1; hh[b if r.blue_win else rd] = hh.get(b if r.blue_win else rd, 0) + 1; h2h[pair] = hh
    lastpair[pair] = (d, b if r.blue_win else rd)
G = G.merge(pd.DataFrame(rows), on="gameid", how="left")
first = G.groupby("patch").date.transform("min"); G["patch_age"] = (G.date - first).dt.days
G["game_no"] = G.game.clip(upper=5); G["playoffs"] = G.playoffs.astype(float)
for lg in ["LCK", "LPL", "LEC", "LCS", "LCP"]: G[f"lg_{lg}"] = (G.league == lg).astype(float)
G["pick"] = (G.p >= 0.5).astype(int); G["wrong"] = (G.pick != G.blue_win).astype(int)
G["conf"] = np.maximum(G.p, 1 - G.p)

print(f"{len(G)} games; wrong {G.wrong.mean():.1%}; the model's expected wrong rate {1-G.conf.mean():.1%}\n")
print("1. ANATOMY")
b = pd.cut(G.conf, [.5, .55, .6, .65, .7, .75, .8, 1.0])
print(G.groupby(b, observed=True).agg(games=("wrong", "size"), said=("conf", "mean"), wrong=("wrong", "mean")).assign(said_wrong=lambda t: 1 - t.said).round(3)[["games", "said_wrong", "wrong"]].to_string())
cm = G[(G.conf >= .75) & (G.wrong == 1)]
print(f"\nconfident misses (model >= 75 % and lost): {len(cm)} of {(G.conf>=.75).sum()} ({len(cm)/(G.conf>=.75).sum():.1%}; expected {1-G[G.conf>=.75].conf.mean():.1%})")
cf = G[G.conf >= .75].copy()
cf["team"] = np.where(cf.p >= .5, cf.blue_team, cf.red_team); cf["lost"] = cf.wrong
def zrow(g): 
    e = (1 - g.conf).sum(); v = (g.conf * (1 - g.conf)).sum(); return pd.Series({"backed": len(g), "lost": g.lost.sum(), "expected": e, "z": (g.lost.sum() - e) / np.sqrt(v)})
by_lg = cf.groupby("league").apply(zrow); by_team = cf.groupby("team").apply(zrow); by_team = by_team[by_team.backed >= 15].sort_values("z", ascending=False)
st = cf.groupby("playoffs").apply(zrow)
print("   confident picks (>= 75 %) lost vs expected, by league:\n", by_lg.round(2).to_string())
print("   playoffs vs regular season:\n", st.round(2).to_string())
print("   teams (>= 15 confident backings) lost most often vs expected:\n", by_team.head(5).round(2).to_string(), "\n   fewest:\n", by_team.tail(3).round(2).to_string())
print(f"   teams with |z| >= 2: {(by_team.z.abs() >= 2).sum()} of {len(by_team)}")

print("\n2. SCAN: score test of every variable against the residual, z = sum(r x) / sqrt(sum p(1-p) x^2), x standardised")
print("   (a rank correlation would be biased here: even a calibrated model shows a negative one for every strength variable)")
cand = [c for c in G.columns if c.startswith("f_") and c not in ("f_player_elo",)] + ["rest_diff", "streak_diff", "load14_diff", "h2h", "prev_game_winner", "patch_age", "weeks", "game_no", "playoffs"] + [c for c in G if c.startswith("lg_")]
res = []
for c in cand:
    x = G[c].astype(float).fillna(0)
    if x.nunique() < 3: continue
    x = (x - x.mean()) / x.std()
    z = (G.r * x).sum() / np.sqrt((G.p * (1 - G.p) * x ** 2).sum()); res.append({"variable": c, "z": z, "p": 2 * (1 - stats.norm.cdf(abs(z)))})
R = pd.DataFrame(res).sort_values("p"); thr = 0.05 / len(R)
print(R.head(10).round(4).to_string(index=False)); print(f"   {len(R)} variables tested; Bonferroni threshold p < {thr:.4f}; passing: {(R.p < thr).sum()}")

print("\n3. SECOND STAGE: logit P = a + b*logit(p) + c*X, trained on earlier years, scored on later ones (log-loss, lower is better)")
G["lp"] = np.log(G.p / (1 - G.p))
pre = [c for c in cand if not c.startswith(("f_syn", "f_class", "f_champ", "f_comfort", "f_lane", "f_synergy", "f_gd", "f_xp", "f_cs", "f_early"))]
drf = [c for c in cand if c.startswith(("f_syn", "f_class", "f_champ", "f_comfort", "f_lane", "f_synergy"))]
sets = {"only logit(p) (recalibration)": [], "+ pre-match variables": pre, "+ pre-match + draft": pre + drf}
for tr_m, te_m, name in [((G.date < "2025-01-01"), (G.date.dt.year == 2025), "train 2024 -> test 2025"), ((G.date < "2026-01-01"), (G.date.dt.year == 2026), "train 2024-25 -> test 2026")]:
    tr, te = G[tr_m], G[te_m]
    out = [("raw model p", log_loss(te.blue_win, te.p))]
    for nm, X in sets.items():
        cols = ["lp"] + X; A, B = tr[cols].astype(float).fillna(0), te[cols].astype(float).fillna(0)
        sc = StandardScaler().fit(A); m = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(A), tr.blue_win)
        out.append((nm, log_loss(te.blue_win, m.predict_proba(sc.transform(B))[:, 1])))
    print(f"  {name} ({len(te)} games): " + " | ".join(f"{a} {v:.4f}" for a, v in out))
    if name.endswith("2026"):
        cols = ["lp"] + pre + drf; A = tr[cols].astype(float).fillna(0); sc = StandardScaler().fit(A)
        co = pd.Series(LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(A), tr.blue_win).coef_[0], index=cols).drop("lp")
        print("   largest second-stage coefficients (per SD):", co.reindex(co.abs().sort_values(ascending=False).index).head(6).round(3).to_dict())

G.to_csv("results/error_patterns_games.csv", index=False)
