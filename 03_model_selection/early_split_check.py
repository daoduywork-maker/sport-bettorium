"""
early_split_check.py  -  does the model struggle in the first weeks of a split / after roster changes? (run from 03_model_selection/)

Early-split = weeks since the league's block start (a block starts after a >= 21-day gap in that
league's schedule: new split, or return from a long break). Fresh roster = either team changed
its starting five within its last 10 games (counting the current one).
Walk-forward by year (train 2022..Y-1, test Y = 2024, 2025, 2026), pre-draft models only:
  M0 team Elo (pipeline log-loss features)      M1 player Elo instead of team Elo
  M2 M0 + experience                            M3 M1 + experience
Then, where odds exist (Apr-Oct 2026 matched series), model vs Pinnacle in early vs late weeks.
"""
import glob, json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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
PE = [f for f in BASE if f != "f_elo"] + ["f_player_elo"]
EXP = ["f_exp_games", "f_exp_major", "f_exp_playoffs", "f_exp_intl", "f_exp_years", "f_rookies"]
MODELS = {"M0 team Elo": BASE, "M1 player Elo": PE, "M2 team Elo + exp": BASE + EXP, "M3 player Elo + exp": PE + EXP}
df[sorted(set(sum(MODELS.values(), [])))] = df[sorted(set(sum(MODELS.values(), [])))].fillna(0)

# ---- weeks since block start (per league)
def add_weeks(d):
    d = d.sort_values("date").copy()
    d["weeks"] = np.nan
    for lg, g in d.groupby("league"):
        gap = g["date"].diff().dt.days.fillna(999)
        block_start = g["date"].where(gap >= 21).ffill()
        d.loc[g.index, "weeks"] = ((g["date"] - block_start).dt.days // 7 + 1).values
    return d
df = add_weeks(df)
df["wk_bucket"] = pd.cut(df.weeks, [0, 2, 4, 8, 99], labels=["wk 1-2", "wk 3-4", "wk 5-8", "wk 9+"])

# ---- fresh roster flag (from player rows)
parts = []
for f in sorted(glob.glob("../01_data/*LoL_esports_match_data*.csv")):
    d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ["gameid", "league", "date", "side", "position", "playerid", "playername", "teamid", "teamname"])
    parts.append(d[d.position.isin(["top", "jng", "mid", "bot", "sup"])])
p = pd.concat(parts); p["league"] = p.league.replace(bf.LEAGUE_MAP); p = p[p.league.isin(bf.LEAGUES)]
p["player"] = p.playerid.fillna(p.playername).astype(str); p["team"] = p.teamid.fillna(p.teamname).astype(str)
p["date"] = pd.to_datetime(p.date)
t = p.groupby(["gameid", "side", "team"]).agg(date=("date", "first"), lineup=("player", lambda s: frozenset(s))).reset_index().sort_values("date")
t["changed"] = (t.groupby("team")["lineup"].shift(1) != t["lineup"]) & t.groupby("team")["lineup"].shift(1).notna()
t["recent"] = t.groupby("team")["changed"].transform(lambda s: s.astype(int).rolling(10, min_periods=1).sum())
fresh = t.pivot_table(index="gameid", columns="side", values="recent", aggfunc="max")
df = df.merge(fresh.max(axis=1).rename("fresh_roster").reset_index(), on="gameid", how="left")
df["fresh"] = np.where(df.fresh_roster.fillna(0) > 0, "fresh roster", "stable roster")

# ---- walk-forward
losses = []
for Y in (2024, 2025, 2026):
    tr, te = df[(df.date >= "2022-01-01") & (df.date < f"{Y}-01-01")], df[df.date.dt.year == Y].copy()
    for name, F in MODELS.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
        pr = np.clip(m.predict_proba(te[F])[:, 1], 1e-6, 1 - 1e-6)
        te[name] = -(te.blue_win * np.log(pr) + (1 - te.blue_win) * np.log(1 - pr))
    losses.append(te)
L = pd.concat(losses)
cols = list(MODELS)
pd.set_option("display.width", 200)
print(f"games {len(L)} (2024-2026), blocks: weeks since block start; fresh roster share {(L.fresh=='fresh roster').mean():.0%}\n")
for key in ["wk_bucket", "fresh"]:
    t_ = L.groupby(key, observed=True)[cols].mean(); t_.insert(0, "n", L.groupby(key, observed=True).size())
    t_["M3 - M0"] = t_["M3 player Elo + exp"] - t_["M0 team Elo"]
    print(t_.round(4).to_string(), "\n")
rng = np.random.default_rng(0)
for lab, mask in [("weeks 1-4", L.weeks <= 4), ("fresh roster", L.fresh == "fresh roster"), ("everything else", (L.weeks > 4) & (L.fresh == "stable roster"))]:
    d = (L.loc[mask, "M3 player Elo + exp"] - L.loc[mask, "M0 team Elo"]).values
    b = [rng.choice(d, len(d)).mean() for _ in range(1000)]
    print(f"M3 - M0 on {lab:16s} n={mask.sum():5d}  {d.mean():+.4f}  95% CI [{np.percentile(b,2.5):+.4f}, {np.percentile(b,97.5):+.4f}]")
print("\nbase log-loss: weeks 1-4 %.4f | later %.4f" % (L.loc[L.weeks <= 4, 'M0 team Elo'].mean(), L.loc[L.weeks > 4, 'M0 team Elo'].mean()))
L.to_csv("results/early_split_losses.csv", index=False)

# ---- market, early vs late (matched Apr-Oct 2026 series)
m = pd.read_csv("../05_betting_simulation/results/matching.csv")
m["d"] = pd.to_datetime(m.date, utc=True).dt.tz_localize(None)
sched = df[["league", "date", "weeks"]].sort_values("date")
m = pd.merge_asof(m.sort_values("d"), sched.rename(columns={"date": "d"}), on="d", by="league", direction="nearest", tolerance=pd.Timedelta(days=4))
for mk, (lab, pc, yc) in {181: ("series", "p_series_a", "team_a_won"), 1847: ("map 1", "p_g1_a", "game1_a_won")}.items():
    x = m[m.market == mk].dropna(subset=[yc, "weeks"]).copy()
    q = (1 / x.odds_a) / (1 / x.odds_a + 1 / x.odds_b); y = x[yc]
    ll = lambda pp: -(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    x["diff"] = ll(x[f"{pc}_logloss"]) - ll(q)
    for nm, mask in [("weeks 1-3", x.weeks <= 3), ("weeks 4+", x.weeks > 3)]:
        z = x.loc[mask, "diff"]; b = [rng.choice(z.values, len(z)).mean() for _ in range(1000)] if len(z) else [np.nan]
        print(f"{lab:7s} {nm:10s} n={len(z):3d} model - market {z.mean():+.4f}  [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]")
