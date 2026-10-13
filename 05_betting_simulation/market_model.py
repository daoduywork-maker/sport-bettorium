"""
market_model.py  -  does our model add information BEYOND Pinnacle's price?  (map-1 winner market)

Stacking: on map-1 games we fit
    logit P(blue wins) = a + b * logit(q_pinnacle) + c * logit(p_ours)
on Jan-May 2026 and test on Jun-Oct 2026.
  - if c ~ 0, our model only repeats what the market already knows
  - if c > 0 and the stacked model beats the market's log-loss out of sample, there is extra information
Two timings:
  pre-draft  : Pinnacle price 60 min before game 1, our pre-draft model (player Elo, no draft)
  post-draft : Pinnacle price 2 min before game 1, our model + all draft features
Our probabilities are always out of sample: model for Jan-May trained on 2024-2025, for Jun-Oct
trained on everything before June. Finally a fixed-stake value-bet simulation on Jun-Oct with
the stacked probability (bootstrap 95 % interval).
Run from 05_betting_simulation/ after the Jan-May odds download.
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
PRE = [f for f in fm["features"] if f != "f_elo"] + ["f_player_elo"]
POST = PRE + ["f_champ_wr", "f_comfort_games", "f_comfort_wr", "f_champ_wr_patch", "f_lane_matchup", "f_synergy"]
df[POST] = df[POST].fillna(0)
logit = lambda p: np.log(p / (1 - p))


def ours(F):
    """Out-of-sample P(blue) for every 2026 game."""
    out = {}
    for lo, hi, cut in [("2026-01-01", "2026-06-01", "2026-01-01"), ("2026-06-01", "2027-01-01", "2026-06-01")]:
        tr = df[df.date < cut]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[F], tr.blue_win)
        te = df[(df.date >= lo) & (df.date < hi)]
        out.update(dict(zip(te.gameid, m.predict_proba(te[F])[:, 1])))
    return out


p_pre, p_post = ours(PRE), ours(POST)
odds = pd.read_csv("../04_odds/odds_series.csv")
odds = odds[odds.bookmaker == "pinnacle"].drop_duplicates("fixtureId")
odds["start"] = pd.to_datetime(odds["start"], utc=True)
g1 = df[(df.date >= "2026-01-01") & (df.game == 1)]
rows, used = [], set()
for _, g in g1.iterrows():
    t = g.date.tz_localize("UTC")
    cand = odds[(odds.start - t).abs() <= pd.Timedelta(hours=36)]
    for _, o in cand.iterrows():
        if o.fixtureId in used:
            continue
        if sim.same_team(g.blue_team, o.team1) and sim.same_team(g.red_team, o.team2):
            flip = False
        elif sim.same_team(g.blue_team, o.team2) and sim.same_team(g.red_team, o.team1):
            flip = True
        else:
            continue
        used.add(o.fixtureId)
        row = {"date": g.date, "league": g.league, "y": g.blue_win,
               "p_pre": p_pre[g.gameid], "p_post": p_post[g.gameid]}
        for tag, mins in [("q_pre", 60), ("q_post", 2)]:
            pr = sim.closing_from_raw(o.fixtureId, "1847", t - pd.Timedelta(minutes=mins))
            if pr is not None:
                ob, orr = (pr[1], pr[0]) if flip else pr
                row[tag] = (1 / ob) / (1 / ob + 1 / orr)
                row["o" + tag[1:]] = (ob, orr)
        rows.append(row)
        break
m = pd.DataFrame(rows).dropna()
fit_set, test_set = m[m.date < "2026-06-01"], m[m.date >= "2026-06-01"]
print(f"map-1 games with Pinnacle prices: fit (Jan-May) {len(fit_set)}, test (Jun-Oct) {len(test_set)}\n")
ll = lambda y, p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
rng = np.random.default_rng(0)
for timing, q, p in [("pre-draft", "q_pre", "p_pre"), ("post-draft", "q_post", "p_post")]:
    X = lambda d: np.column_stack([logit(d[q]), logit(d[p])])
    st = LogisticRegression(C=1e6).fit(X(fit_set), fit_set.y)
    ps = st.predict_proba(X(test_set))[:, 1]
    # bootstrap the log-loss difference stacked - market on the test set
    diff = [ll(test_set.y.values[i], ps[i]) - ll(test_set.y.values[i], test_set[q].values[i])
            for i in (rng.integers(0, len(test_set), len(test_set)) for _ in range(2000))]
    print(f"{timing.upper()}  weights: market {st.coef_[0][0]:.2f}, ours {st.coef_[0][1]:.2f}, intercept {st.intercept_[0]:.2f}")
    print(f"  test log-loss  market {ll(test_set.y, test_set[q]):.4f}   ours {ll(test_set.y, test_set[p]):.4f}"
          f"   stacked {ll(test_set.y, ps):.4f}   (stacked - market 95% CI {np.percentile(diff, 2.5):+.4f} .. {np.percentile(diff, 97.5):+.4f})")
    # fixed-stake value bets on the test set with the stacked probability
    o = test_set["o" + q[1:]].values
    prof, nb = [], 0
    for pb, (ob, orr), y in zip(ps, o, test_set.y):
        if pb > 1 / ob:
            prof.append(100 * (ob - 1) if y == 1 else -100); nb += 1
        elif 1 - pb > 1 / orr:
            prof.append(100 * (orr - 1) if y == 0 else -100); nb += 1
    prof = np.array(prof)
    boot = [prof[rng.integers(0, len(prof), len(prof))].sum() / (100 * len(prof)) for _ in range(2000)] if len(prof) else [0]
    print(f"  value bets: {nb}, yield per $ staked {prof.sum() / (100 * max(nb, 1)):+.1%}"
          f"  (95% CI {np.percentile(boot, 2.5):+.1%} .. {np.percentile(boot, 97.5):+.1%})\n")
m.to_csv("results/market_model_games.csv", index=False)
