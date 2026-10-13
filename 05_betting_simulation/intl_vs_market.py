"""
intl_vs_market.py  -  model vs Pinnacle at international events (MSI 2026, EWC 2026), cross-region vs same-region
(run from 05_betting_simulation/, after 04_odds/fetch_odds.py with the international tournament ids)

Model: log-loss-branch features (final_models.json) with and without REGION Elo, trained on every game since 2022 played
BEFORE the event starts (domestic + earlier internationals). Only series where both teams have >= 5 past games.
Series probability from the average of the two side-orientations, as in series_probs.py. Matching and closing-price
rules are those of simulate.py (price at least CUTOFF_MIN minutes before game 1).
Output: results/intl_vs_market.csv
"""
import json, sys
from collections import Counter
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "../02_features"); sys.path.insert(0, "../03_model_selection")
import build_features as bf
import simulate as sim
from series_probs import series_prob, assign_series

INTL = {"MSI", "WLDs", "FST", "EWC"}
bf.LEAGUES = bf.LEAGUES | INTL
fm = json.load(open("../03_model_selection/results/final_models.json"))["logloss"]
F, C = fm["features"], fm["hyperparams"]["C"]

raw = bf.load("../01_data")
raw, stats, edges = bf.add_edges(raw)
past = bf.rolling_past(raw, stats, edges)
past["elo"] = bf.add_elo(raw)
games = bf.to_game_rows(raw, past, stats)
dom = raw[~raw.league.isin(INTL)]
home = {(t, y): Counter(g).most_common(1)[0][0] for (t, y), g in dom.groupby(["team", "year"])["league"]}
side_team = raw.set_index(["gameid", "side"])["team"]
games["blue_home"] = [home.get((side_team[(g, "Blue")], y)) for g, y in zip(games.gameid, games.year)]
games["red_home"] = [home.get((side_team[(g, "Red")], y)) for g, y in zip(games.gameid, games.year)]
reg, pre = {}, []
for _, r in games.sort_values("date").iterrows():
    hb, hr = r.blue_home, r.red_home
    rb, rr = reg.get(hb, 1500.0), reg.get(hr, 1500.0)
    pre.append((r.gameid, rb - rr if hb and hr else 0.0))
    if r.league in INTL and hb and hr and hb != hr:
        e = 1 / (1 + 10 ** ((rr - rb) / 400))
        reg[hb], reg[hr] = rb + 32 * (r.blue_win - e), rr - 32 * (r.blue_win - e)
games = games.merge(pd.DataFrame(pre, columns=["gameid", "f_region_elo"]), on="gameid")
games["cross_region"] = games.blue_home.notna() & games.red_home.notna() & (games.blue_home != games.red_home)
games["known"] = (games.blue_n_past >= 5) & (games.red_n_past >= 5)
games[F + ["f_region_elo"]] = games[F + ["f_region_elo"]].fillna(0)

rows = []
for (lg, yr), ev in games[games.league.isin(INTL) & (games.year == 2026)].groupby(["league", "year"]):
    if lg not in ("MSI", "EWC"):
        continue                                    # no Pinnacle history for FST / WLDs
    tr = games[(games.date >= "2022-01-01") & (games.date < ev.date.min()) & games.known]
    ev = assign_series(ev.sort_values("date"))
    fitted = {}
    for name, feats in [("base", F), ("region", F + ["f_region_elo"])]:
        fitted[name] = (feats, make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(tr[feats], tr.blue_win))
    for sid, g in ev.groupby("series_id"):
        g = g.sort_values("date"); first = g.iloc[0]
        if not first.known:
            continue
        ta, tb = first.blue_team, first.red_team
        wa = int(((g.blue_team == ta) & (g.blue_win == 1)).sum() + ((g.red_team == ta) & (g.blue_win == 0)).sum())
        bo = 2 * max(wa, len(g) - wa) - 1
        row = {"series_id": f"{lg}{sid}", "date": first.date, "league": lg, "event": f"{lg} {yr}", "team_a": ta, "team_b": tb,
               "best_of": bo, "first_game_no": int(first.game), "team_a_won": int(wa > len(g) - wa), "cross_region": bool(first.cross_region),
               "game1_a_won": int(first.blue_win) if first.game == 1 else np.nan}
        for name, (feats, m) in fitted.items():
            x = first[feats].to_frame().T.astype(float)
            pb = m.predict_proba(x)[0, 1]
            pr = 1 - m.predict_proba(-x)[0, 1]
            row[f"p_g1_a_{name}"] = pb
            row[f"p_series_a_{name}"] = series_prob((pb + pr) / 2, bo)
        rows.append(row)
S = pd.DataFrame(rows)
print(f"Oracle international series 2026 with both teams known: {len(S)} "
      f"(MSI {sum(S.league=='MSI')}, EWC {sum(S.league=='EWC')}); cross-region {S.cross_region.sum()}")

odds = pd.read_csv("../04_odds/odds_series.csv")
odds = odds[odds.tournament.isin(["Mid-Season Invitational", "Esports World Cup"])]
m = sim.match(S, odds)
n_m = m.series_id.nunique() if len(m) else 0
print(f"matched to Pinnacle: {n_m} series")
miss = S[~S.series_id.isin(m.series_id)] if len(m) else S
if len(miss):
    print("unmatched examples:", [(r.team_a, r.team_b, str(r.date)[:10]) for r in miss.head(8).itertuples()])
m.to_csv("results/intl_vs_market_matched.csv", index=False)

ll = lambda y, p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
rng = np.random.default_rng(0)
def ci(v):
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return np.percentile(b, [2.5, 97.5])
out = []
print("\nlog-loss: MODEL - PINNACLE (negative = model better), 95% bootstrap interval")
for mk, (lab, pc, yc) in sim.MARKETS.items():
    d = m[m.market == mk].dropna(subset=[yc]).copy()
    d["q"] = (1 / d.odds_a) / (1 / d.odds_a + 1 / d.odds_b)
    d["marg"] = 1 / d.odds_a + 1 / d.odds_b - 1
    for variant in ("base", "region"):
        d[f"diff_{variant}"] = ll(d[yc], d[f"{pc}_{variant}"]) - ll(d[yc], d.q)
    for nm, x in [("all", d), ("cross-region", d[d.cross_region]), ("same-region", d[~d.cross_region]),
                  ("MSI", d[d.league == "MSI"]), ("EWC", d[d.league == "EWC"])]:
        if len(x) < 5:
            continue
        for variant in ("base", "region"):
            v = x[f"diff_{variant}"].values; lo, hi = ci(v)
            out.append({"market": lab, "subset": nm, "model": variant, "n": len(x), "diff": v.mean(), "lo": lo, "hi": hi,
                        "market_logloss": ll(x[yc], x.q).mean(), "model_logloss": ll(x[yc], x[f'{pc}_{variant}']).mean(),
                        "margin": x.marg.mean()})
R = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
R.to_csv("results/intl_vs_market.csv", index=False)

print("\nfixed-stake 100 value bets with the region model (bet when p - 1/odds > delta), yield per unit staked")
for mk, (lab, pc, yc) in sim.MARKETS.items():
    d = m[m.market == mk].dropna(subset=[yc])
    for delta in (0.0, 0.05):
        pr = []
        for r in d.itertuples():
            p = getattr(r, f"{pc}_region"); y = getattr(r, yc)
            for pp, o, won in [(p, r.odds_a, y == 1), (1 - p, r.odds_b, y == 0)]:
                if pp - 1 / o > delta:
                    pr.append((100 * (o - 1) if won else -100, r.cross_region))
        if pr:
            v = np.array([x[0] for x in pr]) / 100
            print(f"  {lab:7s} delta {delta:.2f}: {len(v):3d} bets, yield {v.mean():+.1%}  95% [{ci(v)[0]:+.0%}, {ci(v)[1]:+.0%}]; "
                  f"cross-region bets {sum(x[1] for x in pr)}")
