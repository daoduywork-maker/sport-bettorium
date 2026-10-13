"""
simulate.py  -  betting simulation, accuracy-driven vs calibration-driven model (Walsh & Joshi 2024, Table 2)

Run from 05_betting_simulation/ after 03_model_selection/series_probs.py and 04_odds/fetch_odds.py.

For every series in Jun-Oct 2026 that has Pinnacle closing odds:
  market 181  (series winner): model probability = P_series(team A)
  market 1847 (map-1 winner) : model probability = P(team A wins game 1 on its actual side)
Strategy: bet on a side when  p_model > 1 / odds   (a value bet; at most one side can qualify)
Rules   : fixed   -> stake 100 per bet
          kelly8  -> stake = bankroll * k*/8,  k* = (p b - (1 - p)) / b,  b = odds - 1   (true Kelly)
Bankroll starts at 10,000; bets are settled in date order.

Outputs: results/summary.csv, results/bets_<branch>_<market>_<rule>.csv, results/matching.csv,
         ../assets/bankroll.png, ../assets/edge_threshold.png
Uncertainty: 1,000 bootstrap resamples of the series (with replacement), whole simulation rerun each time.
"""
import difflib
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOOKMAKER = "pinnacle"
RAW_DIR = "../04_odds/odds_raw"
CUTOFF_MIN = 60     # minutes before Oracle's game-1 timestamp
BANKROLL0 = 10_000
FIXED_STAKE = 100
KELLY_FRACTION = 1 / 8
N_BOOT = 1000
BRANCHES = ["accuracy", "calibration", "logloss"]
DELTAS = [0.0, 0.02, 0.05, 0.10]   # bet only if p_model - 1/odds > delta
MARKETS = {"181": ("series", "p_series_a", "team_a_won"),
           "1847": ("map 1", "p_g1_a", "game1_a_won")}

# Oracle's Elixir name -> OddsPapi name, only where normalised names differ too much
ALIASES = {"DN SOOPers": "Soopers", "Kiwoom DRX": "DRX", "HANJIN BRION": "Brion",
           "BNK FEARX": "FEARX", "Disguised": "Disguisedtoast", "LØS": "Los",
           "9z Globant": "9Z Team", "Vivo Keyd Stars": "Keyd Stars",
           "KaBuM! Ilha das Lendas": "Kabum Idl", "Dplus Kia": "Dplus"}


def norm(name):
    name = ALIASES.get(name, name).lower().replace("ø", "o")
    name = re.sub(r"[^a-z0-9]", "", name)
    for w in ["esports", "gaming", "team"]:
        name = name.replace(w, "")
    return name


def same_team(a, b):
    na, nb = norm(a), norm(b)
    return na == nb or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.8


# ------------------------------------------------------------------ closing odds
def closing_from_raw(fid, market, cutoff):
    """Last Pinnacle price per outcome strictly before `cutoff` (from the cached raw history).

    OddsPapi's fixture start time is sometimes wrong (rescheduled / late), so in-play prices
    such as 1.02 / 16.0 leaked into odds_series.csv. The cutoff used here is the real start of
    game 1 from Oracle's Elixir minus CUTOFF_MIN minutes, so only pre-match prices are used.
    """
    path = os.path.join(RAW_DIR, f"hist_{fid}.json")
    if not os.path.exists(path):
        return None
    hist = json.load(open(path))
    if not isinstance(hist, dict):     # empty reply: no odds for this fixture
        return None
    mk = (((hist.get("bookmakers") or {}).get(BOOKMAKER) or {}).get("markets") or {}).get(market)
    if not mk:
        return None
    out = {}
    for oid, od in mk.get("outcomes", {}).items():
        entries = []
        for v in (od.get("players") or {}).values():
            entries += v if isinstance(v, list) else [v]
        entries = [e for e in entries if e.get("price") and
                   pd.Timestamp(e["createdAt"]) < cutoff]
        if entries:
            out[oid] = max(entries, key=lambda e: pd.Timestamp(e["createdAt"]))["price"]
    if len(out) != 2:
        return None
    (_, p1), (_, p2) = sorted(out.items())
    margin = 1 / p1 + 1 / p2 - 1
    if not (0.0 <= margin <= 0.15):          # not a sane pre-match two-way price
        return None
    return p1, p2


# ------------------------------------------------------------------ matching
def match(series, odds):
    odds = odds[odds["bookmaker"] == BOOKMAKER].copy()
    odds["start"] = pd.to_datetime(odds["start"], utc=True)
    series = series.copy()
    series["date"] = pd.to_datetime(series["date"], utc=True)
    rows, used = [], set()
    for _, s in series.iterrows():
        cand = odds[(odds["start"] - s["date"]).abs() <= pd.Timedelta(hours=36)]
        for fid, g in cand.groupby("fixtureId"):
            if fid in used:
                continue
            o = g.iloc[0]
            if same_team(s.team_a, o.team1) and same_team(s.team_b, o.team2):
                flip = False
            elif same_team(s.team_a, o.team2) and same_team(s.team_b, o.team1):
                flip = True
            else:
                continue
            used.add(fid)
            if s["first_game_no"] != 1:       # game 1 missing -> no clean pre-match time
                break
            # Pinnacle goes live during the draft and between games, so stop 60 min before game 1
            cutoff = s["date"] - pd.Timedelta(minutes=CUTOFF_MIN)
            for market in MARKETS:
                pr = closing_from_raw(fid, market, cutoff)
                if pr is None:
                    continue
                oa, ob = (pr[1], pr[0]) if flip else pr
                rows.append({**s.to_dict(), "fixtureId": fid, "market": market,
                             "odds_a": oa, "odds_b": ob})
            break
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ simulation
def simulate(df, branch, market, rule, delta=0.0):
    """df: one row per series for this market, sorted by date. Returns bets table + bankroll path."""
    _, pcol, ycol = MARKETS[market]
    bank, bets = BANKROLL0, []
    cols = zip(df["date"].values, df["team_a"].values, df["team_b"].values,
               df[f"{pcol}_{branch}"].values, df[ycol].values,
               df["odds_a"].values, df["odds_b"].values)
    for date, ta, tb, p_a, y, odds_a, odds_b in cols:
        if np.isnan(p_a) or np.isnan(y):
            continue
        for side, p, o, won in [("A", p_a, odds_a, y == 1), ("B", 1 - p_a, odds_b, y == 0)]:
            if p - 1 / o > delta:                          # value bet with minimum edge
                if rule == "fixed":
                    stake = FIXED_STAKE
                else:
                    b = o - 1
                    k = (p * b - (1 - p)) / b
                    stake = bank * KELLY_FRACTION * k
                stake = min(stake, bank)
                if stake <= 0:
                    continue
                profit = stake * (o - 1) if won else -stake
                bank += profit
                bets.append({"date": date, "team_a": ta, "team_b": tb,
                             "side": side, "p_model": p, "p_implied": 1 / o, "odds": o,
                             "stake": stake, "won": int(won), "profit": profit, "bankroll": bank})
    return pd.DataFrame(bets), bank


def summarise(bets, bank, n_series):
    staked = bets["stake"].sum() if len(bets) else 0
    return {"series_available": n_series, "bets": len(bets),
            "pct_series_bet": len(bets) / n_series if n_series else np.nan,
            "pct_bets_won": bets["won"].mean() if len(bets) else np.nan,
            "final_bankroll": bank, "ROI_bankroll": bank / BANKROLL0 - 1,
            "yield_per_staked": (bank - BANKROLL0) / staked if staked else np.nan}


def main():
    os.makedirs("results", exist_ok=True)
    series = pd.read_csv("../03_model_selection/results/series_bet.csv")
    odds = pd.read_csv("../04_odds/odds_series.csv")
    m = match(series, odds)
    # consistency check: the series favourite and the map-1 favourite must not be opposite teams
    # by a wide margin (seen in a few fixtures where OddsPapi swapped the series outcomes)
    q = (1 / m.odds_a) / (1 / m.odds_a + 1 / m.odds_b)
    piv = pd.DataFrame({"fid": m.fixtureId, "market": m.market, "q": q}).pivot_table(
        index="fid", columns="market", values="q")
    if {"181", "1847"} <= set(piv.columns):
        bad = piv[((piv["181"] - .5) * (piv["1847"] - .5) < 0) & ((piv["181"] - piv["1847"]).abs() > .3)].index
        if len(bad):
            print(f"dropped {len(bad)} fixtures whose series and map-1 favourites contradict: {list(bad)}")
        m = m[~m.fixtureId.isin(bad)]
    m.to_csv("results/matching.csv", index=False)
    n_matched = m["series_id"].nunique()
    print(f"matched {n_matched} / {len(series)} Oracle series to {BOOKMAKER} odds "
          f"({odds[odds.bookmaker == BOOKMAKER].fixtureId.nunique()} fixtures with odds)\n")

    # --- how good are the probabilities vs the market? (log-loss on matched series)
    print("Log-loss on matched series (lower = better); market = Pinnacle odds with margin removed")
    for market, (label, pcol, ycol) in MARKETS.items():
        d = m[m.market == market].dropna(subset=[ycol, f"{pcol}_accuracy"])
        y = d[ycol].values
        q = (1 / d.odds_a) / (1 / d.odds_a + 1 / d.odds_b)          # de-vigged market prob
        ll = lambda p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        margin = (1 / d.odds_a + 1 / d.odds_b - 1).mean()
        models = "  ".join(f"{b}-model {ll(d[f'{pcol}_{b}']):.4f}" for b in BRANCHES)
        print(f"  {label:6s} n={len(d):3d}  market {ll(q):.4f}  {models}   (avg margin {margin:.1%})")

    # --- simulations + bootstrap
    rng = np.random.default_rng(0)
    rows, paths = [], {}
    for market, (label, pcol, ycol) in MARKETS.items():
        d = m[m.market == market].sort_values("date").reset_index(drop=True)
        boots = [np.sort(rng.integers(0, len(d), len(d))) for _ in range(N_BOOT)]  # same resamples for all
        for branch in BRANCHES:
            for rule in ["fixed", "kelly8"]:
                for delta in DELTAS:
                    bets, bank = simulate(d, branch, market, rule, delta)
                    if delta == 0:
                        bets.to_csv(f"results/bets_{branch}_{market}_{rule}.csv", index=False)
                        paths[(label, branch, rule)] = bets
                    row = {"market": label, "branch": branch, "rule": rule, "delta": delta,
                           **summarise(bets, bank, len(d))}
                    boot = [simulate(d.iloc[idx], branch, market, rule, delta)[1] / BANKROLL0 - 1
                            for idx in boots]
                    row["ROI_CI95_low"], row["ROI_CI95_high"] = np.percentile(boot, [2.5, 97.5])
                    row["P(ROI>0)"] = np.mean(np.array(boot) > 0)
                    rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv("results/summary.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["market", "branch", "rule", "delta", "bets", "pct_bets_won", "ROI_bankroll",
            "ROI_CI95_low", "ROI_CI95_high", "P(ROI>0)"]
    print("\nALL VALUE BETS (delta = 0)")
    print(summ[summ.delta == 0][cols].round(3).to_string(index=False))
    print("\nEDGE THRESHOLD, fixed stake")
    print(summ[summ.rule == "fixed"][cols].round(3).to_string(index=False))

    # --- figure
    plt.style.use("../assets/figstyle.mplstyle")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    colors = {"accuracy": "#1f77b4", "calibration": "#d62728", "logloss": "#2ca02c"}
    for i, label in enumerate(["series", "map 1"]):
        for j, rule in enumerate(["fixed", "kelly8"]):
            ax = axes[i, j]
            ax.axhline(BANKROLL0, ls="--", c="grey", lw=1)
            for branch in BRANCHES:
                b = paths[(label, branch, rule)]
                if len(b):
                    ax.plot(range(len(b) + 1), [BANKROLL0] + list(b["bankroll"]),
                            c=colors[branch], label=f"{branch}-driven")
            ax.set_title(f"{label} winner, {'fixed $100' if rule == 'fixed' else 'eighth-Kelly'}")
            if j == 0:
                ax.set_ylabel("bankroll")
            if i == 1:
                ax.set_xlabel("bet number")
    axes[0, 0].legend()
    fig.suptitle("Pinnacle closing odds, Jun–Oct 2026")
    fig.savefig("../assets/bankroll.png")

    # ROI vs minimum edge (fixed stake), with bootstrap 95 % intervals
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, label in zip(axes, ["series", "map 1"]):
        ax.axhline(0, ls="--", c="grey", lw=1)
        for k, branch in enumerate(BRANCHES):
            t = summ[(summ.market == label) & (summ.branch == branch) & (summ.rule == "fixed")]
            x = np.array(t.delta) * 100 + (k - 1) * 0.3
            ax.errorbar(x, 100 * t.ROI_bankroll,
                        yerr=[100 * (t.ROI_bankroll - t.ROI_CI95_low), 100 * (t.ROI_CI95_high - t.ROI_bankroll)],
                        marker="o", capsize=3, c=colors[branch], label=f"{branch}-driven")
            for xi, yi, n in zip(x, 100 * t.ROI_bankroll, t.bets):
                ax.annotate(str(n), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8, color=colors[branch])
        ax.set_title(f"{label} winner, fixed $100")
        ax.set_xlabel("minimum edge  p_model − 1/odds  (%)")
    axes[0].set_ylabel("ROI (%)  [numbers = bets placed]")
    axes[0].legend()
    fig.savefig("../assets/edge_threshold.png")
    print("\nSaved results/summary.csv, results/bets_*.csv, ../assets/bankroll.png")


if __name__ == "__main__":
    main()
