"""
experience_features.py  -  how experienced are the five starters? (run from 02_features/)

Uses EVERY league in Oracle's Elixir 2021-2026 (academy, ERLs, internationals), so a rookie who
played two years in an academy league counts that experience. Data start in 2021, so careers
longer than that are cut at 2021 (a 10-year veteran looks like a 5-year one): fine for ranking
"new" vs "experienced", not for absolute career length.
For each player, counting only games BEFORE the current one:
  f_exp_games    : mean over starters of log(1 + pro games, any league)
  f_exp_major    : mean of log(1 + games in the 6 major leagues + international events)
  f_exp_playoffs : mean of log(1 + playoff games, any league)
  f_exp_intl     : mean of log(1 + international games: MSI, Worlds, First Stand, EWC)
  f_exp_years    : mean years since the player's first game in the data
  f_rookies      : number of starters with < 30 major-league games
All values blue - red. Experience is known BEFORE the draft (roster only), so these are pre-match
features, usable for series bets.
Output: experience_features.csv
"""
import glob

import numpy as np
import pandas as pd

import build_features as bf

INTL = {"MSI", "WLDs", "FST", "EWC"}
POS = ["top", "jng", "mid", "bot", "sup"]
COLS = ["gameid", "league", "date", "playoffs", "side", "position", "playername", "playerid"]
ROOKIE = 30


def main():
    parts = []
    for f in sorted(glob.glob("../01_data/*LoL_esports_match_data*.csv")):
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        parts.append(d[d["position"].isin(POS)])
    d = pd.concat(parts, ignore_index=True)
    d["league"] = d["league"].replace(bf.LEAGUE_MAP)
    d["date"] = pd.to_datetime(d["date"])
    d["player"] = d["playerid"].fillna(d["playername"]).astype(str)
    d = d[d["player"].ne("nan")].sort_values(["date", "gameid"]).reset_index(drop=True)
    d["major"] = d["league"].isin(bf.LEAGUES | INTL).astype(int)
    d["intl"] = d["league"].isin(INTL).astype(int)
    d["po"] = d["playoffs"].fillna(0).astype(int)

    g = d.groupby("player")
    d["n_all"] = g.cumcount()                                 # games before this one
    for c in ["major", "intl", "po"]:
        d[f"n_{c}"] = g[c].cumsum() - d[c]
    d["years"] = (d["date"] - g["date"].transform("min")).dt.days / 365.25

    d = d[d["league"].isin(bf.LEAGUES | INTL)]
    d = d[d.groupby("gameid")["player"].transform("size") == 10]
    t = d.groupby(["gameid", "side"]).agg(
        games=("n_all", lambda s: np.log1p(s).mean()),
        major=("n_major", lambda s: np.log1p(s).mean()),
        playoffs=("n_po", lambda s: np.log1p(s).mean()),
        intl=("n_intl", lambda s: np.log1p(s).mean()),
        years=("years", "mean"),
        rookies=("n_major", lambda s: (s < ROOKIE).sum())).unstack("side")
    out = pd.DataFrame({f"f_exp_{k}" if k != "rookies" else "f_rookies":
                        t[(k, "Blue")] - t[(k, "Red")] for k in ["games", "major", "playoffs", "intl", "years", "rookies"]})
    out = out.reset_index()
    out.to_csv("experience_features.csv", index=False)
    print(f"Saved experience_features.csv: {len(out)} games")
    print(out.drop(columns="gameid").describe().round(3).to_string())


if __name__ == "__main__":
    main()
