"""
early_features.py  -  early-game strength (gold / XP / CS lead at 10-15 min), run from 02_features/

Rolling mean of each team's previous 10 games (current game excluded), blue - red:
  f_gd10, f_gd15, f_xpd15, f_csd15
LPL has no timeline data in Oracle's Elixir: those games get NaN here (filled with 0 = "no
information" in the model) and f_early_available = 1 only when both teams have values.
Output: early_features.csv
"""
import glob

import pandas as pd

import build_features as bf

STATS = {"golddiffat10": "gd10", "golddiffat15": "gd15", "xpdiffat15": "xpd15", "csdiffat15": "csd15"}
COLS = ["gameid", "league", "date", "side", "position", "teamname", "teamid"] + list(STATS)


def main():
    parts = []
    for f in sorted(glob.glob("../01_data/*LoL_esports_match_data*.csv")):
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        parts.append(d[d["position"] == "team"])
    d = pd.concat(parts, ignore_index=True)
    d["league"] = d["league"].replace(bf.LEAGUE_MAP)
    d = d[d["league"].isin(bf.LEAGUES)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["team"] = d["teamid"].fillna(d["teamname"]).astype(str)
    d = d.sort_values(["date", "gameid", "side"])
    g = d.groupby("team")
    for raw, short in STATS.items():
        d[short] = g[raw].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    b = d[d.side == "Blue"].set_index("gameid")
    r = d[d.side == "Red"].set_index("gameid")
    out = pd.DataFrame(index=b.index)
    for short in STATS.values():
        out["f_" + short] = b[short] - r[short]
    out["f_early_available"] = out["f_gd15"].notna().astype(int)
    out.reset_index().to_csv("early_features.csv", index=False)
    print(f"Saved early_features.csv: {len(out)} games, early stats available for {out.f_early_available.mean():.0%}")


if __name__ == "__main__":
    main()
