"""
synergy_features.py  -  which champion PAIRS (and the mid-jungle-support trio) matter? (run from 02_features/)

f_synergy in draft_features.py averages 3 duos into one number. Here every group is its own
feature, so the model can weight them separately. For each group: win rate of that exact champion
combination on the same team, last 365 days, shrunk to 50 % with 10 pseudo-games, minus 0.5;
value = blue - red. Only games BEFORE the scored game are used.
  f_syn_bot_sup   ADC + support (bot lane)
  f_syn_jng_mid   jungle + mid
  f_syn_jng_sup   jungle + support (roaming / early skirmishes)
  f_syn_mid_sup   mid + support
  f_syn_jng_top   jungle + top
  f_syn_trio      jungle + mid + support (exact trio: rare, so mostly shrunk to 0)
  f_syn_trio_cls  jungle + mid + support CLASS trio (e.g. Diver + Burst + Catcher), pooled
Output: synergy_features.csv
"""
import glob
import re

import pandas as pd

import build_features as bf
from class_features import RunningWindow, key, POS

GROUPS = {"bot_sup": ("bot", "sup"), "jng_mid": ("jng", "mid"), "jng_sup": ("jng", "sup"),
          "mid_sup": ("mid", "sup"), "jng_top": ("jng", "top"), "trio": ("jng", "mid", "sup")}
SPAN = pd.Timedelta(days=365)
COLS = ["gameid", "league", "date", "side", "position", "champion", "result"]


def main():
    cc = pd.read_csv("champion_classes.csv")
    SUB = {key(r.champion): r.subclasses.split("|")[0] for r in cc.itertuples()}   # first subclass
    parts = []
    for f in sorted(glob.glob("../01_data/*LoL_esports_match_data*.csv")):
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        parts.append(d[d["position"].isin(POS)])
    d = pd.concat(parts, ignore_index=True)
    d["league"] = d["league"].replace(bf.LEAGUE_MAP)
    d = d[d["league"].isin(bf.LEAGUES) & d["champion"].notna()].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["k"] = d["champion"].map(key)
    d = d[d.groupby("gameid")["k"].transform("size") == 10].sort_values(["date", "gameid"])

    win = {g: RunningWindow(SPAN) for g in list(GROUPS) + ["trio_cls"]}
    rows = []
    for gid, g in d.groupby("gameid", sort=False):
        now = g["date"].iloc[0]
        T = {s: g[g["side"] == s].set_index("position")["k"] for s in ("Blue", "Red")}
        if any(len(t) != 5 for t in T.values()):
            continue
        keys = {s: {name: tuple(t[p] for p in pos) for name, pos in GROUPS.items()} for s, t in T.items()}
        for s, t in T.items():
            keys[s]["trio_cls"] = tuple(SUB.get(t[p], "?") for p in ("jng", "mid", "sup"))
        row = {"gameid": gid}
        for name, w in win.items():
            row[f"f_syn_{name}"] = w.rate(keys["Blue"][name], now) - w.rate(keys["Red"][name], now)
        rows.append(row)
        bw = int(g.loc[g["side"] == "Blue", "result"].iloc[0])
        for s, y in (("Blue", bw), ("Red", 1 - bw)):
            for name, w in win.items():
                w.add(keys[s][name], now, y)
    out = pd.DataFrame(rows)
    out.to_csv("synergy_features.csv", index=False)
    print(f"Saved synergy_features.csv: {len(out)} games")
    print(out.drop(columns="gameid").describe().loc[["std"]].round(4).to_string())


if __name__ == "__main__":
    main()
