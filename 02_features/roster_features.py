"""
roster_features.py  -  features that know WHO is playing (run from 02_features/)

Team-level stats and team Elo assume a team is the same five people all year. After the
off-season, or when a substitute plays, that is false. From the player rows of Oracle's Elixir:

  f_roster_changes : starters different from the team's previous game (0..5), blue - red
  f_familiarity    : for each starter, share of the team's last 10 games he/she played;
                     averaged over the 5 starters (0 = brand-new lineup, 1 = settled), blue - red
  f_player_elo     : every PLAYER has an Elo; team strength = mean of its 5 starters' Elo.
                     Ratings follow players when they change team.          blue - red

All values use only games BEFORE the current one, plus the current lineup (announced before
the match, so it is pre-match information).
Output: roster_features.csv  (gameid + the 3 features), merged on gameid later.
"""
import glob
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import build_features as bf

K = 24             # same update size as the team Elo
REGRESS = 0.25     # pull 25 % back toward 1500 each new year
POS = {"top", "jng", "mid", "bot", "sup"}
COLS = ["gameid", "league", "date", "side", "position", "playername", "playerid",
        "teamname", "teamid", "result"]


def load_players(data_dir="../01_data"):
    parts = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*LoL_esports_match_data*.csv"))):
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        parts.append(d[d["position"].isin(POS)])
    d = pd.concat(parts, ignore_index=True)
    d["league"] = d["league"].replace(bf.LEAGUE_MAP)
    d = d[d["league"].isin(bf.LEAGUES)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["player"] = d["playerid"].fillna(d["playername"]).astype(str)
    d["team"] = d["teamid"].fillna(d["teamname"]).astype(str)
    n = d.groupby(["gameid", "side"])["player"].transform("size")
    d = d[n == 5]
    ok = d.groupby("gameid")["side"].nunique() == 2
    return d[d["gameid"].isin(ok[ok].index)].sort_values(["date", "gameid"])


def main():
    d = load_players()
    rating, last_year = {}, {}
    history = defaultdict(lambda: deque(maxlen=10))        # team -> last 10 lineups
    rows = []
    for gid, g in d.groupby("gameid", sort=False):
        year = g["date"].iloc[0].year
        side = {}
        for s in ("Blue", "Red"):
            gs = g[g["side"] == s]
            team, lineup = gs["team"].iloc[0], set(gs["player"])
            for p in lineup:
                if p not in rating:
                    rating[p] = 1500.0
                elif last_year.get(p) != year:
                    rating[p] = 1500 + (1 - REGRESS) * (rating[p] - 1500)
                last_year[p] = year
            hist = history[team]
            changes = 5 - len(lineup & hist[-1]) if hist else 5
            fam = np.mean([sum(p in h for h in hist) / len(hist) for p in lineup]) if hist else 0.0
            side[s] = {"team": team, "lineup": lineup, "res": gs["result"].iloc[0],
                       "changes": changes, "fam": fam,
                       "pelo": np.mean([rating[p] for p in lineup])}
        b, r = side["Blue"], side["Red"]
        rows.append({"gameid": gid,
                     "f_roster_changes": b["changes"] - r["changes"],
                     "f_familiarity": b["fam"] - r["fam"],
                     "f_player_elo": b["pelo"] - r["pelo"],
                     "blue_changes": b["changes"], "red_changes": r["changes"]})
        exp_b = 1 / (1 + 10 ** ((r["pelo"] - b["pelo"]) / 400))
        delta = K * (b["res"] - exp_b)
        for p in b["lineup"]:
            rating[p] += delta
        for p in r["lineup"]:
            rating[p] -= delta
        history[b["team"]].append(b["lineup"])
        history[r["team"]].append(r["lineup"])
    out = pd.DataFrame(rows)
    out.to_csv("roster_features.csv", index=False)
    print(f"Saved roster_features.csv: {len(out)} games")
    print("games where a team changed >= 1 starter:", int(((out.blue_changes > 0) | (out.red_changes > 0)).sum()))


if __name__ == "__main__":
    main()
