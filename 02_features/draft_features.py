"""
draft_features.py  -  what does the DRAFT (champions picked) add?  (run from 02_features/)

For each game, using only games played BEFORE it (all values blue - red):
  f_champ_wr       : champion's recent pro win rate (last 90 days, 6 leagues), shrunk to 50 % (20 pseudo-games)
  f_champ_wr_patch : champion's win rate on the CURRENT patch, shrunk toward its 90-day rate (10 pseudo-games)
  f_comfort_games  : log(1 + games the player has played on the champion)
  f_comfort_wr     : player's win rate on the champion, shrunk to 50 % (5 pseudo-games)
  f_lane_matchup   : sum over the 5 lanes of (win rate of my champion vs that exact enemy champion
                     in that lane, last 365 days, shrunk to 50 % with 10 pseudo-games) - 0.5
                     (blue only: the red value is the mirror image)
  f_synergy        : mean over duos (bot+sup, jng+mid, jng+top) of the duo's win rate together
                     (365 days, shrunk to 50 %, 10 pseudo-games) - 0.5
Each per-champion value is averaged over the 5 picks.

IMPORTANT: the draft is only known ~10 minutes before the game starts. These features can
improve a game prediction, but they can NOT be used for a pre-match series bet placed before
the draft; only for a bet placed after the draft (live, map by map).
Output: draft_features.csv
"""
import glob
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import build_features as bf

WIN90, WIN365 = pd.Timedelta(days=90), pd.Timedelta(days=365)
A_CHAMP, A_PATCH, A_PLAYER, A_PAIR = 20, 10, 5, 10
POS = ["top", "jng", "mid", "bot", "sup"]
DUOS = [("bot", "sup"), ("jng", "mid"), ("jng", "top")]
COLS = ["gameid", "league", "date", "patch", "side", "position", "playername", "playerid", "champion", "result"]


class Window:
    """Wins / games of a key over a sliding time window (running sums, so each lookup is O(1))."""
    def __init__(self, span):
        self.span, self.h = span, defaultdict(deque)
        self.s = defaultdict(lambda: [0, 0])            # key -> [games, wins]

    def rate(self, key, now, a, prior=0.5):
        h, s = self.h[key], self.s[key]
        while h and h[0][0] < now - self.span:
            s[1] -= h.popleft()[1]
            s[0] -= 1
        return (s[1] + a * prior) / (s[0] + a)

    def add(self, key, now, win):
        self.h[key].append((now, win))
        self.s[key][0] += 1
        self.s[key][1] += win


def main():
    parts = []
    for f in sorted(glob.glob("../01_data/*LoL_esports_match_data*.csv")):
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        parts.append(d[d["position"].isin(POS)])
    d = pd.concat(parts, ignore_index=True)
    d["league"] = d["league"].replace(bf.LEAGUE_MAP)
    d = d[d["league"].isin(bf.LEAGUES) & d["champion"].notna()].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["player"] = d["playerid"].fillna(d["playername"]).astype(str)
    d["patch"] = d["patch"].astype(str)
    n = d.groupby("gameid")["player"].transform("size")
    d = d[n == 10].sort_values(["date", "gameid"])

    champ90, lane, duo = Window(WIN90), Window(WIN365), Window(WIN365)
    champ_patch = defaultdict(lambda: [0, 0])          # (patch, champ) -> [games, wins]
    pc = defaultdict(lambda: [0, 0])                   # (player, champ) -> [games, wins]
    rows = []
    for gid, g in d.groupby("gameid", sort=False):
        now, patch = g["date"].iloc[0], g["patch"].iloc[0]
        picks = {s: g[g["side"] == s].set_index("position") for s in ("Blue", "Red")}
        if any(len(p) != 5 for p in picks.values()):
            continue
        val = {}
        for s, p in picks.items():
            wr, wrp, cg, cw = [], [], [], []
            for pos in POS:
                c, pl = p.at[pos, "champion"], p.at[pos, "player"]
                base = champ90.rate(c, now, A_CHAMP)
                wr.append(base)
                n_p, w_p = champ_patch[(patch, c)]
                wrp.append((w_p + A_PATCH * base) / (n_p + A_PATCH))
                games, wins = pc[(pl, c)]
                cg.append(np.log1p(games))
                cw.append((wins + 0.5 * A_PLAYER) / (games + A_PLAYER))
            syn = np.mean([duo.rate((a, p.at[a, "champion"], b, p.at[b, "champion"]), now, A_PAIR) - .5
                           for a, b in DUOS])
            val[s] = dict(wr=np.mean(wr), wrp=np.mean(wrp), cg=np.mean(cg), cw=np.mean(cw), syn=syn)
        B, R = picks["Blue"], picks["Red"]
        matchup = sum(lane.rate((pos, B.at[pos, "champion"], R.at[pos, "champion"]), now, A_PAIR) - .5
                      for pos in POS)
        vb, vr = val["Blue"], val["Red"]
        rows.append({"gameid": gid, "f_champ_wr": vb["wr"] - vr["wr"],
                     "f_champ_wr_patch": vb["wrp"] - vr["wrp"],
                     "f_comfort_games": vb["cg"] - vr["cg"], "f_comfort_wr": vb["cw"] - vr["cw"],
                     "f_lane_matchup": matchup, "f_synergy": vb["syn"] - vr["syn"]})
        # ---- update AFTER scoring the game
        bw = int(B["result"].iloc[0])
        for s, p, win in [("Blue", B, bw), ("Red", R, 1 - bw)]:
            for pos in POS:
                c = p.at[pos, "champion"]
                champ90.add(c, now, win)
                champ_patch[(patch, c)][0] += 1
                champ_patch[(patch, c)][1] += win
                pc[(p.at[pos, "player"], c)][0] += 1
                pc[(p.at[pos, "player"], c)][1] += win
            for a, b in DUOS:
                duo.add((a, p.at[a, "champion"], b, p.at[b, "champion"]), now, win)
        for pos in POS:   # lane matchup stored from both perspectives
            lane.add((pos, B.at[pos, "champion"], R.at[pos, "champion"]), now, bw)
            lane.add((pos, R.at[pos, "champion"], B.at[pos, "champion"]), now, 1 - bw)
    out = pd.DataFrame(rows)
    out.to_csv("draft_features.csv", index=False)
    print(f"Saved draft_features.csv: {len(out)} games")
    print(out.drop(columns="gameid").describe().round(3).to_string())


if __name__ == "__main__":
    main()
