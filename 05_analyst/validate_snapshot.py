"""
validate_snapshot.py  -  does the snapshot reproduce the training features?  (run from 05_analyst/)
Rebuild the snapshot using only games BEFORE a chosen historical game, compute the model features for that game's two teams and lineups,
and compare them with the values stored in 02_features/*.csv for that very game.  Differences should be ~0.
"""
import sys
import numpy as np, pandas as pd
import state as st
df = pd.read_csv(st.ROOT / "02_features" / "features.csv", parse_dates=["date"])
for f in ["roster_features.csv", "experience_features.csv"]:
    x = pd.read_csv(st.ROOT / "02_features" / f); df = df.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left")
tests = df[df.date >= "2026-06-01"].sample(2, random_state=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
worst = {}
for r in tests.itertuples():
    S = st.build_state(until=r.date)
    ka, kb = S["team_names"].index[S["team_names"].name == r.blue_team][-1], S["team_names"].index[S["team_names"].name == r.red_team][-1]
    # actual lineups of that game
    la, lb = st.get_lineup(S, ka), st.get_lineup(S, kb)       # last lineup BEFORE the game; use real lineup instead if it differs
    P = pd.concat([pd.read_csv(f, low_memory=False, usecols=lambda c: c in ["gameid", "side", "position", "playerid", "playername"]) for f in
                   [str(st.ROOT / "01_data" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv") for y in (2026,)]])
    P = P[(P.gameid == r.gameid) & P.position.isin(st.POS)]; P["player"] = P.playerid.fillna(P.playername).astype(str)
    la = dict(zip(P[P.side == "Blue"].position, P[P.side == "Blue"].player)); lb = dict(zip(P[P.side == "Red"].position, P[P.side == "Red"].player))
    asof = pd.Timestamp(r.date)
    va, vb = st.side_values(S, ka, la, asof), st.side_values(S, kb, lb, asof)
    print(f"\n{r.date}  {r.blue_team} v {r.red_team}")
    for f in ["f_gspd", "f_wcpm_edge", "f_vspm_edge", "f_firsttower", "f_firstdragon", "f_player_elo", "f_exp_games", "f_exp_major", "f_exp_playoffs", "f_exp_intl", "f_exp_years", "f_rookies"]:
        mine, true = va[f] - vb[f], getattr(r, f)
        print(f"   {f:15s} snapshot {mine:9.3f}   training {true:9.3f}   diff {mine - true:+.3f}")
        worst[f] = max(worst.get(f, 0), abs(mine - true))
print("\nworst absolute difference per feature over the tested games:"); print({k: round(v, 3) for k, v in worst.items()})
