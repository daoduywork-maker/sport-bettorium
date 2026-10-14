"""
export_data.py  -  write the small data files that the app (lol_analyst.py) reads
(run from 05_analyst/ after the feature scripts in 02_features/ have been rebuilt on the newest Oracle's Elixir CSVs)

Writes 05_analyst/data/:
  manifest.json       data date, model settings, file list
  train.csv.gz        training table: every major-league game 2022+, the 12 pre-draft factors (blue - red) and the result
  predictions.csv.gz  walk-forward predictions 2024 onward (model trained only on earlier years), for the true/false-positive charts
  teams.csv           team snapshot: Elo, recent in-game form (last 10 games), last game
  players.csv         player snapshot: Elo, games played (all / major / playoffs / international), first game
  lineups.csv         each team's last starting five by role
  games.csv.gz        every major-league and international game since 2021 (head-to-head, recent form)
  lanes.csv.gz        every same-lane meeting of two players since 2021 (lane head-to-head)
Source: Oracle's Elixir match data (oracleselixir.com), processed by this project.
"""
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import state as st

OUT = st.HERE / "data"
EXP = ["f_exp_games", "f_exp_major", "f_exp_playoffs", "f_exp_intl", "f_exp_years", "f_rookies"]


def main():
    OUT.mkdir(exist_ok=True)
    fm = json.load(open(st.ROOT / "03_model_selection" / "results" / "final_models.json"))["logloss"]
    F = [f for f in fm["features"] if f != "f_elo"] + ["f_player_elo"] + EXP
    C = fm["hyperparams"]["C"]

    # ---------------- training table
    df = pd.read_csv(st.ROOT / "02_features" / "features.csv", parse_dates=["date"])
    for f in ["roster_features.csv", "experience_features.csv"]:
        x = pd.read_csv(st.ROOT / "02_features" / f)
        df = df.merge(x[[c for c in x if c == "gameid" or c.startswith("f_")]], on="gameid", how="left")
    df[F] = df[F].fillna(0)
    keep = ["gameid", "date", "league", "playoffs", "blue_team", "red_team", "blue_win"] + F
    tr = df[df.date >= "2022-01-01"][keep]
    tr.round(5).to_csv(OUT / "train.csv.gz", index=False)

    # ---------------- walk-forward predictions: year Y predicted by a model trained on 2022 .. Y-1
    out = []
    for Y in range(2024, df.date.dt.year.max() + 1):
        a, b = df[(df.date >= "2022-01-01") & (df.date < f"{Y}-01-01")], df[df.date.dt.year == Y].copy()
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(a[F], a.blue_win)
        b["p_blue"] = m.predict_proba(b[F])[:, 1].round(4)
        out.append(b[["gameid", "date", "league", "playoffs", "blue_team", "red_team", "blue_win", "p_blue"]])
    pr = pd.concat(out); pr.to_csv(OUT / "predictions.csv.gz", index=False)

    # ---------------- snapshot tables
    S = st.load_state()
    t = S["team_names"].copy()
    t["elo"] = pd.Series(S["team_elo"]); t["elo_year"] = pd.Series(S["team_elo_year"]); t["n_games"] = S["team_games"]
    h = S["team_elo_hist"]
    t["elo_change"] = [h[k][-1][1] - h[k][0][1] if len(h.get(k, [])) > 1 else 0.0 for k in t.index]
    t["elo_change_games"] = [len(h.get(k, [])) - 1 for k in t.index]
    form = S["team_form"].add_prefix("form_")
    t = t.join(form).rename_axis("team_key").reset_index()
    t.round(5).to_csv(OUT / "teams.csv", index=False)

    p = S["player_info"].copy()
    p["elo"] = pd.Series(S["player_elo"]); p["elo_year"] = pd.Series(S["player_year"]); p["team_key"] = pd.Series(S["player_team"])
    p = p.rename_axis("player_key").reset_index()
    p.round(3).to_csv(OUT / "players.csv", index=False)

    rows = [(k, pos, pid) for k, d in S["lineup_pos"].items() for pos, pid in d.items()]
    pd.DataFrame(rows, columns=["team_key", "role", "player_key"]).to_csv(OUT / "lineups.csv", index=False)

    S["games"].to_csv(OUT / "games.csv.gz", index=False)
    rows = [(a, b, pos, d, w) for (a, b, pos), L in S["lanes"].items() for d, w in L]
    pd.DataFrame(rows, columns=["player_a", "player_b", "role", "date", "winner"]).to_csv(OUT / "lanes.csv.gz", index=False)

    man = {"version": pd.Timestamp.now().strftime("%Y%m%d%H%M"), "data_until": str(S["asof_data"].date()),
           "source": "Oracle's Elixir match data (oracleselixir.com), processed by this project",
           "model": {"type": "logistic regression (StandardScaler + L2)", "C": C, "features": F, "train_from": "2022-01-01"},
           "elo": {"K": st.K_ELO, "yearly_regression": st.REGRESS},
           "files": sorted(f.name for f in OUT.iterdir() if f.name != "manifest.json")}
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:20s} {f.stat().st_size / 1e6:6.2f} MB")
    print(f"train {len(tr)} rows, predictions {len(pr)} rows ({pr.date.min().date()} .. {pr.date.max().date()}), teams {len(t)}, players {len(p)}")


if __name__ == "__main__":
    main()
