"""
series_probs.py  -  turn per-game win probabilities into SERIES win probabilities

Bookmakers price the series (Bo1 / Bo3 / Bo5), our model predicts single games.

For each series in the betting period:
  1. take the features of the first game we have for that series (pre-series information)
  2. team A = blue side of that game
       p_A_on_blue = model(x)          x = blue - red features
       p_A_on_red  = 1 - model(-x)     same teams, sides swapped (the blue-side bonus flips)
       p_game      = average of the two (sides alternate during a series)
  3. series formula, assuming games are independent with the same p:
       Bo1: p
       Bo3: p^2 (3 - 2p)
       Bo5: p^3 (10 - 15p + 6p^2)
  4. best-of is inferred from the result: winner's game wins w -> Bo(2w - 1)

Usage:  python series_probs.py          (run from 03_model_selection/, after pipeline.py)
Output: results/series_bet.csv
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

BET_START = "2026-06-01"


def series_prob(p, best_of):
    if best_of == 1:
        return p
    if best_of == 3:
        return p**2 * (3 - 2 * p)
    if best_of == 5:
        return p**3 * (10 - 15 * p + 6 * p**2)
    return np.nan


def assign_series(df):
    """New series whenever game number restarts at 1 (or drops) for the same team pair."""
    df = df.sort_values("date").copy()
    df["pair"] = df.apply(lambda r: " vs ".join(sorted([r.blue_team, r.red_team])), axis=1)
    sid, out = 0, {}
    for (_, _), g in df.groupby(["league", "pair"], sort=False):
        last = None
        for idx, row in g.iterrows():
            if last is None or row["game"] <= last or row["game"] == 1:
                sid += 1
            out[idx] = sid
            last = row["game"]
    df["series_id"] = pd.Series(out)
    return df


def main():
    df = pd.read_csv("../02_features/features.csv", parse_dates=["date"])
    final = json.load(open(os.path.join("results", "final_models.json")))
    models = {b: joblib.load(os.path.join("results", f"model_{b}.joblib")) for b in final}
    df[[c for c in df if c.startswith("f_")]] = df[[c for c in df if c.startswith("f_")]].fillna(0)

    bet = assign_series(df[df["date"] >= BET_START])
    rows = []
    for sid, g in bet.groupby("series_id"):
        g = g.sort_values("date")
        first = g.iloc[0]
        team_a, team_b = first["blue_team"], first["red_team"]
        wins_a = int(((g.blue_team == team_a) & (g.blue_win == 1)).sum()
                     + ((g.red_team == team_a) & (g.blue_win == 0)).sum())
        wins_b = len(g) - wins_a
        best_of = 2 * max(wins_a, wins_b) - 1
        row = {"series_id": sid, "date": first["date"], "league": first["league"],
               "playoffs": first["playoffs"], "team_a": team_a, "team_b": team_b,
               "best_of": best_of, "score": f"{wins_a}-{wins_b}",
               "games_in_data": len(g), "first_game_no": int(first["game"]),
               "team_a_won": int(wins_a > wins_b)}
        for branch, f in final.items():
            x = first[f["features"]].to_frame().T.astype(float)
            p_blue = models[branch].predict_proba(x)[0, 1]
            p_red = 1 - models[branch].predict_proba(-x)[0, 1]
            p = (p_blue + p_red) / 2
            row[f"p_game_a_{branch}"] = p
            row[f"p_series_a_{branch}"] = series_prob(p, best_of)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join("results", "series_bet.csv"), index=False)
    print(f"{len(out)} series in the betting period")
    print(out["best_of"].value_counts().sort_index().rename("count by best-of").to_string())
    print(f"series where game 1 is missing from features.csv: {(out.first_game_no > 1).sum()}")
    for branch in final:
        p = out[f"p_series_a_{branch}"]
        acc = ((p >= 0.5) == out["team_a_won"]).mean()
        print(f"{branch:11s} series accuracy {acc:.3f}")
    print("Saved results/series_bet.csv")


if __name__ == "__main__":
    main()
