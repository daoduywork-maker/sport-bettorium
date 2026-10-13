"""
build_features.py  -  Oracle's Elixir match data  ->  one row per game (blue - red features)

Usage (in WSL, inside your venv):
    pip install pandas numpy
    cd 02_features && python build_features.py      # reads ../01_data, writes features.csv

What it does
  1. Reads every *LoL_esports_match_data*.csv in --data-dir, keeps the team rows only
  2. Keeps LCS, LEC, LCK, LPL, CBLOL, LCP  (2025 LTA N -> LCS, LTA S -> CBLOL)
  3. Builds per-game "edge" stats for each team (team value - opponent value)
  4. Rolling average of each team's PREVIOUS games only (shift(1) -> no leakage)
  5. Pre-game Elo rating for each team
  6. Blue minus red  ->  one row per game, label = blue_win

Only full-game stats are used (no @15-min stats), because LPL is missing them.
Feature filtering (KS shift test, correlation filter) and standardisation are NOT
done here: they must be fitted on the training split, so they live in the model step.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

LEAGUES = {"LCS", "LEC", "LCK", "LPL", "CBLOL", "LCP"}
LEAGUE_MAP = {"LTA N": "LCS", "LTA S": "CBLOL"}   # 2025 rename

WINDOW = 10        # rolling window (games)
MIN_GAMES = 5      # need at least this many past games for a feature value
ELO_K = 24         # Elo update speed
ELO_REGRESS = 0.25 # pull ratings 25% back toward 1500 at each new year (roster changes)

RAW_COLS = [
    "gameid", "datacompleteness", "league", "year", "split", "playoffs", "date",
    "game", "patch", "side", "position", "teamname", "teamid", "gamelength",
    "result", "kills", "deaths", "firstblood", "firstdragon", "dragons",
    "opp_dragons", "firstherald", "void_grubs", "opp_void_grubs", "firstbaron",
    "barons", "opp_barons", "firsttower", "towers", "opp_towers", "inhibitors",
    "opp_inhibitors", "dpm", "vspm", "earned gpm", "cspm", "gspd", "wcpm",
]


def load(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*LoL_esports_match_data*.csv")))
    if not files:
        raise SystemExit(f"No Oracle's Elixir CSVs found in {data_dir}")
    parts = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, usecols=lambda c: c in RAW_COLS)
        df = df[df["position"] == "team"]
        print(f"  {os.path.basename(f)}: {df['gameid'].nunique()} games (all leagues)")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    df["league"] = df["league"].replace(LEAGUE_MAP)
    df = df[df["league"].isin(LEAGUES)].copy()
    df["date"] = pd.to_datetime(df["date"])
    # Oracle's "year" column labels late-year qualifiers as next season; use calendar year
    df["year"] = df["date"].dt.year

    # keep only clean games: exactly one Blue and one Red row
    sides = df.groupby("gameid")["side"].agg(lambda s: tuple(sorted(s)))
    good = sides[sides == ("Blue", "Red")].index
    dropped = df["gameid"].nunique() - len(good)
    if dropped:
        print(f"  dropped {dropped} games without one Blue + one Red row")
    df = df[df["gameid"].isin(good)]

    # team key: teamid when present, otherwise the name
    df["team"] = df["teamid"].fillna(df["teamname"]).astype(str)
    return df.sort_values(["date", "gameid", "side"]).reset_index(drop=True)


def add_edges(df):
    """Per-game stats, expressed as team minus opponent."""
    opp_cols = ["dpm", "vspm", "earned gpm", "cspm", "wcpm"]
    opp = df[["gameid", "side"] + opp_cols].copy()
    opp["side"] = opp["side"].map({"Blue": "Red", "Red": "Blue"})
    opp = opp.rename(columns={c: "opp_" + c for c in opp_cols})
    df = df.merge(opp, on=["gameid", "side"], how="left")

    e = pd.DataFrame(index=df.index)
    e["win"] = df["result"]
    e["kill_edge"] = df["kills"] - df["deaths"]
    e["dragon_edge"] = df["dragons"] - df["opp_dragons"]
    e["grub_edge"] = df["void_grubs"] - df["opp_void_grubs"]
    e["baron_edge"] = df["barons"] - df["opp_barons"]
    e["tower_edge"] = df["towers"] - df["opp_towers"]
    e["inhib_edge"] = df["inhibitors"] - df["opp_inhibitors"]
    e["dpm_edge"] = df["dpm"] - df["opp_dpm"]
    e["vspm_edge"] = df["vspm"] - df["opp_vspm"]
    e["gpm_edge"] = df["earned gpm"] - df["opp_earned gpm"]
    # cspm, firstherald, firstbaron left out: missing for LPL
    e["wcpm_edge"] = df["wcpm"] - df["opp_wcpm"]
    e["gspd"] = df["gspd"]                      # already relative
    for c in ["firstblood", "firstdragon", "firsttower"]:
        e[c] = df[c]
    e["gamelength_min"] = df["gamelength"] / 60
    return df, list(e.columns), e


def rolling_past(df, stats, edges):
    """Average of each team's previous WINDOW games (current game excluded)."""
    edges = edges.copy()
    edges["team"] = df["team"].values
    out = pd.DataFrame(index=df.index)
    g = edges.groupby("team")
    for c in stats:
        out[c] = g[c].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
        )
    out["n_past_games"] = g.cumcount()
    return out


def add_elo(df):
    """Pre-game Elo for every team row (ratings updated game by game, in date order)."""
    rating, last_year = {}, {}
    pre = {}
    games = df.drop_duplicates("gameid")[["gameid", "date", "year"]]
    by_game = df.set_index(["gameid", "side"])
    for gid, _, year in games.itertuples(index=False):
        b, r = by_game.loc[(gid, "Blue")], by_game.loc[(gid, "Red")]
        for t in (b["team"], r["team"]):
            if t not in rating:
                rating[t] = 1500.0
            elif last_year.get(t) != year:     # new year: partial reset
                rating[t] = 1500 + (1 - ELO_REGRESS) * (rating[t] - 1500)
            last_year[t] = year
        rb, rr = rating[b["team"]], rating[r["team"]]
        pre[(gid, "Blue")], pre[(gid, "Red")] = rb, rr
        exp_b = 1 / (1 + 10 ** ((rr - rb) / 400))
        delta = ELO_K * (b["result"] - exp_b)
        rating[b["team"]] += delta
        rating[r["team"]] -= delta
    return pd.Series([pre[(g, s)] for g, s in zip(df["gameid"], df["side"])], index=df.index)


def to_game_rows(df, past, stats):
    team_feat = pd.concat([df[["gameid", "side", "teamname"]], past], axis=1)
    blue = team_feat[team_feat["side"] == "Blue"].set_index("gameid")
    red = team_feat[team_feat["side"] == "Red"].set_index("gameid")

    meta_cols = ["gameid", "date", "year", "league", "split", "playoffs", "game",
                 "patch", "datacompleteness"]
    meta = df[df["side"] == "Blue"][meta_cols + ["result"]].set_index("gameid")

    out = meta.rename(columns={"result": "blue_win"})
    out["blue_team"] = blue["teamname"]
    out["red_team"] = red["teamname"]
    out["blue_n_past"] = blue["n_past_games"]
    out["red_n_past"] = red["n_past_games"]
    for c in stats + ["elo"]:
        out["f_" + c] = blue[c] - red[c]
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../01_data")
    ap.add_argument("--out", default="features.csv")
    args = ap.parse_args()

    print("1) loading")
    df = load(args.data_dir)
    print(f"   {df['gameid'].nunique()} games in {sorted(df['league'].unique())}")

    print("2) per-game edges")
    df, stats, edges = add_edges(df)

    print("3) rolling averages of past games")
    past = rolling_past(df, stats, edges)

    print("4) Elo")
    past["elo"] = add_elo(df)

    print("5) blue - red")
    games = to_game_rows(df, past, stats)
    feat_cols = [c for c in games.columns if c.startswith("f_")]
    before = len(games)
    # needs history for both teams; columns that are empty for a whole era
    # (e.g. grubs before they existed) are allowed to be NaN
    core = [c for c in feat_cols if c not in ("f_grub_edge",)]
    games = games.dropna(subset=core)
    print(f"   kept {len(games)} / {before} games (rest: a team had < {MIN_GAMES} past games)")

    games.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}: {len(games)} rows, {len(feat_cols)} features")
    print(games.groupby(["year", "league"]).size().unstack(fill_value=0))
    print(f"blue win rate: {games['blue_win'].mean():.3f}")


if __name__ == "__main__":
    main()
