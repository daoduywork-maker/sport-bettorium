"""
state.py  -  step 1 of the matchup analyst: a snapshot of every team and player as of "now"

The model of this project is trained on historical games where every feature is "blue minus red" and uses only
games BEFORE the match.  To analyse a match that has not been played, we need the same quantities for two teams today.
This file streams through all games in 01_data (2021 on) once, with the same rules as 02_features/*.py, and keeps the FINAL state:

  team     : team Elo, rolling mean of the last 10 games for each in-game stat, last 11 Elo values, last game date
  player   : player Elo, last team, games played (all leagues / majors / playoffs / international), first game date
  lineups  : each team's last starting five (by role) and its last 10 lineups
  games    : every game between major-league / international teams (for head-to-head and recent form)
  lanes    : for every pair of players who met in the same lane: dates and winners

The result is cached in cache/state.pkl and rebuilt only when a CSV in 01_data changes (first build: about 1-2 minutes).
"""
import glob
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "02_features"))
import build_features as bf                      # noqa: E402

INTL = {"MSI", "WLDs", "FST", "EWC"}
POS = ["top", "jng", "mid", "bot", "sup"]
START = pd.Timestamp("2021-01-01")               # same start as the features the model was trained on
K_ELO, REGRESS, WINDOW, MIN_GAMES = bf.ELO_K, bf.ELO_REGRESS, bf.WINDOW, bf.MIN_GAMES
CACHE = HERE / "cache" / "state.pkl"
USE = set(bf.RAW_COLS) | {"playername", "playerid"}


def _signature():
    return tuple((Path(f).name, Path(f).stat().st_size, int(Path(f).stat().st_mtime))
                 for f in sorted(glob.glob(str(ROOT / "01_data" / "*LoL_esports_match_data*.csv"))))


def _read_all(until=None):
    files = sorted(glob.glob(str(ROOT / "01_data" / "*LoL_esports_match_data*.csv")))
    if not files:
        raise SystemExit(f"No Oracle's Elixir CSVs in {ROOT / '01_data'}")
    T, P = [], []
    for f in files:
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in USE)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"] >= START]
        if until is not None:
            d = d[d["date"] < until]
        if d.empty:
            continue
        d["league"] = d["league"].replace(bf.LEAGUE_MAP)
        T.append(d[d["position"] == "team"]); P.append(d[d["position"].isin(POS)])
        print(f"   read {Path(f).name}")
    return pd.concat(T, ignore_index=True), pd.concat(P, ignore_index=True)


def build_state(until=None):
    print("Building the snapshot from 01_data (first time only) ...")
    T, P = _read_all(until)
    P["player"] = P["playerid"].fillna(P["playername"]).astype(str)
    P = P[P["player"].ne("nan")]
    P["team"] = P["teamid"].fillna(P["teamname"]).astype(str)
    P["po"] = P["playoffs"].fillna(0).astype(int)
    S = {"asof_data": T["date"].max()}

    # ------------------------------------------------------------ games table (major leagues + internationals)
    Tg = T[T["league"].isin(bf.LEAGUES | INTL)].copy()
    ok = Tg.groupby("gameid")["side"].agg(lambda s: tuple(sorted(s))) == ("Blue", "Red")
    Tg = Tg[Tg["gameid"].isin(ok[ok].index)]
    Tg["team"] = Tg["teamid"].fillna(Tg["teamname"]).astype(str)
    b, r = Tg[Tg.side == "Blue"].set_index("gameid"), Tg[Tg.side == "Red"].set_index("gameid")
    G = pd.DataFrame({"date": b["date"], "league": b["league"], "playoffs": b["playoffs"].fillna(0).astype(int),
                      "split": b["split"], "bkey": b["team"], "rkey": r["team"], "bname": b["teamname"], "rname": r["teamname"],
                      "blue_win": b["result"]}).reset_index().sort_values(["date", "gameid"]).reset_index(drop=True)
    S["games"] = G

    # ------------------------------------------------------------ team stats + Elo (major leagues only, like build_features)
    tm = Tg[Tg["league"].isin(bf.LEAGUES)].copy()
    tm["year"] = tm["date"].dt.year
    tm = tm.sort_values(["date", "gameid", "side"]).reset_index(drop=True)
    tm, stats, edges = bf.add_edges(tm)
    edges = edges.copy(); edges["team"] = tm["team"].values
    last = edges.groupby("team").tail(WINDOW).groupby("team")[stats].mean()
    cnt = edges.groupby("team").size()
    last.loc[last.index.map(lambda t: min(cnt[t], WINDOW) < MIN_GAMES), :] = np.nan
    S["team_form"], S["team_games"], S["stat_names"] = last, cnt, stats

    rating, last_year, hist = {}, {}, defaultdict(lambda: deque(maxlen=WINDOW + 1))
    bb, rr = tm[tm.side == "Blue"].set_index("gameid"), tm[tm.side == "Red"].set_index("gameid")
    for gid, date, year in tm.drop_duplicates("gameid")[["gameid", "date", "year"]].itertuples(index=False):
        bt, rt, res = bb.at[gid, "team"], rr.at[gid, "team"], bb.at[gid, "result"]
        for t in (bt, rt):
            if t not in rating:
                rating[t] = 1500.0
            elif last_year.get(t) != year:
                rating[t] = 1500 + (1 - REGRESS) * (rating[t] - 1500)
            last_year[t] = year
        exp_b = 1 / (1 + 10 ** ((rating[rt] - rating[bt]) / 400))
        d_ = K_ELO * (res - exp_b)
        rating[bt] += d_; rating[rt] -= d_
        hist[bt].append((date, rating[bt])); hist[rt].append((date, rating[rt]))
    S["team_elo"], S["team_elo_year"], S["team_elo_hist"] = rating, last_year, {k: list(v) for k, v in hist.items()}
    names = tm.sort_values("date").groupby("team").agg(name=("teamname", "last"), league=("league", "last"), last_game=("date", "max"))
    S["team_names"] = names

    # ------------------------------------------------------------ player Elo, lineups (major leagues, like roster_features)
    Pm = P[P["league"].isin(bf.LEAGUES)]
    n = Pm.groupby(["gameid", "side"])["player"].transform("size")
    Pm = Pm[n == 5]
    ok2 = Pm.groupby("gameid")["side"].nunique() == 2
    Pm = Pm[Pm["gameid"].isin(ok2[ok2].index)].sort_values(["date", "gameid"])
    prating, pyear, lineup_hist, lineup_pos, plast_team = {}, {}, defaultdict(lambda: deque(maxlen=10)), {}, {}
    for gid, g in Pm.groupby("gameid", sort=False):
        year, side = g["date"].iloc[0].year, {}
        for s in ("Blue", "Red"):
            gs = g[g.side == s]
            lineup = set(gs["player"])
            for p in lineup:
                if p not in prating:
                    prating[p] = 1500.0
                elif pyear.get(p) != year:
                    prating[p] = 1500 + (1 - REGRESS) * (prating[p] - 1500)
                pyear[p] = year
            side[s] = (gs["team"].iloc[0], lineup, gs["result"].iloc[0], np.mean([prating[p] for p in lineup]), gs)
        (bt, bl, bres, bpe, bgs), (rt, rl, _, rpe, rgs) = side["Blue"], side["Red"]
        delta = K_ELO * (bres - 1 / (1 + 10 ** ((rpe - bpe) / 400)))
        for p in bl: prating[p] += delta
        for p in rl: prating[p] -= delta
        for t, l, gs in ((bt, bl, bgs), (rt, rl, rgs)):
            lineup_hist[t].append(frozenset(l)); lineup_pos[t] = dict(zip(gs["position"], gs["player"])); 
            for p in l: plast_team[p] = t
    S["player_elo"], S["player_year"] = prating, pyear
    S["lineup_hist"], S["lineup_pos"], S["player_team"] = {k: list(v) for k, v in lineup_hist.items()}, lineup_pos, plast_team

    # ------------------------------------------------------------ player experience (ALL leagues, like experience_features)
    P["major"] = P["league"].isin(bf.LEAGUES | INTL).astype(int)
    P["intl"] = P["league"].isin(INTL).astype(int)
    ex = P.groupby("player").agg(name=("playername", "last"), n_all=("gameid", "size"), n_major=("major", "sum"), n_intl=("intl", "sum"),
                                 n_po=("po", "sum"), first=("date", "min"), last=("date", "max"))
    S["player_info"] = ex
    S["pname_to_pid"] = {str(nm).lower(): pid for pid, nm in ex["name"].items()}

    # ------------------------------------------------------------ same-lane head-to-head between players
    Ph = P[P["league"].isin(bf.LEAGUES | INTL) & P["gameid"].isin(set(G["gameid"]))]
    piv = Ph.pivot_table(index=["gameid", "position"], columns="side", values="player", aggfunc="first").dropna().reset_index()
    piv = piv.merge(G[["gameid", "date", "blue_win"]], on="gameid")
    lanes = defaultdict(list)
    for gid, pos, bl, rd, date, bw in piv[["gameid", "position", "Blue", "Red", "date", "blue_win"]].itertuples(index=False):
        a, c = (bl, rd) if bl < rd else (rd, bl)
        lanes[(a, c, pos)].append((date, bl if bw == 1 else rd))
    S["lanes"] = dict(lanes)
    return S


def load_state(force=False):
    sig = _signature()
    if not force and CACHE.exists():
        try:
            S = pickle.load(open(CACHE, "rb"))
            if S.get("sig") == sig:
                return S
        except Exception:
            pass
    S = build_state(); S["sig"] = sig
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(S, open(CACHE, "wb"))
    return S


# ---------------------------------------------------------------------- lookups
def find_team(S, query):
    """case-insensitive name match -> team key (most recently active team with that name)"""
    nm = S["team_names"]
    q = query.lower().strip()
    exact = nm[nm["name"].str.lower() == q]
    if exact.empty:
        exact = nm[nm["name"].str.lower().str.contains(q, regex=False)]
    if exact.empty:
        import difflib
        close = difflib.get_close_matches(q, nm["name"].str.lower().tolist(), n=3, cutoff=0.6)
        raise SystemExit(f"Team '{query}' not found." + (f" Did you mean: {', '.join(close)}?" if close else "") + "  (use --list-teams)")
    return exact.sort_values("last_game").index[-1]


def player_elo(S, pid, year):
    r = S["player_elo"].get(pid, 1500.0)
    if pid in S["player_elo"] and S["player_year"].get(pid) != year:
        r = 1500 + (1 - REGRESS) * (r - 1500)
    return r


def team_elo(S, key, year):
    r = S["team_elo"].get(key, 1500.0)
    if S["team_elo_year"].get(key) != year:
        r = 1500 + (1 - REGRESS) * (r - 1500)
    return r


def get_lineup(S, key, override=None):
    """role -> player id.  override: list of 5 player names in role order top,jng,mid,bot,sup"""
    base = dict(S["lineup_pos"].get(key, {}))
    if override:
        for pos, nm in zip(POS, override):
            nm = nm.strip()
            pid = S["pname_to_pid"].get(nm.lower())
            if pid is None:
                print(f"  note: player '{nm}' not found in the data, treated as a brand-new player")
                pid = f"new:{nm}"; S["player_info"].loc[pid] = [nm, 0, 0, 0, 0, pd.NaT, pd.NaT]
            base[pos] = pid
    return base


def experience(S, lineup, asof):
    info = S["player_info"]
    rows = [info.loc[p] if p in info.index else None for p in lineup.values()]
    z = lambda c: [0 if r is None else r[c] for r in rows]
    yrs = [0.0 if (r is None or pd.isna(r["first"])) else (asof - r["first"]).days / 365.25 for r in rows]
    return {"f_exp_games": float(np.mean(np.log1p(z("n_all")))), "f_exp_major": float(np.mean(np.log1p(z("n_major")))),
            "f_exp_playoffs": float(np.mean(np.log1p(z("n_po")))), "f_exp_intl": float(np.mean(np.log1p(z("n_intl")))),
            "f_exp_years": float(np.mean(yrs)), "f_rookies": float(sum(1 for v in z("n_major") if v < 30))}


def side_values(S, key, lineup, asof):
    """everything the model needs for ONE team (the model uses A minus B)"""
    form = S["team_form"].loc[key] if key in S["team_form"].index else pd.Series(dtype=float)
    v = {"f_" + c: form.get(c, np.nan) for c in S["stat_names"]}
    v["f_player_elo"] = float(np.mean([player_elo(S, p, asof.year) for p in lineup.values()]))
    v.update(experience(S, lineup, asof))
    return v
