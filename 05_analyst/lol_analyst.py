"""
lol_analyst.py  -  League of Legends matchup analyst (runs locally on the project's data in 05_analyst/data/)

    pip install pandas numpy scikit-learn matplotlib
    python export_data.py                                   once, and after each data refresh (writes data/)
    python lol_analyst.py "T1" "Gen.G"                      best-of-3, last known lineups
    python lol_analyst.py "T1" "Gen.G" --bo 5 --odds 1.80 2.05
    python lol_analyst.py "T1" "Gen.G" --a-roster "Zeus,Oner,Faker,Gumayusi,Keria"   (top,jng,mid,bot,sup)
    python lol_analyst.py --list-teams [text]

Covers the six major leagues (LCK, LPL, LEC, LCS, CBLOL, LCP) since 2021.  Data: Oracle's Elixir (oracleselixir.com).
Report: 1 win probability   2 why (12 factors)   3 lineups   4 head to head   5 recent form   6 market (with --odds)
        7 how often the model was right about these teams: true/false positives/negatives, also saved as 3 pie charts (PNG)
This is an analysis tool, not betting advice: the bookmaker's closing odds were a better forecast than this model in every test.
"""
import argparse
import contextlib
import io
import json
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
POS = ["top", "jng", "mid", "bot", "sup"]
LABELS = {  # feature -> (label, group, how to print the raw value)
    "f_player_elo": ("Player Elo (lineup strength)", "Lineup", "{:.0f}"),
    "f_gspd": ("Gold spread", "Form", "{:+.3f}"), "f_wcpm_edge": ("Ward clearing edge", "Form", "{:+.2f}"),
    "f_vspm_edge": ("Vision score edge", "Form", "{:+.2f}"), "f_firsttower": ("First-tower rate", "Form", "{:.0%}"),
    "f_firstdragon": ("First-dragon rate", "Form", "{:.0%}"), "f_firstblood": ("First-blood rate", "Form", "{:.0%}"),
    "f_dragon_edge": ("Dragon edge", "Form", "{:+.2f}"),
    "f_exp_games": ("Career games (log)", "Experience", "{:.2f}"), "f_exp_major": ("Major-league games (log)", "Experience", "{:.2f}"),
    "f_exp_playoffs": ("Playoff games (log)", "Experience", "{:.2f}"), "f_exp_intl": ("International games (log)", "Experience", "{:.2f}"),
    "f_exp_years": ("Years as a pro", "Experience", "{:.1f}"), "f_rookies": ("Rookies in lineup", "Experience", "{:.0f}"),
}
GROUPS = {"Lineup": "lineup strength", "Form": "recent in-game form (last 10 games)", "Experience": "experience"}
sig = lambda z: 1 / (1 + np.exp(-z))


# ====================================================================== data
def data_dir(args):
    d = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent / "data"
    if not (d / "manifest.json").exists():
        sys.exit(f"No data in {d}. Run  python export_data.py  in 05_analyst/ first (or pass --data-dir).")
    return d


class Data:
    def __init__(self, d):
        self.man = json.load(open(d / "manifest.json"))
        self.F, self.C = self.man["model"]["features"], self.man["model"]["C"]
        self.regress = self.man["elo"]["yearly_regression"]
        self.teams = pd.read_csv(d / "teams.csv", parse_dates=["last_game"]).set_index("team_key")
        self.players = pd.read_csv(d / "players.csv", parse_dates=["first", "last"]).set_index("player_key")
        lu = pd.read_csv(d / "lineups.csv")
        self.lineups = {k: dict(zip(g.role, g.player_key)) for k, g in lu.groupby("team_key")}
        self.games = pd.read_csv(d / "games.csv.gz", parse_dates=["date"])
        self.lanes = pd.read_csv(d / "lanes.csv.gz", parse_dates=["date"])
        self.train = pd.read_csv(d / "train.csv.gz", parse_dates=["date"])
        self.pred = pd.read_csv(d / "predictions.csv.gz", parse_dates=["date"])
        self.until = pd.Timestamp(self.man["data_until"])
        p = self.players.sort_values("last")
        self.pname = {str(n).lower(): k for k, n in p["name"].items()}      # latest player with that name wins

    # ---------------------------------------------------------------- lookups
    def find_team(self, q):
        nm = self.teams["name"].astype(str); ql = q.lower().strip()
        hit = self.teams[nm.str.lower() == ql]
        if hit.empty: hit = self.teams[nm.str.lower().str.contains(ql, regex=False)]
        if hit.empty:
            import difflib
            close = difflib.get_close_matches(ql, nm.str.lower().tolist(), n=3, cutoff=0.6)
            sys.exit(f"Team '{q}' not found." + (f" Did you mean: {', '.join(close)}?" if close else "") + "  (try --list-teams)")
        return hit.sort_values("last_game").index[-1]

    def elo(self, rating, year_last, year):
        if pd.isna(rating): return 1500.0
        return rating if year_last == year else 1500 + (1 - self.regress) * (rating - 1500)

    def player_elo(self, pid, year):
        if pid not in self.players.index: return 1500.0
        r = self.players.loc[pid]; return self.elo(r["elo"], r["elo_year"], year)

    def team_elo(self, key, year):
        r = self.teams.loc[key]; return self.elo(r["elo"], r["elo_year"], year)

    def lineup(self, key, override=None):
        base = dict(self.lineups.get(key, {}))
        if override:
            for pos, nm in zip(POS, override):
                pid = self.pname.get(nm.strip().lower())
                if pid is None:
                    print(f"  note: player '{nm.strip()}' not in the data, treated as a brand-new player")
                    pid = "new:" + nm.strip()
                    self.players.loc[pid, ["name", "n_all", "n_major", "n_intl", "n_po"]] = [nm.strip(), 0, 0, 0, 0]
                base[pos] = pid
        return base

    def side_values(self, key, lineup, asof):
        v = {}
        for f in self.F:
            col = "form_" + f[2:]
            if col in self.teams.columns: v[f] = self.teams.at[key, col]
        P = [self.players.loc[p] if p in self.players.index else None for p in lineup.values()]
        g = lambda c: [0.0 if r is None or pd.isna(r[c]) else float(r[c]) for r in P]
        yrs = [0.0 if (r is None or pd.isna(r["first"])) else (asof - r["first"]).days / 365.25 for r in P]
        v.update({"f_player_elo": float(np.mean([self.player_elo(p, asof.year) for p in lineup.values()])),
                  "f_exp_games": float(np.mean(np.log1p(g("n_all")))), "f_exp_major": float(np.mean(np.log1p(g("n_major")))),
                  "f_exp_playoffs": float(np.mean(np.log1p(g("n_po")))), "f_exp_intl": float(np.mean(np.log1p(g("n_intl")))),
                  "f_exp_years": float(np.mean(yrs)), "f_rookies": float(sum(1 for x in g("n_major") if x < 30))})
        return v


# ====================================================================== model
def fit(train, F, C, idx=None):
    t = train if idx is None else train.iloc[idx]
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(t[F], t.blue_win)


def contributions(model, x):
    """side-neutral: logit(A beats B) = sum_j w_j x_j / s_j  (blue-side intercept and feature means cancel)"""
    sc, lr = model[0], model[-1]
    return lr.coef_[0] * x / sc.scale_, lr.intercept_[0] - float(np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))


def series(p):
    return {1: p, 3: p * p * (3 - 2 * p), 5: p ** 3 * (10 - 15 * p + 6 * p * p)}


def table(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    ln = lambda r: "  ".join(str(c).ljust(w[i]) if i < 2 else str(c).rjust(w[i]) for i, c in enumerate(r))
    return "\n".join(["   " + ln(header), "   " + "  ".join("-" * x for x in w)] + ["   " + ln(r) for r in rows])


# ====================================================================== true / false positives
def team_view(pred, team, opp=None):
    """one row per game from TEAM's point of view: p = model probability that TEAM wins, won = TEAM won"""
    b = pred[pred.blue_team == team]; r = pred[pred.red_team == team]
    if opp is not None:
        b, r = b[b.red_team == opp], r[r.blue_team == opp]
    v = pd.concat([pd.DataFrame({"date": b.date, "p": b.p_blue, "won": b.blue_win == 1}),
                   pd.DataFrame({"date": r.date, "p": 1 - r.p_blue, "won": r.blue_win == 0})])
    return v.sort_values("date")


def confusion(v):
    pw = v.p > 0.5
    return {"TP": int((pw & v.won).sum()), "FP": int((pw & ~v.won).sum()), "TN": int((~pw & ~v.won).sum()), "FN": int((~pw & v.won).sum())}


CONF_TXT = {"TP": "said WIN, won", "FP": "said WIN, lost", "TN": "said LOSS, lost", "FN": "said LOSS, won"}
CONF_COL = {"TP": "#2a78d6", "TN": "#86b6ef", "FP": "#eb6834", "FN": "#f5b08f"}   # correct = blues, wrong = oranges


def pies(groups, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("   (install matplotlib to also get the pie charts as a PNG)"); return None
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "legend.fontsize": 10,
                         "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})       # = assets/figstyle.mplstyle
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for ax, (name, c) in zip(axes, groups):
        n = sum(c.values())
        if n == 0:
            ax.text(0, 0, "no games\nin the data", ha="center", va="center", color="#52514e"); ax.set_axis_off()
            ax.set_title(name, fontsize=11); continue
        keys = [k for k in ["TP", "TN", "FN", "FP"] if c[k] > 0]
        ax.pie([c[k] for k in keys], colors=[CONF_COL[k] for k in keys], startangle=90, counterclock=False,
               wedgeprops={"edgecolor": "white", "linewidth": 2},
               autopct=lambda pct: f"{pct:.0f}%\n({round(pct * n / 100)})" if pct >= 5 else "",
               pctdistance=0.68, textprops={"color": "#0b0b0b", "fontsize": 10})
        acc = (c["TP"] + c["TN"]) / n
        ax.set_title(f"{name}\n{n} games, model right {acc:.0%}", fontsize=11)
    fig.legend(handles=[Patch(color=CONF_COL[k], label=f"{k}: {CONF_TXT[k]}") for k in ["TP", "TN", "FN", "FP"]],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.subplots_adjust(bottom=0.14)
    path.parent.mkdir(exist_ok=True); fig.savefig(path); plt.close(fig)
    return path


# ====================================================================== PDF report
RESULT = {}          # filled by main(): title, output folder, pie-chart PNG
TEE = None


class Tee:
    """copies everything printed to the terminal into a buffer, so the PDF gets the same text"""
    def __init__(self, out):
        self.out, self.buf = out, io.StringIO()
    def write(self, s):
        self.out.write(s); self.buf.write(s); return len(s)
    def flush(self):
        self.out.flush()


def safe(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)


GUIDE = [
    ("1) Win probability", [
        "One game: the model's chance that the first team wins one game (averaged over both sides). 30.8% = about 3 games in 10.",
        "90% range: the model is refit 200 times on resampled training data; the range is the 5th to 95th percentile. Narrow = stable.",
        "Bo1 / Bo3 / Bo5: chance of winning a series of that length, treating games as independent. The underdog's chance falls as the series gets longer.",
        "Side: the blue-side advantage for two equal teams, and the first team's chance on blue and on red.",
        "Track record: past games (2024 on, each predicted before it was played) where the favourite had about this confidence, and how often the favourite won.",
    ]),
    ("2) Why: factors", [
        "Each row is one factor. The raw values are shown for each team. 'effect' is the change in the first team's probability, in percentage points, when that factor is set to zero. Negative = favours the second team.",
        "Factors overlap, so the effects do not add up exactly to the gap from 50%. Read the group totals first.",
        "Lineup strength: mean Elo of the five starters. Elo is a rating that rises after wins against strong opponents; 1500 is average.",
        "Recent form (last 10 games): gold spread, ward clearing edge, vision edge, first tower rate, first dragon rate. 'Edge' = team value minus opponent value; rates are shares of games (0% to 100%).",
        "Experience: averages over the five starters of log(1 + games), years as a pro, and rookies (players with fewer than 30 major-league games).",
        "Team Elo is shown for context only; it is not one of the model's factors.",
    ]),
    ("3) Lineups", [
        "Last known starting five per team, with Elo and major-league games. 'rookie' = under 30 major-league games; 'NEW' = not in the team's last lineup.",
    ]),
    ("4) Head to head and lanes", [
        "All games since 2021 between the two teams (major leagues and international events). Games on the same day are counted as one series.",
        "Lane by lane: the record of the two players in the same role when they met.",
        "Head to head adds no predictive power beyond team strength.",
    ]),
    ("5) Recent form", [
        "Last 10 results (W = win, L = loss), record, current streak, Elo change over those games, and days since the last game (flagged after 60 days).",
    ]),
    ("6) Market (with --odds)", [
        "Margin: the bookmaker's built-in profit, from the two decimal odds. Margin-free: the market's chance after removing the margin, to compare with the model.",
    ]),
    ("7) True / false positives", [
        "Walk-forward predictions: each game was predicted by a model trained only on earlier years, before the game was played.",
        "'Said win' = the model gave the team more than 50%. TP: said win and won. FP: said win and lost. TN: said loss and lost. FN: said loss and won.",
        "Right = (TP + TN) / all games. In the pie charts, blue slices are correct predictions and orange slices are wrong ones.",
    ]),
    ("Limits", [
        "Analysis tool, not betting advice. Pre-draft only: the draft, patch, injuries and scrims are not included. Cross-region matches are rough.",
    ]),
]


def write_pdf(title, text, png, path):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    W, H = 8.27, 11.69                                        # A4, portrait
    body = []
    for ln in text.splitlines():
        body += textwrap.wrap(ln, 120, replace_whitespace=False, drop_whitespace=False) or [""]
    pages = [body[i:i + 74] for i in range(0, len(body), 74)]
    with PdfPages(path) as pdf:
        # 1) the printed report, as it appears in the terminal
        for k, chunk in enumerate(pages):
            fig = plt.figure(figsize=(W, H))
            top = 0.93
            if k == 0:
                fig.text(0.06, 0.955, title, fontsize=18, weight="bold")
                fig.text(0.06, 0.935, "LoL matchup analyst report", fontsize=9, color="#52514e")
                top = 0.91
            fig.text(0.06, top, "\n".join(chunk), family="DejaVu Sans Mono", fontsize=6.5, va="top", linespacing=1.3)
            fig.text(0.5, 0.02, f"page {k + 1}", fontsize=8, ha="center", color="#52514e")
            pdf.savefig(fig); plt.close(fig)
        # 2) the pie charts
        if png and Path(png).exists():
            fig = plt.figure(figsize=(W, H))
            fig.text(0.06, 0.955, f"Pie charts: {title}", fontsize=14, weight="bold")
            ax = fig.add_axes([0.03, 0.50, 0.94, 0.40])
            ax.imshow(mpimg.imread(png)); ax.axis("off")
            fig.text(0.06, 0.44, "Left: the first team's games. Middle: the second team's games.", fontsize=9.5)
            fig.text(0.06, 0.41, "Right: games between the two teams, from the first team's side.", fontsize=9.5)
            fig.text(0.06, 0.38, "Blue = model right, orange = model wrong. Labels show the share and the number of games.", fontsize=9.5)
            pdf.savefig(fig); plt.close(fig)
        # 3) what each number means
        items = []
        for head, bullets in GUIDE:
            items.append((head, True))
            for b in bullets:
                for j, ln in enumerate(textwrap.wrap(b, 105)):
                    items.append((("  - " if j == 0 else "    ") + ln, False))
            items.append(("", False))
        for i in range(0, len(items), 50):
            fig = plt.figure(figsize=(W, H))
            y = 0.94
            if i == 0:
                fig.text(0.06, 0.955, "How to read the numbers", fontsize=14, weight="bold")
                y = 0.92
            for ln, bold in items[i:i + 50]:
                fig.text(0.06, y, ln, fontsize=9.5, weight="bold" if bold else "normal")
                y -= 0.0165
            pdf.savefig(fig); plt.close(fig)


# ====================================================================== report
def main():
    ap = argparse.ArgumentParser(description="League of Legends matchup analyst")
    ap.add_argument("team_a", nargs="?"); ap.add_argument("team_b", nargs="?")
    ap.add_argument("--bo", type=int, default=3, choices=[1, 3, 5]); ap.add_argument("--date", help="analyse as of this date (YYYY-MM-DD)")
    ap.add_argument("--a-roster", help="5 names, top,jng,mid,bot,sup"); ap.add_argument("--b-roster")
    ap.add_argument("--odds", nargs=2, type=float, metavar=("ODDS_A", "ODDS_B"), help="bookmaker decimal odds, team A first")
    ap.add_argument("--list-teams", nargs="?", const="", metavar="TEXT")
    ap.add_argument("--data-dir"); ap.add_argument("--no-range", action="store_true"); ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="reports", help="folder for the pie-chart PNG (default ./reports)")
    a = ap.parse_args()
    D = Data(data_dir(a))
    if a.list_teams is not None:
        t = D.teams[D.teams["name"].astype(str).str.lower().str.contains(a.list_teams.lower(), regex=False)]
        t = t[t.last_game >= D.until - pd.Timedelta(days=365)]
        for lg, g in t.sort_values("name").groupby("league"): print(f"{lg:6s}", ", ".join(g["name"]))
        print("(teams with a game in the last 12 months)"); return
    if not (a.team_a and a.team_b): ap.error("give two team names, or use --list-teams")
    RESULT.update(title=f"{a.team_a} vs {a.team_b}", outdir=Path(a.out))

    asof = pd.Timestamp(a.date) if a.date else max(pd.Timestamp.today().normalize(), D.until + pd.Timedelta(days=1))
    ka, kb = D.find_team(a.team_a), D.find_team(a.team_b)
    na, nb = str(D.teams.at[ka, "name"]), str(D.teams.at[kb, "name"])
    la, lb = D.lineup(ka, a.a_roster.split(",") if a.a_roster else None), D.lineup(kb, a.b_roster.split(",") if a.b_roster else None)
    F = D.F
    model = fit(D.train, F, D.C)
    va, vb = D.side_values(ka, la, asof), D.side_values(kb, lb, asof)
    missing = [f for f in F if pd.isna(va.get(f)) or pd.isna(vb.get(f))]
    x = np.array([0.0 if f in missing else va[f] - vb[f] for f in F])
    c, side = contributions(model, x); p = float(sig(c.sum()))
    bar = "=" * 78
    print(f"\n{bar}\n  {na}  vs  {nb}     best-of-{a.bo}     as of {asof.date()}   (data until {D.until.date()})\n{bar}")
    if (asof - D.until).days > 30: print(f"  WARNING: data ends {D.until.date()}; rebuild the features and rerun export_data.py for newer data.")
    if missing: print(f"  WARNING: too few recent games for: {', '.join(missing)} (treated as equal)")
    if D.teams.at[ka, "league"] != D.teams.at[kb, "league"]:
        print(f"  WARNING: cross-region match ({D.teams.at[ka, 'league']} vs {D.teams.at[kb, 'league']}). Ratings are earned mostly inside each league;")
        print("  the gap between regions comes only from the few international games, so treat the probability as rough.")

    # ---- 1
    sp = series(p)
    print("\n1) WIN PROBABILITY  (side-neutral)")
    rng_txt = ""
    if not a.no_range:
        r = np.random.default_rng(0); vals = []
        for _ in range(200):
            m = fit(D.train, F, D.C, r.integers(0, len(D.train), len(D.train))); vals.append(sig(contributions(m, x)[0].sum()))
        lo, hi = np.percentile(vals, [5, 95]); rng_txt = f"     (90% range for {na}: {lo:.0%} - {hi:.0%})"
    print(f"   {na}: {p:.1%}   |   {nb}: {1 - p:.1%}   for one game{rng_txt}")
    print(f"   Bo1 {sp[1]:.1%}   Bo3 {sp[3]:.1%}   Bo5 {sp[5]:.1%}   (series assume independent games)   >> Bo{a.bo}: {na} {sp[a.bo]:.1%}")
    print(f"   Side: blue is worth {sig(side) - .5:+.1%} for equal teams;  {na} on blue {sig(c.sum() + side):.1%}, on red {sig(c.sum() - side):.1%}")
    conf = max(p, 1 - p); pr = D.pred; pc = np.maximum(pr.p_blue, 1 - pr.p_blue); s = pr[(pc - conf).abs() <= 0.05]
    if len(s) >= 30:
        right = ((s.p_blue > .5) == (s.blue_win == 1)).mean()
        print(f"   Track record: when the model's favourite was at about {conf:.0%}, it won {right:.0%} of {len(s):,} past games ({pr.date.min().year}-{pr.date.max().year}).")

    # ---- 2
    print(f"\n2) WHY  (points of win probability each factor adds for {na}; negative = favours {nb})")
    rows, grp = [], {}
    for f, cj in zip(F, c):
        lab, gname, fmt = LABELS.get(f, (f, "Form", "{:.3f}"))
        eff = p - sig(c.sum() - cj); grp[gname] = grp.get(gname, 0) + eff
        show = lambda v: "n/a" if pd.isna(v) else fmt.format(v)
        rows.append([f"{gname}: {lab}", "", show(va.get(f)), show(vb.get(f)), f"{eff * 100:+.1f}"])
    print(table([[r[0], r[1], r[2], r[3], r[4]] for r in rows], ["factor", "", na[:14], nb[:14], "effect"]))
    print("   By group: " + "; ".join(f"{GROUPS[g]} {v * 100:+.1f}" for g, v in sorted(grp.items(), key=lambda kv: -abs(kv[1]))))
    print("   Effects are one factor at a time and overlap, so they do not add up exactly; a factor can point against its raw numbers")
    print("   because the model weighs overlapping factors jointly. Read the group totals first.")
    print(f"   Context (not in the model): team Elo {na} {D.team_elo(ka, asof.year):.0f} vs {nb} {D.team_elo(kb, asof.year):.0f}")

    # ---- 3
    print("\n3) LINEUPS  (last starting five; change with --a-roster / --b-roster)")
    for key, nm, lu in ((ka, na, la), (kb, nb, lb)):
        default = set(D.lineups.get(key, {}).values()); rows = []
        for pos in POS:
            pid = lu.get(pos)
            if pid is None: rows.append([pos, "?", "", "", ""]); continue
            r = D.players.loc[pid] if pid in D.players.index else None
            flags = (["rookie"] if r is not None and r["n_major"] < 30 else []) + (["NEW"] if default and pid not in default else [])
            rows.append([pos, r["name"] if r is not None else pid, f"{D.player_elo(pid, asof.year):.0f}",
                         f"{0 if r is None else int(r['n_major'])}", ", ".join(flags)])
        print(f"   {nm}"); print(table(rows, ["role", "player", "Elo", "major games", "note"]))

    # ---- 4
    G = D.games
    m = G[((G.bkey == ka) & (G.rkey == kb)) | ((G.bkey == kb) & (G.rkey == ka))].copy()
    print("\n4) HEAD TO HEAD  (since 2021, major leagues + internationals; adds no predictive power beyond team strength)")
    if m.empty:
        print("   never met")
    else:
        m["a_won"] = np.where(m.bkey == ka, m.blue_win == 1, m.blue_win == 0); m["day"] = m.date.dt.date
        ser = m.groupby("day").agg(league=("league", "first"), po=("playoffs", "max"), g=("a_won", "size"), w=("a_won", "sum")).reset_index()
        rec = m[m.date >= asof - pd.Timedelta(days=730)]
        print(f"   games {na} {int(m.a_won.sum())} - {int((~m.a_won).sum())} {nb}   series {int((ser.w * 2 > ser.g).sum())} - {int((ser.w * 2 < ser.g).sum())}"
              f"   last 2 years {int(rec.a_won.sum())} - {int((~rec.a_won).sum())}")
        for r in ser.sort_values("day").tail(6).iloc[::-1].itertuples():
            print(f"   {r.day}  {r.league:5s}{' playoffs' if r.po else '':9s} {na} {int(r.w)}-{int(r.g - r.w)} {nb}")
    rows = []
    L = D.lanes
    for pos in POS:
        pa, pb = la.get(pos), lb.get(pos)
        if pa is None or pb is None: continue
        x1, x2 = (pa, pb) if pa < pb else (pb, pa)
        h = L[(L.player_a == x1) & (L.player_b == x2) & (L.role == pos)]
        nm_ = lambda q: D.players.at[q, "name"] if q in D.players.index else q
        rows.append([pos, f"{nm_(pa)} v {nm_(pb)}", f"{int((h.winner == pa).sum())}-{int((h.winner == pb).sum())}" if len(h) else "never met",
                     str(h.date.max().date()) if len(h) else ""])
    print("   lane by lane (same two players, same role):"); print(table(rows, ["role", "players", "record", "last met"]))

    # ---- 5
    print("\n5) RECENT FORM")
    for key, nm in ((ka, na), (kb, nb)):
        g = G[(G.bkey == key) | (G.rkey == key)].sort_values("date").tail(10).copy()
        if g.empty: print(f"   {nm}: no games"); continue
        g["won"] = np.where(g.bkey == key, g.blue_win == 1, g.blue_win == 0); g["opp"] = np.where(g.bkey == key, g.rname, g.bname)
        s_ = "".join("W" if w else "L" for w in g.won); k = len(s_) - len(s_.rstrip(s_[-1]))
        rest = (asof - g.date.max()).days; t = D.teams.loc[key]
        print(f"   {nm}: last {len(g)} {s_} ({int(g.won.sum())}-{int((~g.won).sum())}), streak {k}{s_[-1]}, "
              f"Elo {t['elo_change']:+.0f} over {int(t['elo_change_games'])} games, last game {g.date.max().date()} ({rest} days ago)"
              + ("  <- long break" if rest > 60 else ""))

    # ---- 6
    if a.odds:
        oa, ob = a.odds; z = 1 / oa + 1 / ob; qa = (1 / oa) / z; ps = sp[a.bo]
        print(f"\n6) MARKET  odds {oa} / {ob}: margin {z - 1:.1%}, margin-free {na} {qa:.1%}")
        print(f"   model Bo{a.bo} {ps:.1%} vs market {qa:.1%}: {100 * (ps - qa):+.1f} pts. The market was the better forecast in every test of this project;")
        print("   a large gap usually means it knows something the model does not (stand-in, patch, scrim form, draft).")

    # ---- 7
    va_, vb_, vab = team_view(D.pred, na), team_view(D.pred, nb), team_view(D.pred, na, nb)
    groups = [(f"{na}: all games", confusion(va_)), (f"{nb}: all games", confusion(vb_)), (f"{na} vs {nb}\n(from {na}'s side)", confusion(vab))]
    span = f"{D.pred.date.min().date()} to {D.pred.date.max().date()}"
    print(f"\n7) HOW OFTEN WAS THE MODEL RIGHT ABOUT THESE TEAMS?  (walk-forward predictions {span}, each made before the game)")
    print("   positive = model said the team wins (above 50%).  TP said WIN & won, FP said WIN & lost, TN said LOSS & lost, FN said LOSS & won")
    rows = []
    for name, cc in groups:
        n = sum(cc.values())
        pctf = lambda k: f"{cc[k]} ({cc[k] / n:.0%})" if n else "0"
        rows.append([name.replace("\n", " "), "", pctf("TP"), pctf("FP"), pctf("TN"), pctf("FN"), f"{(cc['TP'] + cc['TN']) / n:.0%}" if n else "-"])
    print(table(rows, ["", "", "TP", "FP", "TN", "FN", "right"]))
    if not a.no_plot:
        safe = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)
        out = pies(groups, f"Model predictions {span}: right (blue) vs wrong (orange)", Path(a.out) / f"{safe(na)}_vs_{safe(nb)}.png")
        if out:
            RESULT["png"] = out
            print(f"   pie charts saved: {out.resolve()}")
    print(f"\n{bar}\nModel: pre-draft logistic regression, {len(F)} factors, trained on {len(D.train):,} games {D.train.date.min().year}-{D.train.date.max().year}."
          f"\nNot included: draft, patch, injuries, scrims.  Data: Oracle's Elixir\n{bar}\n")


if __name__ == "__main__":
    TEE = Tee(sys.stdout)
    with contextlib.redirect_stdout(TEE):
        main()
    if RESULT:
        RESULT["outdir"].mkdir(exist_ok=True)
        pdf_path = RESULT["outdir"] / f"{safe(RESULT['title'])}.pdf"
        write_pdf(RESULT["title"], TEE.buf.getvalue(), RESULT.get("png"), pdf_path)
        print(f"   report PDF saved: {pdf_path.resolve()}")
