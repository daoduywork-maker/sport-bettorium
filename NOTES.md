# Working notes

## Environment
- Windows has no Python; everything runs in **WSL** (Ubuntu, Python 3.14) inside a venv.
- Project path in WSL: `/mnt/c/Users/ACER/OneDrive/Desktop/Project-0`
- Packages: `pandas numpy scipy scikit-learn optuna joblib matplotlib`
- OddsPapi key in `oddspapi_key.txt` (never commit, never paste). Network to OddsPapi is unstable from China: use a VPN; `fetch_odds.py` retries and caches.

## Decisions (and why)
| Date | Decision | Why |
|---|---|---|
| 2026-10-13 | Oracle's Elixir instead of gol.gg | Free CSV, all leagues; gol.gg has no bulk export, same LPL gaps, stale since a Riot bug |
| 2026-10-13 | Leagues: LCS, LEC, LCK, LPL, CBLOL, LCP; years 2024–2026 | Major leagues; 3 years ≈ same meta era |
| 2026-10-13 | 2025 LTA N → LCS, LTA S → CBLOL | LTA replaced LCS/CBLOL in 2025, mostly the same teams |
| 2026-10-13 | Full-game stats only (no @15, no CS/min, first Herald/Baron) | Missing for LPL |
| 2026-10-13 | `year` taken from the calendar date | Oracle labels autumn qualifiers as next season |
| 2026-10-13 | KS drop rule: p < 0.01 **and** D > 0.15 | p-value alone dropped 12/17 features incl. Elo with ~2,000 games/year |
| 2026-10-13 | Series probability via $P_{\text{Bo}n}(p)$, p averaged over both sides | Bookmakers price series, model predicts games |
| 2026-10-13 | Odds: Pinnacle closing price, markets 181 (series) and 1847 (map 1) | Pinnacle has by far the most updates; sharpest line |

## OddsPapi facts
- Base `https://api.oddspapi.io/v4`, `?apiKey=`; Cloudflare blocks Python's default User-Agent (error 1010) → send a browser UA.
- LoL `sportId = 18`. Tournaments: LCS 2450, LEC 2452, LCK 2454, LPL 15490 / 39985, CBLOL 26698, LCP 45589.
- Markets: 181 Winner (outcomes 181 = team1, 182 = team2); 1847 First Map Winner (1847/1848).
- Historical odds since Jan 2026; `/historical-odds` cooldown 5 s; 404 `FIXTURE_NOT_FOUND` = empty, not an error.

## Commands
```bash
cd 02_features && python build_features.py
cd 03_model_selection && python pipeline.py          # --trials 10 for a quick run
python series_probs.py && python plot_reliability.py
cd 04_odds && python fetch_odds.py --explore                       # quota, ids, markets
python fetch_odds.py --market 181,1847               # full download, resumable
```

## Figures
- All figures use `assets/figstyle.mplstyle` (`plt.style.use("../assets/figstyle.mplstyle")`): DejaVu Sans 11 pt, same line widths.

## Open checklist
- [x] odds download finished → moved to `04_odds/`
- [ ] fixture ↔ series matching (team-name aliases)
- [ ] `simulate.py`
- [ ] log-loss branch
- [ ] copy the paper PDF to `Literatures/Calibration_betting-Walsh2024.pdf`
- [ ] delete `_to_delete/`
