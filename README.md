# LoL matchup analyst

Type two teams and get a report: the model's win probability, the factors behind it, the starting lineups, head-to-head and recent form, and how often the model was right about these teams, shown as pie charts.

The analyst runs locally on data in `05_analyst/data/`. It makes no network calls.

![T1 vs Gen.G pie charts](assets/analyst_pies_example.png)

## Quick start

In WSL, inside the virtual environment, from `05_analyst/`:

```bash
pip install pandas numpy scikit-learn matplotlib
python export_data.py                       # once, and after each data refresh (about 1 minute)
python lol_analyst.py "T1" "Gen.G"          # best-of-3 with last known lineups
python lol_analyst.py --list-teams lck      # find team names
```

Each run also writes `reports/<team A> vs <team B>.pdf`: the printed report, the pie charts, and a guide that explains each number.

| Option | Effect |
|---|---|
| `--bo 1/3/5` | series length (default 3) |
| `--odds 1.80 2.05` | compare with bookmaker decimal odds (team A first) |
| `--a-roster "Zeus,Oner,Faker,Gumayusi,Keria"` | override a lineup, in role order top, jng, mid, bot, sup |
| `--date 2026-11-01` | analyse as of another date (affects Elo resets and years as a pro) |
| `--no-range`, `--no-plot`, `--out DIR` | skip the 90% range, skip the pie charts, or change the output folder (default `./reports`) |
| `--data-dir DIR` | read data from another folder |

## Where every number comes from

The app reads a snapshot of every team and player as of today. The snapshot is built from Oracle's Elixir match data (since 2021) by `05_analyst/state.py`, using the same rules as the training features in `02_features/`. `validate_snapshot.py` checks that the snapshot reproduces the training features exactly (all differences 0.000).

### Team and player Elo

Elo is a rating that rises when a team or player beats an expected opponent and falls otherwise.

- Start at 1500. Each game, with `E` = expected score of the first side = 1 / (1 + 10^((R_opponent - R_self) / 400)):
  `R_new = R_old + K × (result − E)`, with `K = 24`, `result` = 1 for a win, 0 for a loss.
- Each calendar year, the first time a team or player plays, the rating moves 25% back toward 1500: `R = 1500 + 0.75 × (R − 1500)`. This accounts for roster changes.
- Team Elo uses major-league games (LCK, LPL, LEC, LCS, CBLOL, LCP). In 2025 LTA North is mapped to LCS and LTA South to CBLOL.
- Player Elo is updated on the lineup. Each game, the expected score comes from the mean Elo of each team's five starters, and the same change is added to (or subtracted from) every starter's rating.
- The analyst reports the team Elo as context. It is not one of the 12 model factors.

### Recent form (last 10 games)

For each team, these are averages over its last 10 games (the rolling window in `build_features.py`). A team needs at least 5 past games, or the factor is treated as equal (zero difference) and flagged in the output.

| Factor | Per-game value | Meaning |
|---|---|---|
| `gspd` | Oracle's gold-spread column | Gold lead, as provided by the source |
| `wcpm_edge` | team wards cleared per minute − opponent's | Ward clearing edge |
| `vspm_edge` | team vision score per minute − opponent's | Vision edge |
| `firsttower` | 1 if the team took the first tower, else 0 | Rate of first tower |
| `firstdragon` | 1 if the team took the first dragon, else 0 | Rate of first dragon |

Only games before the one being predicted are used, so the features contain no information from the future.

### Lineup strength

`f_player_elo` is the mean Elo of the team's five starters. The starters are the last lineup seen for that team (or the lineup given with `--a-roster` / `--b-roster`).

### Experience

For each of the five starters, the app counts previous games in the data since 2021, then averages across the five players:

| Factor | Per player | Team value |
|---|---|---|
| `exp_games` | ln(1 + games in all leagues) | mean |
| `exp_major` | ln(1 + major-league games) | mean |
| `exp_playoffs` | ln(1 + playoff games) | mean |
| `exp_intl` | ln(1 + international games) | mean |
| `exp_years` | years since first game in the data | mean |
| `rookies` | 1 if fewer than 30 major-league games | count out of 5 |

The logarithm makes the first games count more than later ones: going from 0 to 100 games matters more than going from 1000 to 1100.

Each starter's flags are printed in the lineup table: `rookie` marks fewer than 30 major-league games, and `NEW` marks a player who is not in the team's last lineup.

## How the 12 features were chosen

The features come from a model-selection pipeline (`03_model_selection/pipeline.py`) with a fixed time split. Training: 2022-2024. Validation: 2025. Test: January to May 2026. Each decision is logged in `03_model_selection/results/selection_log.txt`.

**Step 1: candidate pool (`02_features/build_features.py`).** For each team and game, the edge stats are team value minus opponent value: kills, dragons, void grubs, barons, towers, inhibitors, damage per minute, vision per minute, gold per minute, wards cleared per minute, gold spread, first blood, first dragon, first tower, game length, win. Each team's value is the average of its previous 10 games (the current game is excluded). Team Elo is added. The game row is blue minus red: 17 candidates. Excluded because some leagues lack them: 15-minute stats, CS per minute, first Herald and first Baron.

**Step 2: covariate-shift check (Kolmogorov-Smirnov test, training vs validation).** A feature is dropped only if `p < 0.01` and the KS statistic `D > 0.15`. With about 2,000 games a year, almost any small shift gives `p < 0.01`, so the size threshold is needed. `void grubs` was dropped (p ≈ 1e-163, D = 0.33). Without the size threshold, team Elo would also have been dropped. 16 features remain.

**Step 3: redundancy filter.** For each pair with Spearman |ρ| > 0.7, the feature less correlated with the result is dropped. Seven were dropped: gpm edge, tower edge, dpm edge, win, inhibitor edge, kill edge, baron edge. 9 features remain: team Elo, gold spread, vision edge, dragon edge, first tower, ward clearing edge, first dragon, game length, first blood.

**Step 4: forward selection (greedy).** Starting from nothing, each step adds the feature that most lowers validation log-loss, using logistic regression. The search runs to the end, and the subset with the lowest validation score is kept. The selected subset was team Elo, gold spread, ward clearing edge, vision edge, first tower and first dragon. Dropped: dragon edge, first blood and game length.

**Step 5: tuning.** `C` was tuned as above, on validation log-loss. On the test period the logistic model got log-loss 0.6434 and accuracy 63.3%.

**Changes made afterwards, for the analyst.** The pipeline did not choose the last two blocks below. They were added in later experiments in `02_features/roster_features.py`, `02_features/experience_features.py` and their experiment scripts.

| Block | Features | Why |
|---|---|---|
| Lineup strength | player Elo (mean of the five starters) replaces team Elo | Lower test log-loss, most in January to March (−0.005), and a small gain over the year (−0.002). Players move between teams, so a player's rating follows them |
| Experience | games in all leagues, major and international games, playoff games, years in the data, rookies (< 30 major games). Each as the mean over the five starters | All six together lowered log-loss by 0.003 to 0.005 in every period. Individually each helped less |

Result: 6 pipeline features − team Elo + player Elo + 6 experience features = **12 features**. That is the list in the analyst's section 2, grouped as lineup (1), recent form (5) and experience (6).

**Caveat.** The experience and player-Elo choices were made by looking at the 2026 results, which are also the periods reported in those experiment tables. The 12-feature model therefore has no clean holdout. Its 2026 numbers are optimistic. A clean test needs games played after the choice, which will become available as new data arrives.

**Considered and not used:**

- **Draft features** (champion win rate, player comfort on the champion): the largest single gain (log-loss −0.005), but the draft is known only about 10 minutes before the game. Useful for a map bet placed after the draft, not for a pre-match prediction.
- **Early-game stats** (gold, XP and CS lead at 10 to 15 minutes): no gain, since the model already has Elo, gold spread and objectives.
- **Per-patch champion strength, lane matchups, champion synergy, champion classes**: each gave less than 0.001 log-loss, and some were unstable between periods.
- **Region Elo** for international events: helps cross-region games, but the analyst is trained on domestic games only.

## The model

The model is a logistic regression with 12 factors, trained on every major-league game from 2022 (10,594 games).

Each factor is the difference between the two teams, A minus B (for training, blue minus red). The steps are:

1. Standardise each factor with the training mean `m_j` and standard deviation `s_j`.
2. Fit weights `w_j` with a weak L2 penalty (`C = 99.4`).
3. For a matchup with differences `x_j`, the log-odds that A wins are `logit = b + Σ w_j (x_j − m_j) / s_j`.

To remove the side effect, the report splits the log-odds into two parts:

- **Contribution of factor j:** `c_j = w_j × x_j / s_j`.
- **Side term:** `side = b − Σ w_j × m_j / s_j`. This is the blue-side advantage in the training data.

The side-neutral probability is `p = 1 / (1 + exp(−Σ c_j))`. With side included, A's probability is `1 / (1 + exp(−(Σ c_j + side)))` on blue and `1 / (1 + exp(−(Σ c_j − side)))` on red. The report averages the two sides.

### Loss function and fitting

The model is fitted by minimising **log-loss** (binary cross-entropy), the standard loss for a classifier that outputs probabilities:

```
log-loss = −(1/N) × Σ_i [ y_i × ln(p_i) + (1 − y_i) × ln(1 − p_i) ]
```

`y_i` = 1 if blue won, 0 otherwise; `p_i` = the model's probability that blue won. Log-loss punishes confident wrong answers heavily and rewards probabilities that match the real frequencies. Accuracy only checks which side of 50% each prediction falls on, so it ignores how sure the model was.

The fit adds an L2 penalty on the weights. scikit-learn minimises `C × Σ_i log-loss_i + ½ × Σ_j w_j²`, so a larger `C` means a weaker penalty. The inputs are standardised first, so all weights are on the same scale.

`C = 99.4` was chosen by Bayesian search (Optuna, TPE sampler, 40 trials × 3 seeds) over `C` between 0.0001 and 100 on a log scale, using validation log-loss. The best value is near the top of that range, so the penalty is very weak: the data alone sets the weights.

The model is trained on all major-league games from 2022. For the track record, a model is trained only on earlier years before each walk-forward year.

### Series probability

Games are treated as independent, each with the side-neutral probability `p`:

- Best of 3: `p² × (3 − 2p)`
- Best of 5: `p³ × (10 − 15p + 6p²)`

Real series are not independent: the teams change sides and rosters change between games. Treat the series figures as approximations.

### 90% range

The app refits the model 200 times, each time on a random sample of the training games drawn with replacement (bootstrap, fixed random seed). It computes `p` for the same matchup each time. The range is the 5th to 95th percentile of those 200 values. It shows how much the estimate depends on which games were in the training data. It does not cover factors missing from the model.

### Effect of each factor (section 2)

Each factor's effect is measured one at a time:

`effect_j = p − 1 / (1 + exp(−(Σ c − c_j)))`

This is the change in probability when factor j's contribution is set to zero. Read in percentage points. The effects overlap, so they do not add up exactly to the gap from 50%. The group totals are more reliable.

### Market comparison (`--odds`)

With decimal odds `o_A` and `o_B`, the bookmaker margin is `(1/o_A + 1/o_B) − 1`. The margin-free probability is `(1/o_A) / (1/o_A + 1/o_B)`. The app compares it with the model's series probability.

## The track record (section 1 and section 7)

Walk-forward predictions show how the model has done. For each year Y from 2024, a model trained only on 2022 to Y−1 predicts every game of year Y. Each prediction is made before the game, so there is no look-ahead.

**Track record line (section 1).** The app takes all walk-forward predictions where the favourite's probability is within 5 points of this matchup's, and reports how often the favourite won. It needs at least 30 games.

**True / false positives (section 7).** Each team's view of each game is `p` = the model's probability that the team wins. "Said WIN" means `p > 0.5`.

| Label | Model said | Result |
|---|---|---|
| TP | WIN | won |
| FP | WIN | lost |
| TN | LOSS | lost |
| FN | LOSS | won |

"Right" = (TP + TN) / all games. The three pie charts are:

1. All games of team A
2. All games of team B
3. Only games between A and B, from A's side

Blue slices are correct predictions and orange slices are wrong ones.

## Head to head, lanes and recent form

- **Head to head (section 4):** every game since 2021 between the two teams in the six major leagues and the international events. The series score groups games on the same date between the same two teams. This is an approximation: a series that spans days is counted as separate series. The "last 2 years" count uses games in the last 730 days.
- **Lane head to head:** for each role, the record of the two players who started in that role when the teams met, from the same data.
- **Recent form (section 5):** the last 10 results (W/L), the current streak, the Elo change over those 10 games (rating now minus rating 10 games ago), and the days since the last game. A gap over 60 days is flagged.

## Cross-region matches

When the two teams come from different leagues, the app warns you. Ratings are earned mostly inside each league. The only direct link between regions is the handful of international games, so the probability is rough.

## Data

`05_analyst/data/` is written by `export_data.py` and is not in git. It contains:

| File | Content |
|---|---|
| `train.csv.gz` | the 10,594 training games and their 12 factors |
| `predictions.csv.gz` | walk-forward predictions, 2024 onward |
| `teams.csv` | team Elo, recent form, last game |
| `players.csv` | player Elo and game counts |
| `lineups.csv` | each team's last starting five |
| `games.csv.gz` | every major-league and international game since 2021 |
| `lanes.csv.gz` | every same-role meeting of two players since 2021 |
| `manifest.json` | data date, model settings, file list |

Source: [Oracle's Elixir](https://oracleselixir.com) match data.

## Refreshing the data

1. Put the newest Oracle's Elixir CSVs in `01_data/`.
2. Rebuild the features in `02_features/`: `build_features.py`, `roster_features.py`, `experience_features.py`.
3. From `05_analyst/`, run `python export_data.py`. It rebuilds the snapshot in `cache/` and writes `data/`.
4. Optional: `python validate_snapshot.py` compares the snapshot with the training features.

## Limits

- Pre-draft only. Draft, patch, injuries, scrims and roster news after the data date are not included. Use `--a-roster` for known changes.
- The model has no direct information on draft or champion picks, so it cannot see how a team's strategy changes between games.
- Team names in the track record use each team's current name. Older names are not counted.
- In every test in this project, the bookmaker's closing odds were a better forecast than this model.

**This is an analysis tool, not betting advice.**

## Background: the betting study

The project started as a study of whether a machine-learning model could beat bookmaker odds for professional League of Legends matches. Models trained on the same kind of features did not beat Pinnacle's closing odds in any test. The betting simulation has been removed; it is in the git history.

## Files

| Path | Purpose |
|---|---|
| `05_analyst/lol_analyst.py` | the app |
| `05_analyst/state.py` | builds the team and player snapshot |
| `05_analyst/export_data.py` | writes `05_analyst/data/` |
| `05_analyst/validate_snapshot.py` | checks the snapshot against the training features |
| `02_features/` | per-game features and roster and experience features |
| `03_model_selection/` | model selection and checks |
| `01_data/` | Oracle's Elixir CSVs (not in git) |
| `assets/` | figures and the figure style |
