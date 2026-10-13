# Calibration vs accuracy for LoL esports betting

Reproducing Walsh & Joshi (2024), *"Machine learning for sports betting: should model selection be based on accuracy or calibration?"*, on professional League of Legends (LCK, LPL, LEC, LCS, CBLOL, LCP, 2024–2026) instead of the NBA.

## Status

| Stage | Folder | State |
|---|---|---|
| 1. Raw data (Oracle's Elixir, 2024–2026) | `01_data/` | ✅ done |
| 2. Feature building (blue − red, past games only, Elo) | `02_features/` | ✅ done: 6,477 games, 17 features |
| 3. Model selection (accuracy branch vs calibration branch) + series probabilities | `03_model_selection/` | ✅ done |
| 4. Historical odds (OddsPapi: series winner + map-1 winner) | `04_odds/` | ✅ done: 250 series Jun–Oct 2026, Pinnacle odds for 243 |
| 5. Betting simulation (fixed stake, fractional Kelly) | `05_betting_simulation/` | ⬜ not started |

---

## Introduction

### Betting in one page

A bookmaker offers **decimal odds** $o$: a stake of 1 returns $o$ if the bet wins (stake + profit $o-1$), and 0 if it loses.

- **Implied probability**: $\pi = 1/o$. Odds of 2.50 mean the bookmaker prices the outcome at 40 %.
- **Margin (overround)**: for a two-way market $1/o_1 + 1/o_2 > 1$. The excess (typically 3–6 %) is the bookmaker's commission. Fair odds on a coin flip are 2.00/2.00; a bookmaker offers 1.90/1.90, so $2/1.90 = 1.053$, a 5 % margin.
- **Expected value** of a unit bet, if the true win probability is $p$:

$$\mathrm{EV} = p\,o - 1 .$$

- **Value bet**: $\mathrm{EV} > 0 \iff p > 1/o$. A bettor can only find value bets with a model whose $p$ is *closer to the truth* than the bookmaker's $\pi$.

### Why it matters

Most sports-prediction papers pick the model with the highest **accuracy** (share of winners predicted correctly). But a betting decision does not use the predicted winner; it uses the **number** $p$, compared against $1/o$. A model can be accurate yet badly over- or under-confident, and then it finds "value" that is not there. The paper argues that **calibration** (how honest the probabilities are) is what should drive model selection, and reports +34.7 % vs −35.2 % average ROI on one NBA season.

### Accuracy vs calibration

Two models, one game whose true blue-win probability is 60 %:

| Model | Says | Picks the winner? | Honest? |
|---|---|---|---|
| A | 60 % | yes | yes |
| B | 95 % | yes | no, over-confident |

Accuracy cannot tell A from B. If the odds imply 70 %, A correctly passes and B bets on a negative-EV outcome.

**Calibration.** A model is perfectly calibrated if, among all games where it says $p$, the event happens a fraction $p$ of the time:

$$P(Y = 1 \mid \hat p = p) = p \quad \forall p\in[0,1].$$

**Expected calibration error (ECE).** Split $[0,1]$ into $M$ equal bins $B_1,\dots,B_M$ (the paper uses $M = 20$). With $n$ predictions,

$$\mathrm{ECE} = \sum_{j=1}^{M} \frac{|B_j|}{n}\,\Big|\,\bar y(B_j) - \bar p(B_j)\,\Big| ,$$

where $\bar y(B_j)$ is the observed win rate and $\bar p(B_j)$ the mean predicted probability in bin $j$. For a binary outcome the paper's *classwise*-ECE equals this ECE on the positive class.

**The loophole.** Predicting the base rate (blue wins ≈ 54 %) for every game gives ECE ≈ 0 and is useless. The paper's patch: if fewer than 80 % of the bins are non-empty, ECE is set to 1. We keep this rule, and we also report **log-loss** and **Brier score**, which are *proper scoring rules*: they reward probabilities that are both honest and informative, and cannot be gamed by predicting the base rate.

$$\text{log-loss} = -\frac1n\sum_i\big[y_i\ln\hat p_i + (1-y_i)\ln(1-\hat p_i)\big],\qquad \text{Brier} = \frac1n\sum_i(\hat p_i - y_i)^2 .$$

**Kelly staking.** For a bet at odds $o$ ($b = o-1$) with win probability $p$, the stake fraction that maximises long-run log-wealth is

$$k^\* = \frac{p\,b - (1-p)}{b}.$$

Full Kelly is very aggressive when $p$ is wrong, so the paper bets $k^\*/8$ ("eighth-Kelly"). Kelly makes calibration matter *more*: an over-confident $p$ inflates both the number of bets and their size.

### The model we ended up with: logistic regression

Both branches of the pipeline selected **logistic regression (LR)**; RF, SVM and MLP were not better on the test months.

**Model.** For a game with feature vector $x$ (blue − red differences, standardised), LR says the **log-odds** of a blue win are linear in the features:

$$\ln\frac{p}{1-p} = \beta_0 + \beta^\top x \quad\Longleftrightarrow\quad p = \sigma(\beta_0 + \beta^\top x) = \frac{1}{1 + e^{-(\beta_0 + \beta^\top x)}} .$$

- $\beta_0$ is the side effect: when the two teams are equal ($x = 0$), $p = \sigma(\beta_0)$.
- $\beta_k$: one standard deviation more of feature $k$ multiplies the odds of a blue win by $e^{\beta_k}$.

**Fitting.** $\beta$ minimises the log-loss above plus an L2 penalty,

$$\min_{\beta_0,\beta}\; \sum_i \text{log-loss}_i + \frac{1}{2C}\,\lVert\beta\rVert^2 ,$$

where $C$ is the one hyperparameter the tuner chose (smaller $C$ = stronger shrinkage toward $p = \sigma(\beta_0)$). Because LR is trained *directly on log-loss*, a proper scoring rule, it tends to come out well calibrated without extra steps; this is a classic reason LR is a strong baseline for probability forecasting.

**What the accuracy-driven LR learned** (trained on 2024 – May 2026, $C = 0.196$):

| Feature (blue − red) | 1 SD = | $\beta$ (per SD) | Odds ratio $e^{\beta}$ |
|---|---|---|---|
| intercept (equal teams) | – | +0.143 | $p = 0.536$, the blue-side edge |
| Elo rating | 129 Elo points | **+0.714** | 2.04 |
| dragon edge (last 10 games) | 1.4 dragons | +0.119 | 1.13 |
| ward-clear-per-minute edge | 0.27 | +0.103 | 1.11 |
| first-blood rate | 0.24 | +0.042 | 1.04 |
| Baron edge | 0.67 | −0.042 | 0.96 |
| first-dragon rate | 0.25 | −0.062 | 0.94 |

Elo dominates: a blue team rated 129 points higher has twice the odds of winning. The small negative coefficients belong to features that overlap with Elo and dragon edge (they correct for double counting rather than meaning "first dragons are bad").

**From games to series.** Bookmakers price the series. With per-game probability $p$ for team A (averaged over blue and red side) and games treated as independent,

$$P_{\text{Bo1}} = p,\qquad P_{\text{Bo3}} = p^2(3-2p),\qquad P_{\text{Bo5}} = p^3(10-15p+6p^2).$$

---

## Literature review

Full notes: [`Literatures/literature_review.md`](Literatures/literature_review.md).

| Paper | What it did | What it leaves open |
|---|---|---|
| **Walsh & Joshi 2024** (anchor) | NBA; two pipelines that select features, hyperparameters and model by accuracy vs by classwise-ECE; eighth-Kelly betting on 2018/19; calibration-driven ROI +34.7 % vs −35.2 % | One season, no significance test; ECE loophole patched by an arbitrary 80 %-bins rule; no proper scoring rule; only the NBA |
| Hubáček, Šourek & Železný 2019 | NBA; loss that penalises correlation with bookmaker odds beats optimising accuracy | Not framed as calibration; same single sport |
| Guo et al. 2017 | Modern neural nets are over-confident; introduces temperature scaling, popularises ECE | Not about decisions or betting |
| Kull et al. 2019 | Classwise-ECE and Dirichlet calibration; notes ECE can be gamed by predicting the class distribution | Not about betting |
| Kelly 1956 / Hsieh & Barmish 2015 | Optimal log-growth stake; full Kelly leads to near-certain ruin with model error, fractional Kelly is safer | Assumes $p$ is known |

### The gap

- Walsh & Joshi: evidence from **one sport, one season**, no uncertainty estimate.
- ECE as a *selection* criterion can prefer a **less informative** model (it rewards hugging the base rate).
- Esports markets are younger and possibly less efficient, but untested with this method.

**Overall gap:** it is not known whether calibration-based model selection beats accuracy-based selection outside the NBA, and whether ECE is the right calibration criterion compared to proper scoring rules.

| Gap | Statement |
|---|---|
| G1 | The accuracy-vs-calibration result has only been shown on one NBA season. |
| G2 | ECE-based selection can favour uninformative models; proper scoring rules (log-loss) are untested as the selection criterion in this setting. |
| G3 | Game-level models must be mapped to series-level betting markets in esports. |

### Research questions

| | Question | Gap | Hypothesis |
|---|---|---|---|
| RQ1 | Does calibration-driven model selection give a higher betting ROI than accuracy-driven selection on LoL (Jun–Oct 2026)? | G1 | **H1.** Yes, as in the NBA paper. |
| RQ2 | Does selecting by log-loss beat both? | G2 | **H2.** Log-loss-driven selection is at least as profitable as ECE-driven, because ECE can reward under-confident models. |
| RQ3 | Does betting on series (via $P_{\text{Bo}n}$) or on map 1 directly give better-calibrated decisions? | G3 | **H3.** Map-1 bets are better matched to a game-level model. |

**Novelty**

| | Contribution | Gap |
|---|---|---|
| N1 | First test of the calibration-vs-accuracy selection question on esports, 6 leagues | G1 |
| N2 | Side-by-side ECE vs log-loss as the selection criterion | G2 |
| N3 | Game → series probability mapping tested against real series odds | G3 |

*Limits on the novelty claim:* this is a reproduction study on one betting period (≈ 320 series); results are a case study, not a general proof.
*If the hypotheses fail:* a negative result on H1 is still informative: it would show the NBA result does not transfer, and the reliability plots show *why*.

### Scope

Exactly what G1–G3 require:

| In scope | Needed for |
|---|---|
| Team-level features from Oracle's Elixir, 6 leagues, 2024–2026 | G1 |
| Accuracy / ECE / (log-loss) selection branches | G1, G2 |
| Pinnacle closing odds, series winner + map-1 winner | G1, G3 |
| Fixed-stake and eighth-Kelly simulation | G1 |

Not in the current scope: player-level features, draft/champion features, live (in-game) betting, international events.
Deferred: retraining every split on the two previous splits; bootstrap confidence intervals on ROI.

---

## Methods

### Settings

| | |
|---|---|
| Data | Oracle's Elixir match CSVs 2024–2026, team rows only; LCS, LEC, LCK, LPL, CBLOL, LCP (2025 LTA N → LCS, LTA S → CBLOL) |
| Label | `blue_win` (blue side wins ≈ 54 %) |
| Split (time order, never shuffled) | train 2024 · validation 2025 · test Jan–May 2026 · betting Jun–Oct 2026 |
| Odds | OddsPapi historical odds, Pinnacle closing price; market 181 (series winner), 1847 (map-1 winner) |

### Stage 2: features (`02_features/build_features.py`)

1. Per game, each team's **edge** = team − opponent (kills − deaths, dragons, grubs, Barons, towers, inhibitors, damage/min, vision/min, gold/min, ward clears/min, gold-spent diff, first blood/dragon/tower, game length, win).
2. **Rolling mean of the previous 10 games** (`shift(1)`: the current game never sees itself), at least 5 past games required.
3. **Elo** (K = 24, 25 % reset toward 1500 each new year).
4. Game row = blue − red. @15-min stats, CS/min, first Herald and first Baron are excluded because LPL lacks them.

### Stage 3: model selection (`03_model_selection/pipeline.py`)

| Step | Accuracy branch | Calibration branch |
|---|---|---|
| A. Covariate shift: KS test train vs validation | drop if $p < 0.01$ **and** $D > 0.15$ (shared) | same |
| B. Redundancy: Spearman $\lvert\rho\rvert > 0.7$ → keep the one most correlated with the label | shared | shared |
| C. Sequential forward selection with LR, run to completion | max accuracy | min ECE (20 bins, ≥ 80 % filled) |
| D. Bayesian optimisation (TPE, Optuna), 40 trials × 3 seeds, for LR / RF / SVM / MLP | max accuracy | min ECE |
| E. Refit on train + val, choose best on test | highest accuracy | lowest ECE |

Deviation from the paper: with ~2,000 games per year the KS test rejects almost every feature at $p < 0.01$ (it dropped Elo and accuracy fell to 59 %), so a minimum effect size $D > 0.15$ was added.

`series_probs.py` converts the chosen models' game probabilities into series probabilities (formulas above). Best-of is inferred from the score.

**Checks:** reliability diagram (`assets/reliability_bet.png`); bet-period metrics computed once, after all choices were frozen.

### Stage 4–5: odds and simulation (planned)

Match OddsPapi fixtures to Oracle series by date and team names; bet whenever $p > 1/o$ (value bet); stakes: fixed 100 and eighth-Kelly from a 10,000 bankroll; report ROI, number of bets, and bootstrap intervals.

---

## Results so far

**Selected features**

- Accuracy branch: Elo, dragon edge, first-dragon rate, ward-clear edge, Baron edge, first-blood rate
- Calibration branch: damage/min edge, first-dragon rate, game length (Elo was *not* chosen)

**Test set (Jan–May 2026, 1,460 games), all candidates**

| Branch | Model | Accuracy | ECE | Log-loss | Brier |
|---|---|---|---|---|---|
| accuracy | **LR** | **0.628** | 0.056 | **0.651** | **0.229** |
| accuracy | RF | 0.622 | 0.059 | 0.658 | 0.232 |
| accuracy | SVM | 0.625 | 0.052 | 0.653 | 0.230 |
| accuracy | MLP | 0.626 | 0.068 | 0.656 | 0.230 |
| calibration | **LR** | 0.606 | 0.037 | 0.665 | 0.235 |
| calibration | RF | 0.612 | 0.041 | 0.669 | 0.237 |
| calibration | SVM | 0.607 | 0.038 | 0.664 | 0.235 |
| calibration | MLP | 0.610 | 0.029 (only 75 % bins → rejected) | 0.663 | 0.235 |

**Betting period (Jun–Oct 2026), final models**

| | Accuracy-driven LR | Calibration-driven LR |
|---|---|---|
| Game accuracy (913 games) | **63.8 %** | 60.0 % |
| ECE | 3.85 % | **3.43 %** |
| Log-loss | **0.637** | 0.660 |
| Series accuracy (323 series) | **68.4 %** | 62.5 % |

![Reliability diagram](assets/reliability_bet.png)

Both models track the diagonal. The calibration-driven model has a narrower spread of predictions (the histogram), which is how it gets its lower ECE; the accuracy-driven model is slightly over-confident above 0.85. Baseline: always predicting blue gives ≈ 54 %.

---

## File structure

```
Project-0/
├── README.md, NOTES.md, .gitignore
├── assets/                    figures + shared figure style (figstyle.mplstyle)
├── Literatures/               literature_review.md + paper PDFs (PDFs not tracked)
├── 01_data/                   Oracle's Elixir CSVs 2024–2026 (not tracked)
├── 02_features/
│   ├── build_features.py      raw CSVs -> one row per game, blue - red features
│   └── features.csv
├── 03_model_selection/
│   ├── pipeline.py            KS -> correlation -> SFS -> TPE tuning -> selection, both branches
│   ├── series_probs.py        game -> series probabilities for the betting period
│   ├── plot_reliability.py    -> assets/reliability_bet.png
│   └── results/               logs, scores, models, predictions
├── 04_odds/                   fetch_odds.py, odds_raw/, odds_series.csv   (moved here after download)
└── 05_betting_simulation/     (next)
```

## How to reproduce

In WSL, inside a venv:

```bash
pip install pandas numpy scipy scikit-learn optuna joblib matplotlib
cd /mnt/c/Users/ACER/OneDrive/Desktop/Project-0
cd 02_features && python build_features.py && cd ..          # ~5 s
cd 03_model_selection && python pipeline.py                 # ~5 min (tuning cached in results/)
python series_probs.py && python plot_reliability.py && cd ..
cd ../04_odds && python fetch_odds.py --market 181,1847     # ~35 min, needs oddspapi_key.txt here
```

## Not tracked

`01_data/*.csv`, `Literatures/*.pdf`, `oddspapi_key.txt`, `odds_raw/`, `*.joblib`, `__pycache__/`, `_to_delete/` (see `.gitignore`).

## Next steps

1. Match odds fixtures to Oracle series (date + fuzzy team names).
2. `05_betting_simulation/simulate.py`: value-bet strategy, fixed and eighth-Kelly stakes, both markets (181, 1847), ROI + bootstrap intervals.
3. Add a third, **log-loss-driven** branch (RQ2).
4. Optional: retrain every split on the two previous splits (original plan).
