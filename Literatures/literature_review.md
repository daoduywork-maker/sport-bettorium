# Literature review

PDF naming: `category-firstauthoryear.pdf` (PDFs are kept locally, not tracked).

| File | Reference |
|---|---|
| `Calibration_betting-Walsh2024.pdf` | C. Walsh, A. Joshi (2024). Machine learning for sports betting: should model selection be based on accuracy or calibration? *Machine Learning with Applications* 16, 100539. doi:10.1016/j.mlwa.2024.100539 |
| `Bookmaker_decorrelation-Hubacek2019.pdf` | O. Hubáček, G. Šourek, F. Železný (2019). Exploiting sports-betting market using machine learning. *Int. J. Forecasting* 35(2), 783–796 |
| `Calibration_nets-Guo2017.pdf` | C. Guo, G. Pleiss, Y. Sun, K. Weinberger (2017). On calibration of modern neural networks. *ICML* |
| `Classwise_ECE-Kull2019.pdf` | M. Kull et al. (2019). Beyond temperature scaling: obtaining well-calibrated multi-class probabilities with Dirichlet calibration. *NeurIPS* |
| `Kelly_limits-Hsieh2015.pdf` | C.-H. Hsieh, B. R. Barmish (2015). On Kelly betting: some limitations. *Allerton* |

## Walsh & Joshi 2024 (anchor)

**Question.** For sports betting, should the predictive model be chosen by accuracy or by calibration?

**Data.** NBA 2014/15–2018/19 from basketball-reference; Westgate closing moneyline odds for 2018/19.
Split: train 14/15–15/16, validation 16/17, test 17/18, betting 18/19.

**Features.** Season-to-date average of (team stat − opponent stat), then home − away; previous-season win %; first 10 games per team excluded; KS test (1 %) drops shifted features; standardised.

**Pipeline.** Spearman > 0.7 filter → SFS with LR (accuracy or classwise-ECE, 20 bins, ≥ 80 % bins filled) → BO-TPE hyperparameters for LR, RF, SVM, MLP → pick best on test.

**Result.** Both branches chose SVM. Calibration-driven: ROI +32.5 % (fixed $100) and +36.9 % (eighth-Kelly). Accuracy-driven: +5.6 % and −75.9 %.

**Weaknesses noted while reading**
1. One season, no confidence intervals; "average ROI" is over two simulations.
2. ECE gap is small (4.46 % vs 5.03 %) for a huge ROI gap; most of the gap comes from Kelly compounding.
3. Bets on ~88 % of games: hard to believe the market is wrong that often with a 4–5 % margin and a less accurate model (≈ 64 % vs market ≈ 69 %).
4. Table 2's "Kelly" formula, $K = (P\cdot\text{Bankroll} - (1-P))/\text{Bankroll} \approx P$, never uses the odds, so it is not the Kelly criterion.
5. 80 %-bins rule is arbitrary.
6. Test season used for model choice; only 2018/19 is truly out-of-sample.

## Hubáček et al. 2019
Penalising correlation with the bookmaker's implied probabilities produces more profitable models than optimising accuracy; "a highly accurate model is useless if it coincides with the bookmaker's".

## Guo et al. 2017
Modern deep nets are over-confident; ECE and reliability diagrams; temperature scaling as a one-parameter fix.

## Kull et al. 2019
Defines classwise-ECE; points out that predicting the class distribution gives near-perfect calibration, the loophole that the 80 %-bins rule patches.

## Hsieh & Barmish 2015
Full Kelly with an estimated $p$ leads to near-certain ruin in practice; fractional Kelly trades growth for safety.
