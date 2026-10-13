"""
pipeline.py  -  Walsh & Joshi (2024) model-selection pipeline, applied to LoL
(three branches: accuracy, calibration = ECE, logloss)

Usage (WSL, inside your venv):
    pip install pandas numpy scipy scikit-learn optuna joblib
    python pipeline.py                 # full run (~5 min)
    python pipeline.py --trials 10     # quick test run
    (tuning is cached in results/tuning_cache.json: rerun to resume; delete it to retune)

Input : ../02_features/features.csv  (from build_features.py); run from 03_model_selection/
Output: results/
    selection_log.txt     every decision the pipeline made
    test_scores.csv       accuracy / ECE / log-loss / Brier of every candidate on the test set
    predictions_bet.csv   P(blue wins) for every game in the betting period, from both final models
    model_accuracy.joblib, model_calibration.joblib

Time split (no shuffling, ever):
    train : 2022-2024                  (2021 = warm-up only: Elo, rolling stats, player histories)
    val   : 2025                       (feature selection + hyperparameter tuning)
    test  : 2026-01-01 .. 2026-05-31   (pick the best model of each branch)
    bet   : 2026-06-01 ..              (betting simulation, untouched until the end)
"""
import argparse
import json
import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_BINS = 20          # ECE bins, as in the paper
MIN_FILLED = 0.80    # paper's constraint: >= 80% of bins non-empty, else ECE = 1
KS_ALPHA = 0.01      # covariate-shift test level
KS_MIN_D = 0.15      # ...and minimum shift size (KS statistic) to count as real shift
CORR_MAX = 0.70      # Spearman threshold for "redundant"
SEEDS = [0, 1, 2]    # HPO repeated over seeds, best kept

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(msg)


# ------------------------------------------------------------------ metrics
def ece(y, p, n_bins=N_BINS, constrained=True):
    """Classwise-ECE for a binary problem = ECE on the positive class (blue win)."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    err, filled = 0.0, 0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            filled += 1
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    if constrained and filled / n_bins < MIN_FILLED:
        return 1.0
    return err


def filled_frac(p, n_bins=N_BINS):
    bins = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)
    return len(np.unique(bins)) / n_bins


def score(branch, y, p):
    """Lower is better for both branches."""
    if branch == "accuracy":
        return -accuracy_score(y, p >= 0.5)
    if branch == "logloss":                       # proper scoring rule: honest AND informative
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    return ece(y, p)


# ------------------------------------------------------------------ models
def make_model(name, hp, seed=0):
    if name == "LR":
        m = LogisticRegression(C=hp.get("C", 1.0), max_iter=2000)
    elif name == "RF":
        m = RandomForestClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"],
            random_state=seed, n_jobs=-1)
    elif name == "SVM":
        m = SVC(C=hp["C"], kernel=hp["kernel"], gamma=hp["gamma"],
                probability=True, random_state=seed,
                max_iter=1_000_000)   # cap: poly kernels with large C can run for hours on 6k games
    elif name == "MLP":
        m = MLPClassifier(hidden_layer_sizes=(hp["width"],) * hp["layers"],
                          alpha=hp["alpha"], learning_rate_init=hp["lr"],
                          max_iter=500, early_stopping=True, random_state=seed)
    return make_pipeline(StandardScaler(), m)   # scaler is fitted on training data only


def suggest(trial, name):
    if name == "LR":
        return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
    if name == "RF":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0)}
    if name == "SVM":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True)}
    if name == "MLP":
        return {"width": trial.suggest_int("width", 4, 64, log=True),
                "layers": trial.suggest_int("layers", 1, 3),
                "alpha": trial.suggest_float("alpha", 1e-5, 10, log=True),
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True)}


def fit_predict(name, hp, Xtr, ytr, Xev, seed=0):
    m = make_model(name, hp, seed).fit(Xtr, ytr)
    return m.predict_proba(Xev)[:, 1], m


# ------------------------------------------------------------------ pipeline steps
def ks_filter(tr, va, feats):
    keep = []
    for f in feats:
        res = ks_2samp(tr[f], va[f])
        # with ~2000 games per year almost any tiny shift gives p < 0.01, so a feature
        # is dropped only if the shift is also LARGE (KS statistic D > KS_MIN_D)
        if res.pvalue < KS_ALPHA and res.statistic > KS_MIN_D:
            log(f"  drop {f:20s} KS p = {res.pvalue:.1e}, D = {res.statistic:.3f}")
        else:
            keep.append(f)
    return keep


def corr_filter(tr, feats):
    """Among highly correlated features keep the one most correlated with the label."""
    target = {f: abs(spearmanr(tr[f], tr["blue_win"])[0]) for f in feats}
    keep = []
    for f in sorted(feats, key=target.get, reverse=True):
        clash = [k for k in keep if abs(spearmanr(tr[f], tr[k])[0]) > CORR_MAX]
        if clash:
            log(f"  drop {f:20s} redundant with {clash[0]} (|rho| > {CORR_MAX})")
        else:
            keep.append(f)
    return keep


def forward_selection(branch, tr, va, feats):
    """SFS with LR, run to completion; best subset on the validation set wins."""
    chosen, remaining, path = [], list(feats), []
    while remaining:
        best = None
        for f in remaining:
            p, _ = fit_predict("LR", {}, tr[chosen + [f]], tr["blue_win"], va[chosen + [f]])
            s = score(branch, va["blue_win"], p)
            if best is None or s < best[1]:
                best = (f, s)
        chosen.append(best[0])
        remaining.remove(best[0])
        path.append((list(chosen), best[1]))
    subset, s = min(path, key=lambda t: t[1])
    return subset, s


def tune(branch, name, tr, va, feats, n_trials):
    best = None
    for seed in SEEDS:
        def objective(trial):
            hp = suggest(trial, name)
            p, _ = fit_predict(name, hp, tr[feats], tr["blue_win"], va[feats], seed)
            return score(branch, va["blue_win"], p)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        if best is None or study.best_value < best[1]:
            best = (study.best_params, study.best_value, seed)
    return best


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../02_features/features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--trials", type=int, default=40, help="Optuna trials per model per seed")
    ap.add_argument("--models", default="LR,RF,SVM,MLP", help="comma list, e.g. LR for a fast run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.features, parse_dates=["date"]).sort_values("date")
    feats = [c for c in df.columns if c.startswith("f_")]
    df[feats] = df[feats].fillna(0)

    tr = df[df["date"].dt.year.between(2022, 2024)]   # 2021 = warm-up only (Elo, rolling stats, histories)
    va = df[df["date"].dt.year == 2025]
    te = df[(df["date"] >= "2026-01-01") & (df["date"] < "2026-06-01")]
    bt = df[df["date"] >= "2026-06-01"]
    log(f"train {len(tr)} | val {len(va)} | test {len(te)} | bet {len(bt)} games")
    log(f"blue win rate: train {tr.blue_win.mean():.3f}, test {te.blue_win.mean():.3f}\n")

    log("STEP A  covariate-shift check (KS test, train vs val)")
    feats = ks_filter(tr, va, feats)
    log(f"  -> {len(feats)} features left\n")

    log("STEP B  correlation filter (on train)  [shared by both branches]")
    feats = corr_filter(tr, feats)
    log(f"  -> subset A ({len(feats)}): {feats}\n")

    rows, final = [], {}
    cache_path = os.path.join(args.out, "tuning_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for branch in ["accuracy", "calibration", "logloss"]:
        log(f"===== {branch.upper()} BRANCH =====")
        subset, s = forward_selection(branch, tr, va, feats)
        log(f"STEP C  forward selection -> {subset}  (val score {s:.4f})")

        best_model = None
        for name in args.models.split(","):
            # tuning results are cached, so an interrupted run resumes where it stopped
            key_c = f"{branch}|{name}|{','.join(subset)}|{args.trials}"
            if key_c in cache:
                hp, vs, seed = cache[key_c]
            else:
                hp, vs, seed = tune(branch, name, tr, va, subset, args.trials)
                cache[key_c] = [hp, vs, seed]
                with open(cache_path, "w") as fh:
                    json.dump(cache, fh, indent=1)
            # refit on train+val, evaluate on test
            trva = pd.concat([tr, va])
            p, _ = fit_predict(name, hp, trva[subset], trva["blue_win"], te[subset], seed)
            y = te["blue_win"]
            row = {"branch": branch, "model": name, "val_score": vs,
                   "test_accuracy": accuracy_score(y, p >= 0.5),
                   "test_ECE": ece(y, p, constrained=False),
                   "test_ECE_constrained": ece(y, p),
                   "bins_filled": filled_frac(p),
                   "test_logloss": log_loss(y, p), "test_brier": brier_score_loss(y, p),
                   "hyperparams": json.dumps(hp), "seed": seed}
            rows.append(row)
            log(f"STEP D  {name:3s} tuned {hp}")
            log(f"        test: acc {row['test_accuracy']:.4f}  ECE {row['test_ECE']:.4f}  "
                f"bins filled {row['bins_filled']:.0%}  logloss {row['test_logloss']:.4f}")
            key = {"accuracy": -row["test_accuracy"], "calibration": row["test_ECE_constrained"],
                   "logloss": row["test_logloss"]}[branch]
            if best_model is None or key < best_model[0]:
                best_model = (key, name, hp, seed)
        _, name, hp, seed = best_model
        log(f"STEP E  selected {branch}-driven model: {name}\n")
        final[branch] = {"model": name, "hyperparams": hp, "seed": seed, "features": subset}

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "test_scores.csv"), index=False)

    log("FINAL  refit on train+val+test, predict the betting period")
    full = pd.concat([tr, va, te])
    out = bt[["gameid", "date", "league", "game", "blue_team", "red_team", "blue_win"]].copy()
    for branch, f in final.items():
        p, m = fit_predict(f["model"], f["hyperparams"], full[f["features"]], full["blue_win"],
                           bt[f["features"]], f["seed"])
        out[f"p_blue_{branch}"] = p
        joblib.dump(m, os.path.join(args.out, f"model_{branch}.joblib"))
        log(f"  {branch:11s} {f['model']}: bet-period acc {accuracy_score(bt.blue_win, p >= .5):.4f}"
            f"  ECE {ece(bt.blue_win, p, constrained=False):.4f}  logloss {log_loss(bt.blue_win, p):.4f}")
    out.to_csv(os.path.join(args.out, "predictions_bet.csv"), index=False)
    with open(os.path.join(args.out, "final_models.json"), "w") as fh:
        json.dump(final, fh, indent=2)
    with open(os.path.join(args.out, "selection_log.txt"), "w") as fh:
        fh.write("\n".join(LOG))
    log(f"\nSaved everything in {args.out}/")


if __name__ == "__main__":
    main()
