"""
plot_reliability.py  -  reliability diagram of both final models on the betting period
Run from 03_model_selection/ after pipeline.py.  Writes ../assets/reliability_bet.png
A perfectly calibrated model lies on the diagonal: among games given p = 0.7, blue wins 70%.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("../assets/figstyle.mplstyle")
df = pd.read_csv("results/predictions_bet.csv")
N_BINS = 10
edges = np.linspace(0, 1, N_BINS + 1)

fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
for branch, color in [("accuracy", "#1f77b4"), ("calibration", "#d62728"), ("logloss", "#2ca02c")]:
    p, y = df[f"p_blue_{branch}"].values, df["blue_win"].values
    idx = np.clip(np.digitize(p, edges) - 1, 0, N_BINS - 1)
    xs, ys = [], []
    for b in range(N_BINS):
        m = idx == b
        if m.sum() >= 10:                      # skip near-empty bins
            xs.append(p[m].mean()); ys.append(y[m].mean())
    ax.plot(xs, ys, marker="o", c=color, label=f"{branch}-driven LR")
    axh.hist(p, bins=edges, histtype="step", color=color, lw=1.5)
ax.set_ylabel("observed blue win rate")
ax.set_title(f"Reliability, betting period (Jun–Oct 2026, {len(df)} games)")
ax.legend(loc="upper left")
axh.set_xlabel("predicted P(blue wins)")
axh.set_ylabel("games")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig("../assets/reliability_bet.png")
print("Saved ../assets/reliability_bet.png")
