# [Eric] Figures for the November slides and report (custom palette, no blue).
# Usage: python make_figures.py <neural_nets/eric_v1 dir> <output figs dir> <folder with IBMPlexSans-*.otf>
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import average_precision_score as AP, precision_recall_curve
from scipy.stats import rankdata

RES = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FONT_DIR = Path(sys.argv[3])
for f in FONT_DIR.glob("IBMPlexSans-*.otf"):
    fm.fontManager.addfont(str(f))

INK, MUTED, GRID, SURF = "#17211C", "#56605A", "#D9DDD5", "#FBFBF8"
EMERALD, AMBER, GRAY = "#16855A", "#A97A06", "#8C938E"
plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 13, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "savefig.dpi": 220})

def pred(name, seed=0):
    return pd.read_parquet(RES / f"{name}__seed{seed}_val_predictions.parquet")

base = pred("logistic_regression"); y = base.y.to_numpy()
names = ["logistic_regression", "net_0_hidden", "mlp_tiny", "mlp_small", "mlp_medium",
         "mlp_small_weighted", "mlp_small_emb", "lightgbm_all"]
labels = {"logistic_regression": "Logistic regression (team settings)", "net_0_hidden": "Network, 0 hidden layers",
          "mlp_tiny": "MLP tiny (64)", "mlp_small": "MLP small (256-128)", "mlp_medium": "MLP medium (512-256-128)",
          "mlp_small_weighted": "MLP small + fraud weight", "mlp_small_emb": "MLP small + embeddings",
          "lightgbm_all": "LightGBM (team settings)"}
score = {n: pred(n).score.to_numpy() for n in names}
ap = {n: AP(y, s) for n, s in score.items()}

# 1. AP bar chart (seed 0)
fig, ax = plt.subplots(figsize=(10, 5.2))
order = names[::-1]
cols = [AMBER if n == "lightgbm_all" else GRAY if n in ("logistic_regression", "net_0_hidden") else EMERALD for n in order]
bars = ax.barh([labels[n] for n in order], [ap[n] for n in order], color=cols, height=0.62)
for b, n in zip(bars, order):
    ax.text(b.get_width() + 0.008, b.get_y() + b.get_height() / 2, f"{ap[n]:.3f}", va="center", fontsize=13, color=INK)
ax.axvline(y.mean(), color=MUTED, ls=":", lw=1.5)
ax.text(y.mean() + 0.006, len(order) - 0.45, f"random = {y.mean():.3f}", color=MUTED, fontsize=11, va="bottom")
ax.set_ylim(-0.6, len(order) + 0.1)
ax.set_xlim(0, 0.72); ax.set_xlabel("Validation average precision (higher is better)")
ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0, labelsize=13)
fig.tight_layout(); fig.savefig(OUT / "fig_ap_bars.png"); plt.close(fig)

# 2. Seed stability (dot plot)
seed_rows = []
for f in sorted(RES.glob("*__seed*_val_predictions.parquet")):
    n, sd = f.name.split("__seed")
    seed_rows.append({"model": n, "seed": int(sd.split("_")[0]), "ap": AP(y, pd.read_parquet(f).score)})
seeds = pd.DataFrame(seed_rows)
multi = seeds.groupby("model").filter(lambda d: len(d) >= 3)
fig, ax = plt.subplots(figsize=(8.5, 2.9))
for i, (n, d) in enumerate(multi.groupby("model", sort=False)):
    c = EMERALD if n == "mlp_small_emb" else GRAY
    ax.scatter(d.ap, [i] * len(d), s=110, color=c, edgecolor=SURF, linewidth=2, zorder=3)
    ax.plot([d.ap.min(), d.ap.max()], [i, i], color=c, lw=2, zorder=2)
    ax.text(d.ap.max() + 0.004, i, f"mean {d.ap.mean():.3f} ± {d.ap.std():.3f}", va="center", fontsize=12)
ax.set_yticks(range(multi.model.nunique()), [labels[n] for n in multi.model.unique()])
ax.set_ylim(-0.7, multi.model.nunique() - 0.3)
ax.set_xlim(multi.ap.min() - 0.02, multi.ap.max() + 0.06)
ax.set_xlabel("Validation AP, one dot per random seed"); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(OUT / "fig_seeds.png"); plt.close(fig)
seeds.to_csv(OUT / "seed_table.csv", index=False)

# 3. Learning curve of the embedding model (over-fitting story)
h = pd.DataFrame(json.loads((RES / "mlp_small_emb__seed0_history.json").read_text())["history"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(h.epoch, h.train_loss, color=INK, lw=2, marker="o", ms=5)
axes[0].set(title="Training loss keeps falling", xlabel="epoch", ylabel="loss on training months")
best = h.loc[h.val_ap.idxmax()]
axes[1].plot(h.epoch, h.val_ap, color=EMERALD, lw=2, marker="o", ms=5)
axes[1].scatter([best.epoch], [best.val_ap], s=160, facecolor="none", edgecolor=AMBER, lw=2.5, zorder=4)
axes[1].annotate(f"kept: epoch {int(best.epoch)}", (best.epoch, best.val_ap), xytext=(10, -28),
                 textcoords="offset points", color=INK, fontsize=12)
axes[1].set(title="Validation AP peaks, then stalls", xlabel="epoch", ylabel="AP on later months")
for a in axes:
    a.title.set_fontsize(14)
fig.tight_layout(); fig.savefig(OUT / "fig_learning_curve.png"); plt.close(fig)

# 4. PR curves: logistic, best net, LightGBM
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for n, c, ls in [("logistic_regression", GRAY, "-"), ("mlp_small_emb", EMERALD, "-"), ("lightgbm_all", AMBER, "-")]:
    p, r, _ = precision_recall_curve(y, score[n])
    ax.plot(r, p, color=c, lw=2.2, ls=ls, label=f"{labels[n]}  (AP {ap[n]:.3f})")
ax.axhline(y.mean(), color=MUTED, ls=":", lw=1.3)
ax.set(xlabel="Recall: share of fraud caught", ylabel="Precision: share of alerts that are fraud", xlim=(0, 1), ylim=(0, 1.02))
ax.legend(frameon=False, fontsize=11, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig_pr_curves.png"); plt.close(fig)

# 5. Calibration
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(6.2, 5.2))
for n, c in [("logistic_regression", GRAY), ("mlp_small_weighted", AMBER), ("mlp_small_emb", EMERALD)]:
    fp, mp = calibration_curve(y, score[n], n_bins=15, strategy="quantile")
    ax.plot(mp, fp, marker="o", ms=5, lw=2, color=c, label=labels[n])
ax.plot([1e-4, 1], [1e-4, 1], ls=":", color=MUTED, label="perfect calibration")
ax.set(xscale="log", yscale="log", xlabel="average predicted score", ylabel="actual fraud rate", xlim=(1e-3, 1), ylim=(1e-3, 1))
ax.legend(frameon=False, fontsize=10, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_calibration.png"); plt.close(fig)

json.dump({"ap": ap, "fraud_rate": float(y.mean())}, open(OUT / "fig_numbers.json", "w"), indent=1)
print({k: round(v, 4) for k, v in ap.items()})
print(seeds.groupby("model").ap.agg(["mean", "std", "count"]).round(4))
