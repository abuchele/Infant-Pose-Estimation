"""Plot per-epoch training loss curves from the Explorer job logs (logs/<name>-<jobid>.out).

Two quantities per run:
  pose loss    heatmap MSE (the pose network's supervised loss, without the adversarial term)
  D loss       domain-classifier BCE on detached features (Step I)

How the per-epoch value is obtained (the logs print one line every 20 batches):
  - function_stable.py runs (stable, paper): `Loss_mse` running average over the epoch, last printed
    line (covers batches 0..80 of 100, or 0..20 of 32). That meter resets every epoch.
  - function.py runs (retrain, backbones, lambda sweep, seed study): the printed `Loss_p` average is
    cumulative over the whole run, so it is NOT used. Instead the per-batch values on the printed lines
    are averaged per epoch, with MSE = Loss_p + LAMBDA * Loss_d_2 (Loss_p = MSE - LAMBDA * BCE).
  - D loss: mean of the per-batch `Loss_d` values on the printed lines (its average is also cumulative).
  D pre-training lines (`Accuracy_d:` without `Loss_p`) are skipped.

Usage: python tools/plot_train_loss.py --logs outputs/train_curves/logs --out outputs/train_curves
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode), assigned in fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LINE = re.compile(r"^Epoch: \[(\d+)\]\[(\d+)/(\d+)\]")


def parse(path, lam):
    ep_vals = defaultdict(lambda: {"mse": [], "mse_avg": None, "d": []})
    for ln in open(path, errors="replace"):
        m = LINE.match(ln)
        if not m or "Loss_p" not in ln:
            continue
        e = int(m.group(1))
        lp = re.search(r"Loss_p ([-\d.e]+) \(", ln)
        ld2 = re.search(r"Loss_d_2 ([-\d.e]+)", ln)
        ld = re.search(r"Loss_d ([-\d.e]+) \(", ln)
        mse = re.search(r"Loss_mse ([-\d.e]+) \(([-\d.e]+)\)", ln)
        rec = ep_vals[e]
        if mse:
            rec["mse"].append(float(mse.group(1)))
            rec["mse_avg"] = float(mse.group(2))
        elif lp and ld2:
            rec["mse"].append(float(lp.group(1)) + lam * float(ld2.group(1)))
        if ld:
            rec["d"].append(float(ld.group(1)))
    eps = sorted(ep_vals)
    mse = np.array([ep_vals[e]["mse_avg"] if ep_vals[e]["mse_avg"] is not None else np.mean(ep_vals[e]["mse"])
                    for e in eps])
    d = np.array([np.mean(ep_vals[e]["d"]) if ep_vals[e]["d"] else np.nan for e in eps])
    return np.array(eps), mse, d


def find(logdir, jobid):
    hits = glob.glob(os.path.join(logdir, f"*-{jobid}.out"))
    if not hits:
        raise SystemExit(f"no log for job {jobid} in {logdir}")
    return hits[0]


def style(ax, ylab, steps):
    for s in steps:
        ax.axvline(s, color=GRID, lw=1, ls="--", zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_ylabel(ylab, color=INK2, fontsize=9)


def figure(name, title, series, logdir, outdir, steps, with_d=True, note=""):
    """series: list of (label, [jobids], lambda). Several jobids = seeds: mean line + min-max band."""
    rows = 2 if with_d else 1
    fig, axes = plt.subplots(rows, 1, figsize=(7.5, 3.2 * rows + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    last_x = 0
    for k, (label, jobs, lam) in enumerate(series):
        col = C[k]
        runs = [parse(find(logdir, j), lam) for j in jobs]
        n = min(len(r[0]) for r in runs)
        x = runs[0][0][:n] + 1  # epochs 1..N
        last_x = max(last_x, x[-1])
        for ax, idx in zip(axes, (1, 2)):
            Y = np.array([r[idx][:n] for r in runs])
            mean = np.nanmean(Y, 0)
            if len(runs) > 1:
                ax.fill_between(x, np.nanmin(Y, 0), np.nanmax(Y, 0), color=col, alpha=0.18, lw=0)
            ax.plot(x, mean, color=col, lw=2, label=label + (f" (mean of {len(runs)} seeds)" if len(runs) > 1 else ""))
    style(axes[0], "pose loss (heatmap MSE)", steps)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    axes[0].yaxis.get_offset_text().set(color=INK2, fontsize=8)
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if with_d:
        style(axes[1], "D loss (BCE)", steps)
    axes[-1].set_xlabel("epoch", color=INK2, fontsize=9)
    axes[-1].set_xlim(1, last_x)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.07), ncol=2,
                   labelcolor=INK, borderaxespad=0)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    foot = "Dashed lines: LR drops. " + note
    fig.text(0.01, 0.005, foot, color=INK2, fontsize=7, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L, O = a.logs, a.out
    seed_note = "Shaded band: min-max over seeds 1-3. "
    repo = "Repo recipe: Adam 1e-4, batch 20, 20 epochs, data/syrip (wrong synthetic labels)."

    figure("1_retrain_backbones.png", "Retraining and other backbones (lambda 0, single runs)",
           [("HRNet-W48 retrain (11499751)", ["11499751"], 0.0),
            ("HRNet-W48 retrain r2 (11506191)", ["11506191"], 0.0),
            ("WRN-50-2 (11504441)", ["11504441"], 0.0),
            ("WRN-50-2 r2 (11506193)", ["11506193"], 0.0),
            ("DenseNet-121 (11504443)", ["11504443"], 0.0)],
           L, O, [10, 15], note=repo + " WRN/DenseNet start from ImageNet weights, HRNet from COCO pose.")
    figure("2_lambda_sweep.png", "Adversarial weight lambda (single runs)",
           [("lambda 0 (11499751)", ["11499751"], 0.0),
            ("lambda 1e-4 (11568463)", ["11568463"], 1e-4),
            ("lambda 5e-4 (11568556)", ["11568556"], 5e-4),
            ("lambda 2.5e-3 (11568480)", ["11568480"], 2.5e-3)],
           L, O, [10, 15], note=repo)
    figure("3_seed_study.png", "Seed study (3 seeds per setting)",
           [("A: lambda 0", ["11572224", "11572259", "11572268"], 0.0),
            ("B: lambda 5e-4", ["11570596", "11570602", "11570614"], 5e-4),
            ("C: lambda 5e-4 + 10-ep D pre-training", ["11570594", "11570600", "11570613"], 5e-4)],
           L, O, [10, 15], note=seed_note + repo + " C's D pre-training epochs are not shown.")
    figure("4_stable_adversarial.png", "Stable adversarial term, D locked (3 seeds per setting)",
           [("neg_bce, 5e-4", ["11575662", "11575687", "11576777"], 5e-4),
            ("confusion, 5e-4", ["11575632", "11575693", "11576832"], 5e-4),
            ("confusion, 2.5e-3", ["11575663", "11575698", "11577176"], 2.5e-3),
            ("confusion, 1e-2", ["11575689", "11576070", "11577381"], 1e-2)],
           L, O, [10, 15], note=seed_note + repo)
    paper = [("lambda 0, corrected labels (11605176)", "11605176", 0.0),
             ("lambda 5e-4, corrected labels (11605184)", "11605184", 5e-4),
             ("lambda 5e-4, A100 twin (11623294)", "11623294", 5e-4)]
    paper = [(l, [j], lam) for l, j, lam in paper if glob.glob(os.path.join(L, f"*-{j}.out"))]
    figure("5_paper_recipe.png", "Paper recipe (seed 1)", paper, L, O, [70, 90],
           note="Adam 1e-3, batch 64, 100 epochs, data/syrip_fix (corrected labels). "
                "Per-epoch value = running mean over batches 0-20 of 32; the log prints it to 5 decimals, "
                "hence the steps. Pending runs are added when their logs exist.")


if __name__ == "__main__":
    main()
