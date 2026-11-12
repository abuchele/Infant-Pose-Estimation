"""Visualize successful and failed pose predictions on a COCO-format test set.

Scores each test annotation by OKS against the model's prediction for that image (same formula
and COCO sigmas as pycocotools; the GT `area` is the scale), then writes:
  per_image.csv        every image ranked by OKS, with posture and worst keypoint
  best_grid.jpg        the N highest-OKS images (successes)
  worst_grid.jpg       the N lowest-OKS images (failures)
  keypoints.png        per-keypoint failure rate and OKS by posture
  summary.txt          counts, the fraction of images above OKS 0.5 / 0.75 / 0.9, and images with
                       left/right swaps (a L/R pair that fits the GT >0.2 better with labels exchanged)

Each tile is cropped around the GT box. Blue = ground truth, orange = prediction, a grey line joins
each GT/prediction pair, and keypoints with per-keypoint similarity < --kp-fail are circled in red.
Hollow orange markers are predictions with confidence < --conf.

Example:
  python tools/vis_cases.py \
      --gt data/syrip/annotations/person_keypoints_validate_infant.json \
      --pred outputs/vis_cases/nb_l5e-4_s2/keypoints_validate_infant_results_0.json \
      --img-dir data/syrip/images/validate_infant --out outputs/vis_cases/nb_l5e-4_s2 --n 6 --cols 3
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                   1.07, 1.07, .87, .87, .89, .89]) / 10.0
GT_C, PRED_C, FAIL_C = "#1f77b4", "#ff7f0e", "#d62728"


def oks(gt, pred, area):
    """Return (OKS, per-keypoint similarity with NaN for unlabeled, pixel distance)."""
    g = np.asarray(gt, float).reshape(-1, 3)
    p = np.asarray(pred, float).reshape(-1, 3)
    vis = g[:, 2] > 0
    d2 = ((g[:, :2] - p[:, :2]) ** 2).sum(1)
    k = np.exp(-d2 / (2 * area * (2 * SIGMAS) ** 2 + np.spacing(1)))
    k[~vis] = np.nan
    return float(np.nanmean(k)) if vis.any() else float("nan"), k, np.sqrt(d2)


PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]  # COCO left/right


def lr_swaps(gt, pred, area, names, margin=0.2):
    """Left/right pairs whose prediction fits the GT clearly better with the labels exchanged."""
    p = np.asarray(pred, float).reshape(-1, 3)
    out = []
    for a, b in PAIRS:
        q = p.copy()
        q[[a, b]] = q[[b, a]]
        _, k0, _ = oks(gt, p, area)
        _, k1, _ = oks(gt, q, area)
        if not (np.isnan(k0[a]) or np.isnan(k0[b])) and k1[[a, b]].mean() > k0[[a, b]].mean() + margin:
            out.append(names[a].replace("left_", ""))
    return out


def draw(ax, img, gt, pred, ks, bbox, skeleton, title, conf_thr, kp_fail):
    g = np.asarray(gt, float).reshape(-1, 3)
    p = np.asarray(pred, float).reshape(-1, 3)
    x, y, w, h = bbox
    m = 0.15 * max(w, h)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(img.width, x + w + m), min(img.height, y + h + m)
    ax.imshow(img.crop((int(x0), int(y0), int(x1), int(y1))), extent=(x0, x1, y1, y0))
    for a, b in skeleton:
        a, b = a - 1, b - 1
        if g[a, 2] > 0 and g[b, 2] > 0:
            ax.plot(g[[a, b], 0], g[[a, b], 1], color=GT_C, lw=1.6, alpha=0.9)
        ax.plot(p[[a, b], 0], p[[a, b], 1], color=PRED_C, lw=1.6, alpha=0.9)
    for j in range(len(g)):
        if g[j, 2] > 0:
            ax.plot([g[j, 0], p[j, 0]], [g[j, 1], p[j, 1]], color="0.6", lw=0.8)
            ax.plot(g[j, 0], g[j, 1], "o", color=GT_C, ms=3.5)
        low = p[j, 2] < conf_thr
        ax.plot(p[j, 0], p[j, 1], "o", ms=3.5, mec=PRED_C,
                mfc="none" if low else PRED_C)
        if not np.isnan(ks[j]) and ks[j] < kp_fail:
            ax.plot(g[j, 0], g[j, 1], "o", ms=11, mfc="none", mec=FAIL_C, mew=1.4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def grid(rows, path, header, args, imgs, gts, preds, names, skeleton):
    n = len(rows)
    cols = min(args.cols, n)
    nr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(3.6 * cols, 3.9 * nr + 0.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        aid = r["ann_id"]
        g = gts[aid]
        img = Image.open(os.path.join(args.img_dir, imgs[g["image_id"]]["file_name"])).convert("RGB")
        worst = r["worst_kp"]
        swap = f"  L/R swap: {r['lr_swaps']}" if r["lr_swaps"] else ""
        title = (f"{r['file']}  OKS {r['oks']:.3f}\n{r['posture']}  worst: {worst}{swap}")
        draw(ax, img, g["keypoints"], preds[g["image_id"]]["keypoints"], r["_ks"], g["bbox"],
             skeleton, title, args.conf, args.kp_fail)
    fig.suptitle(header + "\n(blue = GT, orange = prediction, red ring = keypoint similarity "
                 f"< {args.kp_fail}, hollow = confidence < {args.conf})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.9 * nr + 0.5)), h_pad=2.0)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True, help="COCO keypoint results json from test_adaptive_model.py")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6, help="tiles per best/worst grid")
    ap.add_argument("--cols", type=int, default=3, help="tiles per row (default 6 tiles in 2 rows)")
    ap.add_argument("--conf", type=float, default=0.3, help="low-confidence threshold (hollow marker)")
    ap.add_argument("--kp-fail", type=float, default=0.5, help="per-keypoint similarity counted as failed")
    ap.add_argument("--label", default="", help="run name for figure titles")
    args = ap.parse_args()

    G = json.load(open(args.gt))
    R = json.load(open(args.pred))
    imgs = {i["id"]: i for i in G["images"]}
    names = G["categories"][0]["keypoints"]
    skeleton = G["categories"][0]["skeleton"]
    gts = {a["id"]: a for a in G["annotations"] if not a.get("iscrowd", 0)}

    by_img = defaultdict(list)
    for r in R:
        by_img[r["image_id"]].append(r)
    # One prediction per GT box: the highest-OKS prediction on that image (Test sets here have one infant).
    rows, preds = [], {}
    missing = 0
    for aid, g in gts.items():
        cands = by_img.get(g["image_id"], [])
        if not cands:
            missing += 1
            continue
        scored = [(oks(g["keypoints"], c["keypoints"], g["area"]), c) for c in cands]
        (o, ks, dist), best = max(scored, key=lambda t: t[0][0])
        preds[g["image_id"]] = best
        j = int(np.nanargmin(ks))
        im = imgs[g["image_id"]]
        rows.append({"ann_id": aid, "image_id": g["image_id"], "file": im["file_name"],
                     "posture": im.get("posture", "?"), "oks": o,
                     "n_failed_kp": int(np.nansum(ks < args.kp_fail)),
                     "worst_kp": names[j], "worst_kp_sim": float(ks[j]),
                     "worst_kp_px": float(dist[j]), "det_score": best["score"],
                     "lr_swaps": " ".join(lr_swaps(g["keypoints"], best["keypoints"], g["area"], names)),
                     "_ks": ks})
    rows.sort(key=lambda r: -r["oks"])
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "per_image.csv"), "w", newline="") as f:
        cols = [k for k in rows[0] if not k.startswith("_")] + names
        w = csv.writer(f)
        w.writerow(["rank"] + cols)
        for i, r in enumerate(rows, 1):
            w.writerow([i] + [round(r[k], 4) if isinstance(r[k], float) else r[k] for k in cols[:-len(names)]]
                       + [round(float(v), 4) for v in r["_ks"]])

    n = min(args.n, len(rows))
    lab = f"{args.label}: " if args.label else ""
    grid(rows[:n], os.path.join(args.out, "best_grid.jpg"), f"{lab}best {n} of {len(rows)} by OKS",
         args, imgs, gts, preds, names, skeleton)
    grid(rows[::-1][:n], os.path.join(args.out, "worst_grid.jpg"), f"{lab}worst {n} of {len(rows)} by OKS",
         args, imgs, gts, preds, names, skeleton)

    K = np.array([r["_ks"] for r in rows])
    fail = np.nanmean(np.where(np.isnan(K), np.nan, K < args.kp_fail), axis=0)
    post = defaultdict(list)
    for r in rows:
        post[r["posture"]].append(r["oks"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1.3]})
    order = np.argsort(-fail)
    a1.bar(range(len(names)), fail[order], color=PRED_C)
    a1.set_xticks(range(len(names)), [names[i] for i in order], rotation=60, ha="right", fontsize=8)
    a1.set_ylabel(f"fraction of images with similarity < {args.kp_fail}")
    a1.set_title("Per-keypoint failure rate", fontsize=10)
    a1.spines[["top", "right"]].set_visible(False)
    ps = sorted(post)
    a2.boxplot([post[p] for p in ps], showfliers=True)
    a2.set_xticks(range(1, len(ps) + 1), [f"{p}\n(n={len(post[p])})" for p in ps], fontsize=8)
    a2.set_ylabel("image OKS")
    a2.set_title("OKS by posture", fontsize=10)
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{lab}{len(rows)} test images", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "keypoints.png"), dpi=120)
    plt.close(fig)

    o = np.array([r["oks"] for r in rows])
    lines = [f"run: {args.label or args.pred}", f"gt: {args.gt}", f"pred: {args.pred}",
             f"annotations scored: {len(rows)} (no prediction: {missing})",
             f"mean OKS {o.mean():.4f}, median {np.median(o):.4f}",
             *[f"OKS >= {t}: {(o >= t).mean():.3f}" for t in (0.5, 0.75, 0.9)],
             "mean OKS by posture: " + ", ".join(f"{p} {np.mean(post[p]):.3f} (n={len(post[p])})" for p in ps),
             f"images with a left/right swap: {sum(bool(r['lr_swaps']) for r in rows)} "
             f"({', '.join(r['file'] + ': ' + r['lr_swaps'] for r in rows if r['lr_swaps'])})",
             "per-keypoint failure rate: " + ", ".join(f"{names[i]} {fail[i]:.2f}" for i in order)]
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
