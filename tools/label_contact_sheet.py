"""Contact sheet of keypoint labels drawn on training images, to check labels by eye.

Draws the same random sample of images once per annotation file, in the same order, so sheets made
from different files can be compared tile by tile. Keypoints are drawn exactly as stored (image pixel
coordinates), as the training loader would use them. Left-side joints/limbs blue, right-side orange,
face and torso centre lines grey; unlabelled keypoints (v=0) are not drawn.

Example:
  python tools/label_contact_sheet.py --img-dir data/syrip/images/train_infant \
      --ann 2020=path/2020.json --ann 2021=path/2021.json --synthetic-only --n 20 --cols 10 --seed 0 \
      --pairs train10142,train10394,train10940,train10991 --out outputs/label_check

--pairs (exactly two --ann files): also writes pairs_<A>_vs_<B>.jpg, the listed images enlarged with the two
label sets side by side, two pairs per row.
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

LEFT, RIGHT, MID = (42, 120, 214), (235, 104, 52), (120, 120, 120)
# COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
EDGES = [(5, 7, LEFT), (7, 9, LEFT), (11, 13, LEFT), (13, 15, LEFT), (5, 11, LEFT),
         (6, 8, RIGHT), (8, 10, RIGHT), (12, 14, RIGHT), (14, 16, RIGHT), (6, 12, RIGHT),
         (5, 6, MID), (11, 12, MID), (0, 1, LEFT), (1, 3, LEFT), (0, 2, RIGHT), (2, 4, RIGHT)]


def side(j):
    return MID if j == 0 else (LEFT if j % 2 == 1 else RIGHT)


def tile(img_path, ann, size):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    s = size / max(W, H)
    im = im.resize((max(1, round(W * s)), max(1, round(H * s))))
    canvas = Image.new("RGB", (size, size), (252, 252, 251))
    ox, oy = (size - im.width) // 2, (size - im.height) // 2
    canvas.paste(im, (ox, oy))
    d = ImageDraw.Draw(canvas)
    k = ann["keypoints"]
    pts = {j: (ox + k[3 * j] * s, oy + k[3 * j + 1] * s) for j in range(17) if k[3 * j + 2] > 0}
    lw = max(2, size // 110)
    for a, b, c in EDGES:
        if a in pts and b in pts:
            d.line([pts[a], pts[b]], fill=c, width=lw)
    r = lw + 1
    for j, (x, y) in pts.items():
        d.ellipse([x - r, y - r, x + r, y + r], fill=side(j), outline=(255, 255, 255))
    out_of_frame = sum(not (0 <= k[3 * j] < W and 0 <= k[3 * j + 1] < H) for j in range(17) if k[3 * j + 2] > 0)
    return canvas, (W, H), out_of_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ann", action="append", required=True, help="NAME=path/to/annotations.json (repeatable)")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--tile", type=int, default=260)
    ap.add_argument("--pairs", default="", help="comma-separated file names (no .jpg) for the side-by-side figure")
    ap.add_argument("--pair-tile", type=int, default=420)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sets = []
    for spec in a.ann:
        name, path = spec.split("=", 1)
        js = json.load(open(path))
        ims = {i["id"]: i for i in js["images"]}
        anns = {}
        for x in js["annotations"]:
            anns.setdefault(x["image_id"], x)
        sets.append((name, path, ims, anns))

    common = set.intersection(*[set(s[2]) for s in sets])
    if a.synthetic_only:
        common = {i for i in common if sets[0][2][i].get("is_synthetic")}
    ids = sorted(random.Random(a.seed).sample(sorted(common), min(a.n, len(common))))
    for name, _, ims, _ in sets[1:]:
        assert all(ims[i]["file_name"] == sets[0][2][i]["file_name"] for i in ids), "file names differ between sets"

    os.makedirs(a.out, exist_ok=True)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    rows = (len(ids) + a.cols - 1) // a.cols
    report = []
    for name, path, ims, anns in sets:
        T, cap = a.tile, 20
        sheet = Image.new("RGB", (a.cols * T, rows * (T + cap) + 34), (252, 252, 251))
        d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{name} labels: {os.path.basename(path)}  ({len(ids)} random {'synthetic ' if a.synthetic_only else ''}"
                       f"images, seed {a.seed}; blue = left, orange = right)", fill=(11, 11, 11), font=font)
        size_mismatch = oof = 0
        for n, i in enumerate(ids):
            fn = ims[i]["file_name"]
            t, (W, H), o = tile(os.path.join(a.img_dir, fn), anns[i], T - 6)
            r, c = divmod(n, a.cols)
            x0, y0 = c * T + 3, 34 + r * (T + cap)
            sheet.paste(t, (x0, y0))
            mism = (W, H) != (ims[i]["width"], ims[i]["height"])
            size_mismatch += mism
            oof += o > 0
            label = fn.replace(".jpg", "") + ("  size!" if mism else "") + (f"  {o} kp outside" if o else "")
            d.text((x0 + 2, y0 + T - 4), label, fill=(82, 81, 78), font=font)
        p = os.path.join(a.out, f"labels_{name}.jpg")
        sheet.save(p, quality=88)
        report.append(f"{name}: {p}; json size != image size: {size_mismatch}/{len(ids)}; "
                      f"images with keypoints outside the frame: {oof}/{len(ids)}")
    if a.pairs:
        assert len(sets) == 2, "--pairs needs exactly two --ann files"
        names = a.pairs.split(",")
        by_name = {im["file_name"].rsplit(".", 1)[0]: i for i, im in sets[0][2].items()}
        T, cap, per_row = a.pair_tile, 22, 2
        rows_p = (len(names) + per_row - 1) // per_row
        sheet = Image.new("RGB", (per_row * 2 * T, rows_p * (T + cap) + 34), (252, 252, 251))
        d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{sets[0][0]} (left) vs {sets[1][0]} (right) labels on the same image; blue = left, orange = right",
               fill=(11, 11, 11), font=font)
        for n, nm in enumerate(names):
            i = by_name[nm]
            r, c = divmod(n, per_row)
            for k, (sname, _, ims, anns) in enumerate(sets):
                t, (W, H), _ = tile(os.path.join(a.img_dir, ims[i]["file_name"]), anns[i], T - 6)
                x0, y0 = (c * 2 + k) * T + 3, 34 + r * (T + cap)
                sheet.paste(t, (x0, y0))
                d.text((x0 + 2, y0 + T - 4), f"{nm} ({W}x{H})  {sname}", fill=(82, 81, 78), font=font)
        p = os.path.join(a.out, f"pairs_{sets[0][0]}_vs_{sets[1][0]}.jpg")
        sheet.save(p, quality=88)
        report.append(f"pairs: {p} ({len(names)} images)")
    open(os.path.join(a.out, "sample_ids.txt"), "w").write(" ".join(str(i) for i in ids) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
