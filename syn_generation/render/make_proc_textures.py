"""Procedural clothing textures for SMIL (template.obj UV layout = SMPL UV layout).

Each texture starts from one of the MINI-RGBD infant textures (skin, face, hair, diaper) and
paints a garment over the covered body regions:
  onesie_long / onesie_short / footed_sleeper / shirt_pants / shirt_shorts / tank_leggings /
  bodysuit (short sleeves, diaper-length legs) / diaper_only (base texture unchanged)
Garment cuts come from SMIL skinning weights + rest-pose (T-pose) geometry, evaluated per texel
by rasterising every face into UV space, so sleeves/legs end at consistent body positions.
Patterns (solid, stripes, dots, checks, heather) are functions of the 3D rest-pose position,
so they continue across UV seams. Base-texture luminance detail is kept as fabric folds.

    python make_proc_textures.py --n 24 --seed 0 --out textures_proc/proc_v1
Writes <out>/proc_XXXX.png (1024x1024) + <out>/textures.json (garment parameters per file).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy import ndimage

from smil_np import SMIL

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = [16, 17, 18, 19, 20, 21, 22, 23]
LEG = [1, 2, 4, 5, 7, 8, 10, 11]
HEAD = [12, 15]
GARMENTS = ['onesie_long', 'onesie_short', 'footed_sleeper', 'shirt_pants', 'shirt_shorts',
            'tank_leggings', 'bodysuit', 'diaper_only']


def load_uv(path):
    vt, fv, ft = [], [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == 'vt':
            vt.append([float(p[1]), float(p[2])])
        elif p[0] == 'f':
            fv.append([int(x.split('/')[0]) - 1 for x in p[1:4]])
            ft.append([int(x.split('/')[1]) - 1 for x in p[1:4]])
    return np.array(vt), np.array(fv), np.array(ft)


def vertex_attrs(m):
    """Per-vertex attributes in the rest pose: pos(3), arm_t, leg_t, armw, legw, headw, y."""
    v, J = m(np.zeros(72), None)
    w = m.weights
    xs = abs(J[16, 0]) * 0.5 + abs(J[17, 0]) * 0.5            # shoulder |x|
    xw = abs(J[20, 0]) * 0.5 + abs(J[21, 0]) * 0.5            # wrist |x|
    yh = (J[1, 1] + J[2, 1]) / 2                              # hip y
    ya = (J[7, 1] + J[8, 1]) / 2                              # ankle y
    arm_t = (np.abs(v[:, 0]) - xs) / (xw - xs)                # 0 shoulder .. 1 wrist (>1 hand)
    leg_t = (yh - v[:, 1]) / (yh - ya)                        # 0 hip .. 1 ankle (>1 foot)
    a = np.c_[v, arm_t, leg_t, w[:, ARM].sum(1), w[:, LEG].sum(1), w[:, HEAD].sum(1),
              w[:, [22, 23]].sum(1), w[:, [10, 11]].sum(1)]
    info = dict(waist_y=float(J[3, 1]), neck_y=float(J[12, 1]), hip_y=float(yh))
    return a, info


def raster_uv(vt, ft, fv, attrs, S):
    """Interpolate per-vertex attrs over each face in UV space -> (S,S,K) + coverage mask."""
    K = attrs.shape[1]
    out = np.zeros((S, S, K), np.float32)
    cov = np.zeros((S, S), bool)
    P = np.c_[vt[:, 0] * (S - 1), (1 - vt[:, 1]) * (S - 1)]   # same mapping as the renderer
    for f in range(len(ft)):
        p = P[ft[f]]
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, S - 1), min(y1, S - 1)
        xx, yy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        q = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        T = np.array([[p[0, 0] - p[2, 0], p[1, 0] - p[2, 0]], [p[0, 1] - p[2, 1], p[1, 1] - p[2, 1]]])
        if abs(np.linalg.det(T)) < 1e-12:
            continue
        l12 = np.linalg.solve(T, (q - p[2]).T).T
        b = np.c_[l12, 1 - l12.sum(1)]
        ok = (b >= -0.02).all(1)
        if not ok.any():
            continue
        b = np.clip(b[ok], 0, None)
        b /= b.sum(1, keepdims=True)
        qi = q[ok].astype(int)
        out[qi[:, 1], qi[:, 0]] = b @ attrs[fv[f]]
        cov[qi[:, 1], qi[:, 0]] = True
    # fill texels outside all faces with the nearest covered texel (avoids seam bleeding)
    _, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
    return out[iy, ix], cov


def hsv(h, s, v):
    c = cv2.cvtColor(np.uint8([[[h * 179, s * 255, v * 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return c.astype(np.float32) / 255


def rand_colour(rng):
    kind = rng.random()
    if kind < 0.45:   # pastel
        return hsv(rng.random(), rng.uniform(0.15, 0.45), rng.uniform(0.8, 0.98))
    if kind < 0.75:   # saturated
        return hsv(rng.random(), rng.uniform(0.5, 0.9), rng.uniform(0.45, 0.9))
    if kind < 0.9:    # white / grey / near-black
        return np.full(3, rng.choice([0.93, 0.7, 0.45, 0.15]) + rng.uniform(-0.04, 0.04), np.float32)
    return hsv(rng.uniform(0.05, 0.12), rng.uniform(0.2, 0.5), rng.uniform(0.5, 0.85))  # beige


def pattern(pos, rng):
    """(S,S,3) colour field from rest-pose positions (metres) + parameter dict."""
    c1, c2 = rand_colour(rng), rand_colour(rng)
    kind = rng.choice(['solid', 'hstripe', 'vstripe', 'dots', 'checks', 'heather'],
                      p=[0.3, 0.2, 0.08, 0.17, 0.1, 0.15])
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    per = rng.uniform(0.012, 0.035)
    if kind == 'solid':
        m = np.zeros_like(x)
    elif kind == 'hstripe':
        m = (np.sin(2 * np.pi * y / per) > rng.uniform(-0.4, 0.4)).astype(np.float32)
    elif kind == 'vstripe':
        m = (np.sin(2 * np.pi * (x + 0.5 * z) / per) > 0).astype(np.float32)
    elif kind == 'dots':
        g = pos / per
        d = np.linalg.norm(g - np.round(g), axis=-1)
        m = (d < rng.uniform(0.18, 0.32)).astype(np.float32)
    elif kind == 'checks':
        m = ((np.floor(x / per) + np.floor(y / per) + np.floor(z / per)) % 2).astype(np.float32)
    else:
        n = ndimage.gaussian_filter(rng.random(x.shape).astype(np.float32), 1.0)
        m = np.clip((n - n.mean()) * 6 + 0.5, 0, 1) * 0.35
        c2 = c1 * 0.75
    col = c1 * (1 - m[..., None]) + c2 * m[..., None]
    return col.astype(np.float32), dict(pattern=str(kind), c1=c1.round(3).tolist(),
                                        c2=c2.round(3).tolist(), period_m=round(float(per), 4))


def soft(x, edge, width=0.015):
    """1 where x < edge, smooth step of +-width."""
    return np.clip((edge - x) / width * 0.5 + 0.5, 0, 1)


def make_one(A, info, base, rng, garment):
    pos = A[..., 0:3]
    arm_t, leg_t, armw, legw, headw, handw, footw = (A[..., i] for i in range(3, 10))
    y = pos[..., 1]
    meta = dict(garment=garment)
    if garment == 'diaper_only':
        return base.copy(), meta
    long_ = garment in ('onesie_long', 'footed_sleeper', 'shirt_pants')
    sleeve = {'onesie_long': rng.uniform(0.85, 0.98), 'footed_sleeper': rng.uniform(0.9, 0.98),
              'shirt_pants': rng.uniform(0.8, 0.98), 'onesie_short': rng.uniform(0.25, 0.45),
              'shirt_shorts': rng.uniform(0.25, 0.45), 'bodysuit': rng.uniform(0.2, 0.4),
              'tank_leggings': -0.05}[garment]
    legl = {'onesie_long': rng.uniform(0.85, 0.98), 'footed_sleeper': 2.0,
            'shirt_pants': rng.uniform(0.85, 0.98), 'onesie_short': rng.uniform(0.3, 0.5),
            'shirt_shorts': rng.uniform(0.3, 0.5), 'bodysuit': 0.04,
            'tank_leggings': rng.uniform(0.8, 0.98)}[garment]
    neck = info['neck_y'] - rng.uniform(0.012, 0.03)
    armish = np.clip((armw - 0.4) / 0.2, 0, 1)
    legish = np.clip((legw - 0.4) / 0.2, 0, 1)
    torso = (1 - armish) * (1 - legish) * soft(y, neck, 0.004) * np.clip(1 - headw * 2, 0, 1)
    arm_cov = armish * soft(arm_t, sleeve) * (1 - np.clip(handw * 3 - 1, 0, 1))
    leg_cov = legish * soft(leg_t, legl)
    if garment != 'footed_sleeper':
        leg_cov *= 1 - np.clip(footw * 3 - 1, 0, 1)
    cover = np.clip(torso + arm_cov + leg_cov, 0, 1)

    top, pt = pattern(pos, rng)
    meta.update(sleeve=round(float(sleeve), 3), leg=round(float(legl), 3), top=pt)
    if garment in ('shirt_pants', 'shirt_shorts', 'tank_leggings'):
        bottom, pb = pattern(pos, rng)
        meta['bottom'] = pb
        waist = info['waist_y'] + rng.uniform(-0.01, 0.02)
        is_bottom = np.maximum(legish, (1 - armish) * (1 - soft(y, waist, 0.004)))
        cloth = top * (1 - is_bottom[..., None]) + bottom * is_bottom[..., None]
    else:
        cloth = top
    # trim: contrasting band at sleeve / leg hems and neckline
    if rng.random() < 0.5:
        tc = rand_colour(rng)
        band = np.zeros_like(y)
        band = np.maximum(band, armish * (np.abs(arm_t - sleeve) < 0.04))
        if legl < 1.5:
            band = np.maximum(band, legish * (np.abs(leg_t - legl) < 0.04))
        band = np.maximum(band, (1 - armish) * (1 - legish) * (np.abs(y - neck) < 0.006))
        cloth = cloth * (1 - band[..., None]) + tc * band[..., None]
        meta['trim'] = tc.round(3).tolist()
    # fabric shading: keep base-texture luminance detail (folds) and a little noise
    L = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    detail = L - cv2.GaussianBlur(L, (0, 0), 6)
    shade = 1 + 0.8 * detail + 0.04 * ndimage.gaussian_filter(rng.standard_normal(L.shape), 2)
    cloth = np.clip(cloth * shade[..., None], 0, 1)
    out = base * (1 - cover[..., None]) + cloth * cover[..., None]
    return out.astype(np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--base-dir', default=os.path.join(HERE, 'textures', 'infant_txt'))
    ap.add_argument('--smil', default=os.path.join(HERE, 'smil_web.pkl'))
    ap.add_argument('--template', default=os.path.join(HERE, 'template.obj'))
    ap.add_argument('--garments', default=','.join(GARMENTS))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit('%s exists' % a.out)
    m = SMIL(a.smil)
    vt, fv, ft = load_uv(a.template)
    if not np.array_equal(fv, m.f):
        raise SystemExit('template.obj faces differ from SMIL faces')
    attrs, info = vertex_attrs(m)
    A, _ = raster_uv(vt, ft, fv, attrs, a.size)
    bases = sorted(glob.glob(os.path.join(a.base_dir, '*.png')))
    garments = a.garments.split(',')
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out)
    rec = []
    for i in range(a.n):
        bp = bases[rng.integers(len(bases))]
        base = cv2.resize(cv2.imread(bp), (a.size, a.size)).astype(np.float32) / 255
        g = garments[i % len(garments)]            # cycle garments so a small set covers all
        tex, meta = make_one(A, info, base, rng, g)
        fn = 'proc_%04d.png' % i
        cv2.imwrite(os.path.join(a.out, fn), np.round(tex * 255).astype(np.uint8))
        meta.update(file=fn, base=os.path.basename(bp))
        rec.append(meta)
    json.dump(dict(args=vars(a), textures=rec), open(os.path.join(a.out, 'textures.json'), 'w'), indent=1)
    print('wrote %d textures to %s' % (a.n, a.out))


if __name__ == '__main__':
    main()
