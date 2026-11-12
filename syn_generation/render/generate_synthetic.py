"""Generate synthetic SyRIP-style infant images + COCO-17 keypoint annotations.

Replaces render/image_generation.py, which needs opendr 0.78 (does not build on Python >= 3.9
or current macOS; see AGENT.md 2026-11-03) and writes no labels. Same idea as the original:
SMIL body (pose + shape) -> textured mesh (template.obj UVs, infant textures) rendered with a
pinhole camera (f = W/2, principal point at the image centre, like the original) over a
background image, several global rotations per body. Differences from the original:
  - pure numpy/OpenCV rasteriser (face-id painter's algorithm + perspective-correct barycentric
    interpolation, back faces culled), so it runs on any CPU (Mac or Explorer)
  - camera distance and image offset chosen so the whole infant is in frame
  - proper 3D view rotations (yaw/pitch/roll) instead of adding to the axis-angle vector
  - optional Lambertian shading (default on; the released SyRIP synthetic images are shaded)
  - writes COCO-17 keypoints (SMIL joints for body points, SMPL-topology vertices for
    nose/eyes/ears, see smil_np.SMIL), visibility (2 visible / 1 occluded: for joints, another body part is drawn at the joint's
    pixel; for nose/eyes/ears, a depth test),
    tight bbox and area from the rendered mask, in the SyRIP annotation format

Body sources:
  --source mini    : MINI-RGBD smil_params (pose + per-sequence betas), every --stride-th frame
  --source fitted  : SMPLify-X/SMIL fits, <bodies-dir>/<name>/000.pkl (global_orient, body_pose,
                     betas), as written by smplifyx/main.py

    python generate_synthetic.py --source mini --mini-root MINI-RGBD_web --stride 50 \
        --views 4 --out outputs/mini_test --start-id 20000 --max-images 20 --overlays 20
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smil_np import SMIL  # noqa: E402

COCO_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
              'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
              'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
SURFACE_KPTS = set(SMIL.COCO_VERTS)   # nose/eyes/ears lie on the surface; others are inside

G = {}  # per-worker globals


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_obj_uv(path):
    vt, ft = [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == 'vt':
            vt.append([float(p[1]), float(p[2])])
        elif p[0] == 'f':
            ft.append([int(x.split('/')[1]) - 1 for x in p[1:4]])
    return np.array(vt), np.array(ft)


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in files
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    return sorted(out)


def body_list(args):
    """[(name, pose72, betas)]"""
    bodies = []
    if args.source == 'mini':
        seqs = args.mini_seqs.split(',') if args.mini_seqs else \
            sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.mini_root, '[0-9][0-9]')))
        for s in seqs:
            betas = np.loadtxt(os.path.join(args.mini_root, s, 'smil_shape_betas.txt'))
            poses = sorted(glob.glob(os.path.join(args.mini_root, s, 'smil_params', '*_pose.txt')))
            for p in poses[::args.stride]:
                fr = os.path.basename(p)[:5]
                bodies.append(('mini/%s/%s' % (s, fr), np.loadtxt(p), betas))
    else:
        for p in sorted(glob.glob(os.path.join(args.bodies_dir, '*', '000.pkl'))):
            d = pickle.load(open(p, 'rb'))
            pose = np.concatenate([np.ravel(d['global_orient']), np.ravel(d['body_pose'])])
            bodies.append(('fitted/%s' % os.path.basename(os.path.dirname(p)), pose, np.ravel(d['betas'])))
    return bodies


def init_worker(args):
    G['args'] = args
    G['smil'] = SMIL(args.smil)
    G['vt'], G['ft'] = load_obj_uv(args.template)
    cats = [c for c in args.bg_cats.split(',') if c]
    G['bgs'] = [b for c in cats for b in list_images(os.path.join(args.bg_dir, c))] if cats \
        else list_images(args.bg_dir)
    G['texs'] = sorted(glob.glob(os.path.join(args.tex_dir, '**', '*.png'), recursive=True) +
                       glob.glob(os.path.join(args.tex_dir, '**', '*.jpg'), recursive=True))
    G['tex_cache'] = {}
    m = G['smil']
    # body part of each face = joint with the largest skinning weight (majority over its vertices)
    vpart = m.weights.argmax(1)
    fp = vpart[m.f]
    G['face_part'] = np.where(fp[:, 1] == fp[:, 2], fp[:, 1], fp[:, 0])
    children = {j: set(np.nonzero(m.parent == j)[0]) for j in range(24)}
    G['allowed'] = {j: {j, int(m.parent[j])} | children[j] for j in range(24)}
    if not G['bgs'] or not G['texs']:
        raise RuntimeError('no backgrounds in %s or no textures in %s' % (args.bg_dir, args.tex_dir))


def place_camera(v, kp, W, H, f, rng, scale_range, margin):
    """Translate the (root-centred) body so it projects fully inside the image."""
    ext = max(np.ptp(v[:, 0]), np.ptp(v[:, 1]))
    s = rng.uniform(*scale_range)
    d = f * ext / (s * min(W, H)) + max(0.0, -v[:, 2].min())
    v = v + np.array([0, 0, d])
    kp = kp + np.array([0, 0, d])
    for _ in range(20):
        uv = proj(v, W, H, f)
        lo, hi = uv.min(0), uv.max(0)
        # random image-space shift that keeps the bbox inside the margins
        ax = (margin - lo[0], W - margin - hi[0])
        ay = (margin - lo[1], H - margin - hi[1])
        if ax[0] > ax[1] or ay[0] > ay[1]:
            v = v + np.array([0, 0, 0.1 * d])
            kp = kp + np.array([0, 0, 0.1 * d])
            continue
        sx, sy = rng.uniform(*ax), rng.uniform(*ay)
        zc = np.median(v[:, 2])
        t = np.array([sx * zc / f, sy * zc / f, 0])
        v2, kp2 = v + t, kp + t
        uv2 = proj(v2, W, H, f)
        if uv2.min() >= 0 and (uv2[:, 0] < W).all() and (uv2[:, 1] < H).all():
            return v2, kp2
        v = v + np.array([0, 0, 0.05 * d])
        kp = kp + np.array([0, 0, 0.05 * d])
    return None, None


def proj(p, W, H, f):
    return np.stack([f * p[:, 0] / p[:, 2] + W / 2.0, f * p[:, 1] / p[:, 2] + H / 2.0], 1)


def render(v, faces, vt, ft, tex, W, H, f, light):
    """Returns rgb (H,W,3) float [0,1], mask (H,W) bool, depth (H,W) (inf where empty),
    face index per pixel (H,W) (-1 where empty)."""
    uv = proj(v, W, H, f)
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = (n * (-tri.mean(1))).sum(1) > 0                       # camera at origin, CCW outward
    idx = np.nonzero(front)[0]
    zmean = tri[idx, :, 2].mean(1)
    idx = idx[np.argsort(-zmean)]                                 # far to near (painter's)
    fid = np.zeros((H, W, 3), np.uint8)
    pts = np.round(uv[faces[idx]] * 16).astype(np.int32)          # 4-bit sub-pixel
    for k, i in enumerate(idx):
        c = int(i) + 1
        cv2.fillConvexPoly(fid, pts[k], (c & 255, (c >> 8) & 255, (c >> 16) & 255),
                           lineType=cv2.LINE_8, shift=4)
    fid = fid[:, :, 0].astype(np.int64) | (fid[:, :, 1].astype(np.int64) << 8) | \
        (fid[:, :, 2].astype(np.int64) << 16)
    mask = fid > 0
    ys, xs = np.nonzero(mask)
    fi = fid[ys, xs] - 1
    a, b, c = uv[faces[fi, 0]], uv[faces[fi, 1]], uv[faces[fi, 2]]
    p = np.stack([xs + 0.5, ys + 0.5], 1)
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    bw = np.clip(np.stack([1 - w1 - w2, w1, w2], 1), 0, 1)
    bw /= bw.sum(1, keepdims=True)
    iz = 1.0 / v[faces[fi], 2]                                    # perspective-correct weights
    pw = bw * iz
    z = 1.0 / pw.sum(1)
    pw = pw * z[:, None]
    tuv = (vt[ft[fi]] * pw[:, :, None]).sum(1)
    th, tw = tex.shape[:2]
    mx = np.clip(tuv[:, 0] * (tw - 1), 0, tw - 1)                # bilinear texture lookup
    my = np.clip((1 - tuv[:, 1]) * (th - 1), 0, th - 1)
    x0, y0 = np.floor(mx).astype(np.int64), np.floor(my).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, tw - 1), np.minimum(y0 + 1, th - 1)
    fx, fy = (mx - x0)[:, None], (my - y0)[:, None]
    col = (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy) +
           tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)
    if light is not None:
        vn = np.zeros_like(v)
        fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        for j in range(3):
            np.add.at(vn, faces[:, j], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        nn = (vn[faces[fi]] * pw[:, :, None]).sum(1)
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        shade = light[0] + light[1] * np.clip(nn @ light[2], 0, 1)
        col = col * shade[:, None]
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[ys, xs] = col
    depth = np.full((H, W), np.inf)
    depth[ys, xs] = z
    face = np.full((H, W), -1, np.int64)
    face[ys, xs] = fi
    return rgb, mask, depth, face


def background(path, W, H, rng):
    bg = cv2.imread(path)
    if bg is None:
        raise RuntimeError('cannot read background ' + path)
    s = max(W / bg.shape[1], H / bg.shape[0]) * rng.uniform(1.0, 1.3)
    bg = cv2.resize(bg, (int(math.ceil(bg.shape[1] * s)), int(math.ceil(bg.shape[0] * s))),
                    interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    y = rng.integers(0, bg.shape[0] - H + 1)
    x = rng.integers(0, bg.shape[1] - W + 1)
    bg = bg[y:y + H, x:x + W]
    if rng.random() < 0.5:
        bg = bg[:, ::-1]
    return bg.astype(np.float32) / 255


def make_one(job):
    args, m = G['args'], G['smil']
    img_id, name, pose, betas, view, n_views = job
    rng = np.random.default_rng([args.seed, img_id])
    W, H = args.width, args.height
    f = W / 2.0 * args.focal_scale
    verts, joints = m(pose, betas)
    kp3 = m.coco17(verts, joints)
    root = joints[0].copy()
    yaw = 2 * math.pi * (view + rng.uniform(-0.5, 0.5)) / n_views
    pitch = math.radians(rng.uniform(-args.max_tilt, args.max_tilt))
    roll = math.radians(rng.uniform(-args.max_tilt, args.max_tilt))
    R = rot('z', roll) @ rot('x', pitch) @ rot('y', yaw)
    v = (verts - root) @ R.T
    kp = (kp3 - root) @ R.T
    v, kp = place_camera(v, kp, W, H, f, rng, (args.min_scale, args.max_scale), args.margin)
    if v is None:
        return None
    tex_path = G['texs'][rng.integers(len(G['texs']))]
    tex = G['tex_cache'].get(tex_path)
    if tex is None:
        tex = cv2.imread(tex_path).astype(np.float32) / 255
        G['tex_cache'][tex_path] = tex
    light = None
    if args.shading == 'lambert':
        # direction from the surface towards the light: roughly towards the camera (-z)
        ld = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), -1.0])
        light = (args.ambient, 1 - args.ambient, ld / np.linalg.norm(ld))
    rgb, mask, depth, face = render(v, m.f, G['vt'], G['ft'], tex, W, H, f, light)
    bg_path = G['bgs'][rng.integers(len(G['bgs']))]
    bg = background(bg_path, W, H, rng)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)[:, :, None]
    img = rgb * alpha + bg * (1 - alpha)
    img = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)

    uv = proj(kp, W, H, f)
    kps = []
    for k in range(17):
        x, y = uv[k]
        xi, yi = int(x), int(y)
        if not (0 <= x < W and 0 <= y < H):
            kps += [0.0, 0.0, 0]
            continue
        if not mask[yi, xi]:
            occluded = False
        elif k in SURFACE_KPTS:   # surface point: depth test
            occluded = kp[k, 2] > depth[yi, xi] + args.surface_tol
        else:                     # joint: occluded if another body part is drawn over it
            occluded = G['face_part'][face[yi, xi]] not in G['allowed'][SMIL.COCO_JOINTS[k]]
        kps += [round(float(x), 2), round(float(y), 2), 1 if occluded else 2]
    ys, xs = np.nonzero(mask)
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
    fn = 'train%05d.jpg' % img_id
    cv2.imwrite(os.path.join(args.out, 'images', fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if args.overlays and img_id - args.start_id < args.overlays:
        ov = img.copy()
        for a_, b_ in SKELETON:
            if kps[3 * (a_ - 1) + 2] and kps[3 * (b_ - 1) + 2]:
                cv2.line(ov, (int(kps[3 * (a_ - 1)]), int(kps[3 * (a_ - 1) + 1])),
                         (int(kps[3 * (b_ - 1)]), int(kps[3 * (b_ - 1) + 1])), (0, 255, 255), 1)
        for k in range(17):
            if kps[3 * k + 2]:
                col = (0, 255, 0) if kps[3 * k + 2] == 2 else (0, 0, 255)
                col = (255, 0, 0) if 'left' in COCO_NAMES[k] and kps[3 * k + 2] == 2 else col
                cv2.circle(ov, (int(kps[3 * k]), int(kps[3 * k + 1])), 3, col, -1)
        cv2.rectangle(ov, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.out, 'overlays', fn), ov)
    image = {'file_name': fn, 'is_synthetic': True, 'frame_id': 0, 'height': H, 'id': img_id,
             'is_labeled': True, 'nframes': 0, 'original_file_name': '%s/view%d' % (name, view),
             'width': W, 'image_source': 'generate_synthetic:' + args.source}
    ann = {'bbox': bbox, 'bbox_head': [], 'category_id': 1, 'id': img_id, 'image_id': img_id,
           'keypoints': kps, 'scores': [], 'track_id': 0,
           'num_keypoints': int(sum(1 for k in range(17) if kps[3 * k + 2] > 0)),
           'segmentation': '[]', 'area': float(mask.sum()), 'iscrowd': 0}
    meta = {'id': img_id, 'body': name, 'view': view, 'yaw_deg': round(math.degrees(yaw), 1),
            'pitch_deg': round(math.degrees(pitch), 1), 'roll_deg': round(math.degrees(roll), 1),
            'texture': os.path.basename(tex_path), 'background': os.path.relpath(bg_path, args.bg_dir)}
    return image, ann, meta


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', choices=['mini', 'fitted'], required=True)
    p.add_argument('--mini-root', default=os.path.join(here, 'MINI-RGBD_web'))
    p.add_argument('--mini-seqs', default='', help='comma list, e.g. 01,02 (default: all)')
    p.add_argument('--stride', type=int, default=50, help='MINI-RGBD: use every N-th frame')
    p.add_argument('--bodies-dir', default=os.path.join(here, 'bodies'))
    p.add_argument('--smil', default=os.path.join(here, 'smil_web.pkl'))
    p.add_argument('--template', default=os.path.join(here, 'template.obj'))
    p.add_argument('--bg-dir', default=os.path.join(here, 'bg_img', 'lsun'))
    p.add_argument('--bg-cats', default='',
                   help='comma list of sub-folders of --bg-dir to use, e.g. bedroom,living_room '
                        '(default: all). Some LSUN scenes (classroom, restaurant, conference_room) '
                        'contain unlabelled people')
    p.add_argument('--tex-dir', default=os.path.join(here, 'textures'))
    p.add_argument('--out', required=True)
    p.add_argument('--start-id', type=int, default=20000,
                   help='first image id; file names are train%%05d.jpg, so ids must be < 100000 '
                        'and must not collide with existing SyRIP ids (real 1-500, synthetic 10000-10999)')
    p.add_argument('--views', type=int, default=10, help='global rotations per body (original: 10)')
    p.add_argument('--max-images', type=int, default=0, help='stop after this many (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--focal-scale', type=float, default=1.0, help='f = W/2 * this (original: 1)')
    p.add_argument('--min-scale', type=float, default=0.45, help='body size / min(W,H), low')
    p.add_argument('--max-scale', type=float, default=0.85, help='body size / min(W,H), high')
    p.add_argument('--max-tilt', type=float, default=30.0, help='max |pitch|, |roll| in degrees')
    p.add_argument('--margin', type=int, default=8, help='min px between body and image border')
    p.add_argument('--shading', choices=['lambert', 'none'], default='lambert')
    p.add_argument('--ambient', type=float, default=0.55)
    p.add_argument('--surface-tol', type=float, default=0.01, help='m; same for nose/eyes/ears')
    p.add_argument('--overlays', type=int, default=0, help='write keypoint overlays for the first N')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    if not 0 < args.start_id < 100000:
        p.error('--start-id must be in 1..99999')
    if os.path.exists(os.path.join(args.out, 'annotations.json')):
        p.error('%s already has annotations.json; use a new --out' % args.out)
    os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    if args.overlays:
        os.makedirs(os.path.join(args.out, 'overlays'), exist_ok=True)

    bodies = body_list(args)
    jobs = []
    for name, pose, betas in bodies:
        for vw in range(args.views):
            jobs.append([None, name, pose, betas, vw, args.views])
    if args.max_images:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(jobs))[:args.max_images]
        jobs = [jobs[i] for i in sorted(order)]
    for i, j in enumerate(jobs):
        j[0] = args.start_id + i
    if jobs and jobs[-1][0] >= 100000:
        p.error('ids would exceed 99999; lower --start-id or the number of images')
    print('bodies %d, views %d, images %d, ids %d-%d, workers %d'
          % (len(bodies), args.views, len(jobs), jobs[0][0], jobs[-1][0], args.workers), flush=True)

    t0 = time.time()
    results = []
    with Pool(args.workers, initializer=init_worker, initargs=(args,)) as pool:
        for k, r in enumerate(pool.imap(make_one, [tuple(j) for j in jobs], chunksize=4)):
            results.append(r)
            if (k + 1) % 100 == 0:
                print('  %d/%d  %.1f img/s' % (k + 1, len(jobs), (k + 1) / (time.time() - t0)), flush=True)
    ok = [r for r in results if r is not None]
    skipped = len(results) - len(ok)
    with open(os.path.join(args.out, 'annotations.json'), 'w') as fh:
        json.dump({'images': [r[0] for r in ok], 'annotations': [r[1] for r in ok],
                   'categories': [{'supercategory': 'person', 'id': 1, 'name': 'person',
                                   'keypoints': COCO_NAMES, 'skeleton': SKELETON}]}, fh)
    with open(os.path.join(args.out, 'meta.jsonl'), 'w') as fh:
        for r in ok:
            fh.write(json.dumps(r[2]) + '\n')
    with open(os.path.join(args.out, 'args.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=1)
    vis = np.array([a['keypoints'][2::3] for _, a, _ in ok])
    print('done: %d images (%d skipped: could not fit in frame) in %.0fs; keypoints visible %.1f%%, '
          'occluded %.1f%%, out of frame %.1f%%'
          % (len(ok), skipped, time.time() - t0, 100 * (vis == 2).mean(), 100 * (vis == 1).mean(),
             100 * (vis == 0).mean()))


if __name__ == '__main__':
    main()
