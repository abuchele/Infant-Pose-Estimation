"""Build a training data root = base SyRIP root + generated synthetic sets.

Output layout (what lib/dataset/syrip.py expects), built from symlinks plus one new json:
  <out>/annotations/person_keypoints_train_infant.json      base train + generated (new file)
  <out>/annotations/person_keypoints_{validate,train_pre}_infant.json -> base
  <out>/images/{validate_infant,train_pre_infant}           -> base
  <out>/images/train_infant/<file>                          -> base or generated image (per file)

    python tools/make_syrip_aug.py --base data/syrip_fix --gen syn_generation/render/outputs/<run> \
        [--gen ...] [--max-per-gen N] --out data/syrip_aug_<name>
Refuses id or file-name collisions. Paths in symlinks are absolute.
"""
import argparse
import json
import os
import random
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--gen', action='append', required=True, help='generate_synthetic.py --out dir')
    p.add_argument('--max-per-gen', type=int, default=0, help='random subset of each set (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    if os.path.exists(a.out):
        sys.exit('%s exists' % a.out)
    base = os.path.abspath(a.base)
    bj = json.load(open(os.path.join(base, 'annotations', 'person_keypoints_train_infant.json')))
    images, anns = list(bj['images']), list(bj['annotations'])
    ids = {i['id'] for i in images}
    files = {i['file_name']: os.path.join(base, 'images', 'train_infant', i['file_name']) for i in images}
    n_base = len(images)
    added = []
    for g in a.gen:
        g = os.path.abspath(g)
        gj = json.load(open(os.path.join(g, 'annotations.json')))
        gi = gj['images']
        if a.max_per_gen and len(gi) > a.max_per_gen:
            gi = random.Random(a.seed).sample(gi, a.max_per_gen)
        keep = {i['id'] for i in gi}
        for i in gi:
            if i['id'] in ids or i['file_name'] in files:
                sys.exit('collision: id %s / %s from %s' % (i['id'], i['file_name'], g))
            ids.add(i['id'])
            files[i['file_name']] = os.path.join(g, 'images', i['file_name'])
        images += gi
        anns += [x for x in gj['annotations'] if x['image_id'] in keep]
        added.append((g, len(gi)))

    os.makedirs(os.path.join(a.out, 'annotations'))
    os.makedirs(os.path.join(a.out, 'images', 'train_infant'))
    for s in ('validate', 'train_pre'):
        os.symlink(os.path.realpath(os.path.join(base, 'annotations', 'person_keypoints_%s_infant.json' % s)),
                   os.path.join(a.out, 'annotations', 'person_keypoints_%s_infant.json' % s))
        os.symlink(os.path.realpath(os.path.join(base, 'images', '%s_infant' % s)),
                   os.path.join(a.out, 'images', '%s_infant' % s))
    for fn, src in files.items():
        if not os.path.exists(src):
            sys.exit('missing image ' + src)
        os.symlink(os.path.realpath(src), os.path.join(a.out, 'images', 'train_infant', fn))
    out = dict(bj)
    out['images'], out['annotations'] = images, anns
    with open(os.path.join(a.out, 'annotations', 'person_keypoints_train_infant.json'), 'w') as fh:
        json.dump(out, fh)
    n_syn = sum(1 for i in images if i.get('is_synthetic'))
    print('base %d images; added %s; total %d (%d synthetic, %d real) -> %s'
          % (n_base, ', '.join('%d from %s' % (n, os.path.basename(g)) for g, n in added),
             len(images), n_syn, len(images) - n_syn, a.out))


if __name__ == '__main__':
    main()
