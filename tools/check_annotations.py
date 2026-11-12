"""Sanity-check a SyRIP-style data root before training.

For annotations/person_keypoints_train_infant.json under <root>:
  - every image file exists and its pixel size equals the json width/height
  - every labelled keypoint lies inside the image
  - every image has exactly one annotation
Exits non-zero on any failure. Reads only image headers (PIL), so it is fast.

    python tools/check_annotations.py data/syrip_fix/
"""
import json
import os
import sys

from PIL import Image


def main(root):
    ann = os.path.join(root, 'annotations', 'person_keypoints_train_infant.json')
    d = json.load(open(ann))
    by_img = {}
    for a in d['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)
    fails = []
    n_syn = 0
    for im in d['images']:
        n_syn += bool(im.get('is_synthetic'))
        p = os.path.join(root, 'images', 'train_infant', im['file_name'])
        if not os.path.exists(p):
            fails.append('missing file ' + im['file_name'])
            continue
        w, h = Image.open(p).size
        if (w, h) != (im['width'], im['height']):
            fails.append('%s: file %dx%d, json %dx%d' % (im['file_name'], w, h, im['width'], im['height']))
        anns = by_img.get(im['id'], [])
        if len(anns) != 1:
            fails.append('%s: %d annotations' % (im['file_name'], len(anns)))
            continue
        k = anns[0]['keypoints']
        for j in range(len(k) // 3):
            x, y, v = k[3 * j:3 * j + 3]
            if v > 0 and not (-1 <= x <= w + 1 and -1 <= y <= h + 1):
                fails.append('%s: keypoint %d (%.0f,%.0f) outside %dx%d' % (im['file_name'], j, x, y, w, h))
                break
    print('check_annotations: %s: %d images (%d synthetic), %d problems'
          % (ann, len(d['images']), n_syn, len(fails)))
    for f in fails[:20]:
        print('   ', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main(sys.argv[1])
