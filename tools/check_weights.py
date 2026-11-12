"""Report which keys of a checkpoint do / do not match the adaptive pose model.

tools/test_adaptive_model.py loads weights with strict=False, which silently
ignores missing and unexpected keys. This prints them.
"""
from __future__ import absolute_import, division, print_function

import argparse

import torch

import _init_paths
from config import cfg, update_config
import models


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--weights', required=True)
    a = p.parse_args()

    update_config(cfg, argparse.Namespace(
        cfg=a.cfg, opts=[], modelDir='', logDir='', dataDir='', prevModelDir=''))

    model_p, _ = eval('models.' + cfg.MODEL.NAME + '.get_adaptive_pose_net')(
        cfg, is_train=False)
    ckpt = torch.load(a.weights, map_location='cpu')
    print('checkpoint top-level type:', type(ckpt).__name__,
          '| #entries:', len(ckpt))
    res = model_p.load_state_dict(ckpt, strict=False)

    n_model = len(model_p.state_dict())
    print('model keys: %d | checkpoint keys: %d' % (n_model, len(ckpt)))
    print('missing keys (in model, not in checkpoint): %d' % len(res.missing_keys))
    for k in res.missing_keys[:30]:
        print('   missing   ', k)
    print('unexpected keys (in checkpoint, not in model): %d' % len(res.unexpected_keys))
    for k in res.unexpected_keys[:30]:
        print('   unexpected', k)
    print('loaded keys: %d / %d model keys' % (n_model - len(res.missing_keys), n_model))


if __name__ == '__main__':
    main()
