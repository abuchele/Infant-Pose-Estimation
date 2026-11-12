"""Preflight for adaptive_pose_cnn: build the model with ImageNet weights, run one
forward pass on CPU and check the feature / heatmap shapes and frozen tensors."""
from __future__ import absolute_import, division, print_function

import argparse

import torch

import _init_paths
from config import cfg, update_config
import models


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    a = p.parse_args()
    update_config(cfg, argparse.Namespace(
        cfg=a.cfg, opts=[], modelDir='', logDir='', dataDir='', prevModelDir=''))

    model_p, model_d = models.adaptive_pose_cnn.get_adaptive_pose_net(cfg, is_train=True)
    model_p.eval()
    model_d.eval()
    w, h = cfg.MODEL.IMAGE_SIZE
    with torch.no_grad():
        feat, hm = model_p(torch.zeros(2, 3, h, w))
        logits = model_d(feat)
    hw, hh = cfg.MODEL.HEATMAP_SIZE
    print('backbone:', cfg.MODEL.EXTRA.BACKBONE)
    print('feature:', tuple(feat.shape), '| heatmaps:', tuple(hm.shape),
          '| domain logits:', tuple(logits.shape))
    assert tuple(feat.shape) == (2, 48, hh, hw)
    assert tuple(hm.shape) == (2, cfg.MODEL.NUM_JOINTS, hh, hw)
    names = [n for n, _ in model_p.named_parameters()]
    frozen = [n for n in names if n.startswith(model_p.frozen_prefixes)]
    print('parameter tensors: %d | frozen: %d (%s ... %s)'
          % (len(names), len(frozen), frozen[0], frozen[-1]))
    print('parameters: %.1fM' % (sum(t.numel() for t in model_p.parameters()) / 1e6))
    assert frozen
    print('PREFLIGHT OK')


if __name__ == '__main__':
    main()
