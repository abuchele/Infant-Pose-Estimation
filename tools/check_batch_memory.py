"""Measure peak GPU memory for the training steps at a given batch size, on random input.

Mimics tools/train_adaptive_model_hrnet.py:
  - pose step: first 109 parameter tensors frozen, forward + MSE-style backward at --batch
  - D pre-stage: pose forward WITH grad (as the training script does), D on detached
    features, D backward, at --pre-batch
Exits non-zero (with the OOM message) if either step does not fit.

    python tools/check_batch_memory.py --cfg <yaml> --batch 64 --pre-batch 128
"""
import argparse
import sys

import torch

import _init_paths  # noqa: F401
from config import cfg, update_config
import models


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--pre-batch', type=int, default=0)
    a = p.parse_args()
    update_config(cfg, argparse.Namespace(
        cfg=a.cfg, opts=[], modelDir='', logDir='', dataDir='', prevModelDir=''))

    torch.backends.cudnn.benchmark = True
    model_p, model_d = eval('models.' + cfg.MODEL.NAME + '.get_adaptive_pose_net')(cfg, is_train=True)
    model_p, model_d = model_p.cuda().train(), model_d.cuda().train()
    for i, prm in enumerate(model_p.parameters()):
        if i <= 108:
            prm.requires_grad = False
    w, h = cfg.MODEL.IMAGE_SIZE
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print('device %s, total %.1f GiB' % (torch.cuda.get_device_name(0), total))
    ok = True

    try:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(a.batch, 3, h, w, device='cuda')
        feat, out = model_p(x)
        loss = out.pow(2).mean() + 0.0005 * model_d(feat).mean()
        loss.backward()
        print('pose step  batch %d: peak %.1f GiB' % (a.batch, peak_gb()))
    except torch.cuda.OutOfMemoryError as e:
        print('pose step  batch %d: OOM (%s)' % (a.batch, str(e).splitlines()[0]))
        ok = False
    finally:
        x = feat = out = loss = None
        model_p.zero_grad(set_to_none=True)
        model_d.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    if a.pre_batch:
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(a.pre_batch, 3, h, w, device='cuda')
            feat, out = model_p(x)
            model_d(feat.detach()).mean().backward()
            print('D pre-stage batch %d: peak %.1f GiB' % (a.pre_batch, peak_gb()))
        except torch.cuda.OutOfMemoryError as e:
            print('D pre-stage batch %d: OOM (%s)' % (a.pre_batch, str(e).splitlines()[0]))
            ok = False

    print('check_batch_memory: %s' % ('OK' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
