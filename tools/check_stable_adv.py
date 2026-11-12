"""Preflight for lib/core/function_stable.py: checks the adversarial-term logic on
small random tensors with the real HRNet DomainClassifier. Exits non-zero on failure.

    python tools/check_stable_adv.py
"""
import math
import sys

import torch

import _init_paths  # noqa: F401
from config import cfg
from core.function_stable import adversarial_term, set_requires_grad
from models.adaptive_pose_hrnet import DomainClassifier


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    crit = torch.nn.BCEWithLogitsLoss()
    label = torch.tensor([[0.], [1.], [0.], [1.]], device=dev)
    lam = 0.0005
    fails = []

    def run(adv_loss, lock_d):
        c = cfg.clone()
        c.defrost()
        c.TRAIN.LAMBDA = lam
        c.TRAIN.ADV_LOSS = adv_loss
        c.TRAIN.LOCK_D = lock_d
        d = DomainClassifier(None).to(dev).train()
        f = torch.randn(4, 48, 96, 72, device=dev, requires_grad=True)
        if lock_d:
            set_requires_grad(d, False)
        logits = d(f)
        term, bce_true = adversarial_term(c, crit, logits, label)
        term.backward()
        d_grads = [p.grad for p in d.parameters()]
        if lock_d:
            set_requires_grad(d, True)
        restored = all(p.requires_grad for p in d.parameters())
        return term.item(), bce_true.item(), logits.detach(), f.grad, d_grads, restored

    # 1. neg_bce reproduces the original term
    term, bce, _, fg, dg, _ = run('neg_bce', False)
    if not math.isclose(term, -lam * bce, rel_tol=1e-5):
        fails.append('neg_bce term {} != -lambda*bce {}'.format(term, -lam * bce))
    if not any(g is not None and g.abs().sum() > 0 for g in dg):
        fails.append('LOCK_D False: expected gradient in D (original leak), got none')

    # 2. confusion term is bounded below by lambda*ln2
    term, _, logits, fg, dg, restored = run('confusion', True)
    if term < lam * math.log(2) - 1e-9:
        fails.append('confusion term {} below lambda*ln2 {}'.format(term, lam * math.log(2)))

    # 3. LOCK_D: no gradient reaches D, features still get one, flags restored
    if any(g is not None for g in dg):
        fails.append('LOCK_D True: D parameters received gradient')
    if fg is None or fg.abs().sum() == 0:
        fails.append('LOCK_D True: features received no gradient')
    if not restored:
        fails.append('LOCK_D True: requires_grad not restored on D')

    # 4. confusion gradient vanishes when D outputs exactly 0.5 (logit 0)
    z = torch.zeros(4, 1, device=dev, requires_grad=True)
    c = cfg.clone(); c.defrost(); c.TRAIN.LAMBDA = lam; c.TRAIN.ADV_LOSS = 'confusion'
    t, _ = adversarial_term(c, crit, z, label)
    t.backward()
    if z.grad.abs().max() > 1e-8:
        fails.append('confusion gradient at logit 0 is {}, expected 0'.format(z.grad.abs().max().item()))
    if not math.isclose(t.item(), lam * math.log(2), rel_tol=1e-5):
        fails.append('confusion term at logit 0 is {}, expected lambda*ln2'.format(t.item()))

    if fails:
        print('check_stable_adv: FAIL')
        for f in fails:
            print('  -', f)
        sys.exit(1)
    print('check_stable_adv: OK (device {}; neg_bce matches original; confusion bounded, '
          'zero gradient at D=0.5; LOCK_D blocks D gradient and restores flags)'.format(dev))


if __name__ == '__main__':
    main()
