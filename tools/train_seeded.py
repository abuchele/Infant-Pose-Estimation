"""Run a training script with all RNGs seeded first.

Usage:
    python tools/train_seeded.py --seed N tools/train_adaptive_model_hrnet.py --cfg ... [opts]

The original training scripts never set a seed, so the domain-classifier init,
data shuffling and augmentation differ uncontrollably between runs. This wrapper
seeds Python, NumPy and torch (CPU + CUDA), then executes the target script
unchanged as __main__. DataLoader workers (torch >= 1.9) derive their torch and
NumPy seeds from the main-process torch RNG, so augmentation is seeded too.

Not bit-reproducible: cfg CUDNN.BENCHMARK true / DETERMINISTIC false is kept
(same as all earlier runs), so cuDNN kernel choice still adds GPU nondeterminism.
"""
import random
import runpy
import sys

import numpy as np
import torch


def main():
    argv = sys.argv[1:]
    stable = False
    if argv and argv[0] == '--stable-adv':
        stable = True
        argv = argv[1:]
    if len(argv) < 3 or argv[0] != '--seed':
        sys.exit('usage: train_seeded.py [--stable-adv] --seed N <script.py> [script args]')
    seed = int(argv[1])
    script = argv[2]

    if stable:
        # Swap in lib/core/function_stable.train_adaptive (ADV_LOSS / LOCK_D options)
        # before the training script runs `from core.function import train_adaptive`.
        import _init_paths  # noqa: F401  (tools/ is sys.path[0]; adds lib/)
        import core.function
        import core.function_stable
        core.function.train_adaptive = core.function_stable.train_adaptive
        print('train_seeded: using core.function_stable.train_adaptive', flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('train_seeded: seed={} script={}'.format(seed, script), flush=True)

    sys.argv = [script] + argv[3:]
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
