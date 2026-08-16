"""Thread-count and random-seed configuration.

BLAS backends (OpenBLAS/MKL, etc.) grab multiple threads by default, so
running many worker processes via ProcessPoolExecutor would oversubscribe
the CPU. Each process is pinned to a single thread instead.

Important: the thread-count environment variables must be set BEFORE
numpy/BLAS is imported. The runner scripts import this module first;
set_env() is executed at import time.
"""

import os


def set_env(parallel: bool = True) -> None:
    """Pin BLAS and friends to a single thread per process."""
    if parallel:
        # In case joblib is used with process-based backends (harmless otherwise)
        os.environ["JOBLIB_START_METHOD"] = "spawn"
    else:
        os.environ["MAX_JOBS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Apply the thread limits at import time (this module must be imported
# before numpy).
set_env(parallel=True)


def set_seed(seed: int) -> None:
    """Fix the random seeds for reproducibility.

    Note: the experiments generate data through np.random.default_rng(seed),
    so this mainly fixes the built-in random module and hashing.
    """
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
