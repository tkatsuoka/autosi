"""乱数シードと並列スレッド数の設定。

BLAS (OpenBLAS/MKL 等) は既定で複数スレッドを掴むため、ProcessPoolExecutor
で多数プロセスを立てると 1 プロセスが複数コアを占有し、計算ノードの他人の
CPU まで食い尽くしてしまう。各プロセスを 1 スレッドに固定して防ぐ。

重要: スレッド数の環境変数は numpy/BLAS が import される「前」に設定する
必要がある。run_experiment.py の冒頭で import より先に set_env() 相当を
呼ぶこと (本モジュールも import 時に set_env を実行する)。
"""

import os


def set_env(parallel: bool = True) -> None:
    """BLAS 等のスレッド数を 1 に固定する (他コアを奪わないため)。"""
    if parallel:
        # joblib をプロセス分離で使う場合に備える (本実験では未使用だが無害)
        os.environ["JOBLIB_START_METHOD"] = "spawn"
    else:
        os.environ["MAX_JOBS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# import 時点で確実にスレッド制限を効かせる (numpy より先に読まれる前提)
set_env(parallel=True)


def set_seed(seed: int) -> None:
    """再現性のため各種乱数シードを固定する。"""
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 注: 実験のデータ生成は np.random.default_rng(seed) を使うため、
    #     これは built-in random / hash の固定が主目的。
