"""marginal screening / lasso / cv_r2_lasso の 3 手法を
iid・correlated データで実行し、p 値と計算時間を CSV に出力する。

各反復で
  1. データ生成 (null: signal=0)
  2. 観測データで一度アルゴリズムを実行して eta を取得
  3. si.inference で p 値を計算 (この区間が計算時間)
を行い、結果を 1 行として蓄積する。

Windows のマルチプロセスに対応するため worker はトップレベル関数とし、
siarray をプロセス間で受け渡さず各 worker 内でデータを生成する。

使い方:
  python src/run_experiment.py --iter 200 --workers 1
"""

import os

# 並列実行時に各プロセスが BLAS スレッドを増やさないよう先に固定する。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import argparse
import traceback
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autosi as si
from src.data import generate_data
from src.methods import marginal_screening, lasso, cv_r2_lasso

METHODS = ["marginal_screening", "lasso", "cv_r2_lasso"]
DATA_TYPES = ["iid", "correlated"]


def _build_algorithm(method, X, cfg):
    """method 名に応じて y のみを引数に取るアルゴリズム関数を構築する。"""
    n, p = X.shape
    if method == "marginal_screening":
        return partial(marginal_screening, X=X)
    if method == "lasso":
        return partial(
            lasso, X=X, n_samples=n, n_features=p,
            alpha=cfg["alpha"], max_iter=cfg["max_iter"], tol=cfg["tol"],
        )
    if method == "cv_r2_lasso":
        # fold は反復ごとに固定 (CV の分割を eta 計算と inference で一致させる)
        rng = np.random.default_rng(cfg["fold_seed"])
        fold_indices = np.array_split(rng.permutation(n), cfg["K"])
        lambda_grid = np.logspace(cfg["lam_lo"], cfg["lam_hi"], cfg["n_lambda"])
        return partial(
            cv_r2_lasso, X=X, n_samples=n, n_features=p,
            fold_indices=fold_indices, lambda_grid=lambda_grid,
            K=cfg["K"], max_iter=cfg["max_iter"], tol=cfg["tol"],
        )
    raise ValueError(f"unknown method: {method}")


def run_single(task):
    """1 試行: データ生成 -> eta 取得 -> p 値計算 (時間計測)。"""
    method, data_type, seed, cfg = task
    try:
        X_np, y_np, Sigma = generate_data(
            data_type=data_type,
            n_samples=cfg["n_samples"],
            n_features=cfg["n_features"],
            seed=seed,
            signal=cfg["signal"],
            n_signal=cfg["n_signal"],
        )
        X = si.siarray(X_np)
        y = si.siarray(y_np)

        cfg_local = dict(cfg, fold_seed=seed)
        algorithm = _build_algorithm(method, X, cfg_local)

        eta = algorithm(y)
        start = time.time()
        result = si.inference(eta=eta, prob_vec=y, var=Sigma, algorithm=algorithm)
        elapsed = time.time() - start

        return {
            "method": method,
            "data_type": data_type,
            "seed": seed,
            "p_value": float(result.p_value),
            "time": elapsed,
            "n_samples": cfg["n_samples"],
            "n_features": cfg["n_features"],
            "signal": cfg["signal"],
            "n_signal": cfg["n_signal"],
        }
    except Exception:
        # 1 試行の失敗で全体を止めないようログだけ残して None を返す。
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=200, help="各条件の反復回数")
    parser.add_argument("--workers", type=int, default=1, help="並列ワーカー数")
    # ジョブ分割用 (qsub などで一部条件だけ実行したいとき)
    parser.add_argument("--methods", type=str, default=",".join(METHODS),
                        help="カンマ区切りの手法名 (既定: 全手法)")
    parser.add_argument("--data_types", type=str, default=",".join(DATA_TYPES),
                        help="カンマ区切りのデータ種別 (既定: 全種別)")
    parser.add_argument("--seed_start", type=int, default=None,
                        help="seed 範囲の開始 (既定: 0)")
    parser.add_argument("--seed_end", type=int, default=None,
                        help="seed 範囲の終了 (排他的, 既定: --iter)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_features", type=int, default=5)
    parser.add_argument("--signal", type=float, default=0.0,
                        help="真の係数の傾き Δ (0=null/FPR, >0=power)")
    parser.add_argument("--n_signal", type=int, default=3,
                        help="信号を載せる先頭特徴量の数")
    parser.add_argument("--out", type=str, default="result/experiment_results.csv")
    # lasso / cv 共通ハイパラ
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-4)
    # cv_r2_lasso 固有
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_lambda", type=int, default=5)
    parser.add_argument("--lam_lo", type=float, default=-4.0)
    parser.add_argument("--lam_hi", type=float, default=0.0)
    args = parser.parse_args()

    cfg = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "signal": args.signal,
        "n_signal": args.n_signal,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "K": args.K,
        "n_lambda": args.n_lambda,
        "lam_lo": args.lam_lo,
        "lam_hi": args.lam_hi,
    }

    # 実行対象の条件を決定 (ジョブ分割に対応)
    methods = [m for m in args.methods.split(",") if m]
    data_types = [d for d in args.data_types.split(",") if d]
    seed_start = args.seed_start if args.seed_start is not None else 0
    seed_end = args.seed_end if args.seed_end is not None else args.iter

    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method: {m} (choices: {METHODS})")
    for d in data_types:
        if d not in DATA_TYPES:
            raise ValueError(f"unknown data_type: {d} (choices: {DATA_TYPES})")

    # 条件 (手法 × データ種別 × seed) のタスクを列挙
    tasks = [
        (method, data_type, seed, cfg)
        for method in methods
        for data_type in data_types
        for seed in range(seed_start, seed_end)
    ]
    print(f"methods={methods} data_types={data_types} "
          f"seeds=[{seed_start},{seed_end}) -> {len(tasks)} tasks")

    records = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="experiments"):
            rec = run_single(task)
            if rec is not None:
                records.append(rec)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_single, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(tasks),
                            desc="experiments"):
                rec = fut.result()
                if rec is not None:
                    records.append(rec)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved {len(df)} rows -> {args.out}")

    # 条件ごとの簡易サマリを表示
    if not df.empty:
        summary = df.groupby(["method", "data_type"]).agg(
            n=("p_value", "size"),
            mean_p=("p_value", "mean"),
            fpr_005=("p_value", lambda s: float((s < 0.05).mean())),
            mean_time=("time", "mean"),
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
