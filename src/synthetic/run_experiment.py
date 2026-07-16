"""marginal screening / lasso / cv_r2_lasso の 3 手法を
iid・corr データで実行し、p 値と計算時間を CSV に出力する。

各反復で
  1. データ生成 (null: signal=0)
  2. 観測データで一度アルゴリズムを実行して eta を取得
  3. si.inference で p 値を計算 (この区間が計算時間)
を行い、結果を 1 行として蓄積する。

Windows のマルチプロセスに対応するため worker はトップレベル関数とし、
siarray をプロセス間で受け渡さず各 worker 内でデータを生成する。

使い方:
  python src/synthetic/run_experiment.py --iter 200 --workers 1
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# numpy/BLAS より先に import して各プロセスを 1 スレッドに固定する
# (他人のコアを奪わないため)。import 時に set_env が走る。
from src.common.environment import set_env, set_seed

import time
import argparse
import traceback
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

import autosi as si
from src.synthetic.data import generate_data
from src.common.methods import marginal_screening, lasso, cv_r2_lasso

METHODS = ["marginal_screening", "lasso", "cv_r2_lasso"]
DATA_TYPES = ["iid", "corr"]


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


def _data_split_p_value(method, X_np, y_np, Sigma, cfg):
    """データ分割法 (比較手法): 前半で特徴選択し、後半で古典的 z 検定を行う。

    他手法と同じ n 行を半分ずつ選択と推論に使うため、データ量はフェア。
    推論半分は選択と独立なので通常の z 検定で valid になる
    (corr ノイズでは分割間に相関が残るため厳密には近似)。

    Returns
    -------
    (p_value, selected_index)
    """
    n = X_np.shape[0]
    n1 = n // 2
    X1_np, X2_np = X_np[:n1], X_np[n1:]
    y1_np, y2_np = y_np[:n1], y_np[n1:]

    # 前半: 他手法と同じアルゴリズムを 1 回だけ実行して特徴を選ぶ
    X1 = si.siarray(X1_np)
    algorithm = _build_algorithm(method, X1, cfg)
    eta1 = algorithm(si.siarray(y1_np))
    selected = int(np.argmax(eta1.data @ X1_np))

    # 後半: 選択に使っていないデータで OLS 係数の z 検定
    eta2 = np.linalg.solve(X2_np.T @ X2_np, X2_np.T)[selected]
    stat = float(eta2 @ y2_np)
    if np.isscalar(Sigma):
        var_stat = float(Sigma) * float(eta2 @ eta2)
    else:
        var_stat = float(eta2 @ Sigma[n1:, n1:] @ eta2)
    p_value = float(2.0 * norm.sf(abs(stat) / np.sqrt(var_stat)))
    return p_value, selected


def run_single(task):
    """1 試行: データ生成 -> eta 取得 -> p 値計算 (時間計測)。"""
    method, data_type, seed, cfg = task
    # 各ワーカープロセスを 1 スレッドに固定し (他コアを奪わない)、シードを揃える
    set_env(parallel=True)
    set_seed(seed)
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

        # 観測データで実際に選択された特徴量を復元する。
        # eta は (XᵀX)⁻¹Xᵀ の選択行なので eta·X ≈ e_idx (選択位置だけ 1) になる。
        selected_index = int(np.argmax(eta.data @ X_np))
        # 真の活性特徴量は先頭 n_signal 個 (signal!=0 のときのみ、null では活性なし)。
        # is_signal=False の棄却は Type I error なので power 集計から除外できる。
        n_active = min(cfg["n_signal"], cfg["n_features"]) if cfg["signal"] != 0.0 else 0
        is_signal = bool(selected_index < n_active)

        oc_p = si.inference(eta=eta, prob_vec=y, var=Sigma, algorithm=algorithm, inference_mode="over_conditioning").p_value
        start = time.time()
        result = si.inference(eta=eta, prob_vec=y, var=Sigma, algorithm=algorithm)
        elapsed = time.time() - start

        naive_p = result.naive_p_value()
        # sicore の bonferroni_p_value は「比較数の対数」を受け取る。
        # 比較数は取り得る選択パターンの総数 2^p とし、log(2^p) = p*log(2) を渡す。
        bonferroni_p = result.bonferroni_p_value(
            log_num_comparisons=cfg["n_features"] * np.log(2)
        )

        # データ分割法 (比較手法)。前半だけで選ぶため選択結果は proposed と
        # 異なり得るので、power の条件付け用に is_signal も別途記録する。
        ds_p, ds_selected = _data_split_p_value(
            method, X_np, y_np, Sigma, cfg_local
        )
        ds_is_signal = bool(ds_selected < n_active)

        return {
            "method": method,
            "data_type": data_type,
            "seed": seed,
            "p_value": float(result.p_value),
            "oc_p_value": float(oc_p),
            "naive_p_value": float(naive_p),
            "bonferroni_p_value": float(bonferroni_p),
            "time": elapsed,
            "n_samples": cfg["n_samples"],
            "n_features": cfg["n_features"],
            "signal": cfg["signal"],
            "n_signal": cfg["n_signal"],
            # power を「実際に信号が載った特徴量が選ばれた試行」のみで測るための情報
            "selected_index": selected_index,
            "is_signal": is_signal,
            # データ分割法の結果 (選択は前半のみで行うため別に記録)
            "ds_p_value": ds_p,
            "ds_selected_index": ds_selected,
            "ds_is_signal": ds_is_signal,
        }
    except Exception:
        # 1 試行の失敗 (数値誤差など) で全体を止めない。ログを残して次へ進む。
        print(f"[skip] method={method} data_type={data_type} seed={seed} "
              "が失敗 (次の試行へ続行)", file=sys.stderr)
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
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="全 seed に加える定数。手法ごとに変えると "
                             "手法間で異なるデータになり独立性が保てる")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_features", type=int, default=15,
                        help="特徴量数 p")
    parser.add_argument("--signal", type=float, default=0.0,
                        help="真の係数の傾き Δ (0=null/FPR, >0=power)")
    parser.add_argument("--n_signal", type=int, default=5,
                        help="信号を載せる先頭特徴量の数")
    parser.add_argument("--out", type=str, default="result/experiment_results.csv")
    # lasso / cv 共通ハイパラ
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Lasso 正則化係数 (sklearn と同じ "
                             "(1/(2n))RSS + alpha*|beta|_1 規約)")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    # cv_r2_lasso 固有
    parser.add_argument("--K", type=int, default=3)
    # λ グリッドは logspace(lam_lo, lam_hi, n_lambda)。λ < 1e-2 はほぼ
    # 無正則化で計算コストだけ食うため [1e-2, 1] の 5 点に絞る。
    parser.add_argument("--n_lambda", type=int, default=5)
    parser.add_argument("--lam_lo", type=float, default=-2.0)
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

    # 条件 (手法 × データ種別 × seed) のタスクを列挙。
    # seed_offset を足すことで手法ごとに異なるデータを使える (独立性確保)。
    tasks = [
        (method, data_type, seed + args.seed_offset, cfg)
        for method in methods
        for data_type in data_types
        for seed in range(seed_start, seed_end)
    ]
    print(f"methods={methods} data_types={data_types} "
          f"seeds=[{seed_start},{seed_end})+{args.seed_offset} "
          f"-> {len(tasks)} tasks")

    records = []
    n_failed = 0
    if args.workers <= 1:
        for task in tqdm(tasks, desc="experiments"):
            rec = run_single(task)
            if rec is not None:
                records.append(rec)
            else:
                n_failed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_single, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(tasks),
                            desc="experiments"):
                # ワーカープロセス自体の異常終了でも全体を止めず次へ進む
                try:
                    rec = fut.result()
                except Exception:
                    traceback.print_exc()
                    rec = None
                if rec is not None:
                    records.append(rec)
                else:
                    n_failed += 1

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    failed_note = f" (失敗 {n_failed}/{len(tasks)} 件はスキップ)" if n_failed else ""
    print(f"\nsaved {len(df)} rows -> {args.out}{failed_note}")

    # 条件ごとの簡易サマリを表示
    if not df.empty:
        summary = df.groupby(["method", "data_type"]).agg(
            n=("p_value", "size"),
            mean_p=("p_value", "mean"),
            fpr_005=("p_value", lambda s: float((s < 0.05).mean())),
            ds_005=("ds_p_value", lambda s: float((s < 0.05).mean())),
            # power 実験で「信号特徴量が選ばれた試行」の割合 (null では常に 0)
            signal_rate=("is_signal", "mean"),
            mean_time=("time", "mean"),
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
