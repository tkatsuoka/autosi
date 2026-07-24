"""実データ実験: UCI データセットのサブサンプリングで選択的推論を評価する。

si4pipeline 論文の実データ実験に準拠し、各 (データセット, サンプル数 n) で
  1. 元データから n 行を非復元でサブサンプリング
  2. X を列標準化 (定数列は除外)、y も標準化。分散 sigma^2 は OLS 残差
     から推定 (自由度 n - p - 1、切片相当の 1 自由度を差し引く)
  3. アルゴリズム (y の中心化 + 特徴選択) を実行して eta を取得
  4. si.inference で選択的 p 値を計算 (この区間が計算時間)
を繰り返す。実データは真の信号が未知のため、
power は「棄却率 P(p 値 < α)」として集計する (proposed vs oc の比較)。

切片の扱い: 本実装の Lasso は切片を学習しないため、X の列中心化に加えて
y の中心化を「アルゴリズム内で」行う (切片を無罰則で入れるのと等価)。
中心化は y の線形変換なので autosi がそのまま追跡でき、valid な推論になる。

使い方:
  uv run --extra experiments python src/realdata/datasets.py                      # 事前に一度だけ
  uv run --extra experiments python src/realdata/run_realdata.py --iter 10 --datasets wine_red
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# numpy/BLAS より先に import して各プロセスを 1 スレッドに固定する
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
from src.common import methods as methods_lib
from src.common.methods import marginal_screening, lasso, cv_r2_lasso
from src.realdata.datasets import DATASET_NAMES, load_dataset

METHODS = ["marginal_screening", "lasso", "cv_r2_lasso"]


def _subsample(dataset, n_samples, seed):
    """データセットから n 行をサブサンプリングし、標準化して返す。"""
    X_full, y_full = load_dataset(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_full), size=n_samples, replace=False)
    X, y = X_full[idx], y_full[idx]

    # 列標準化。サブサンプル内で定数になった列は除外する (XtX の特異化防止)
    std = X.std(axis=0)
    keep = std > 1e-12
    X = (X[:, keep] - X[:, keep].mean(axis=0)) / std[keep]

    # y も標準化する (実データ実験の慣例)。生スケールのままだと係数の
    # スケールに対して収束判定 tol が厳しすぎて座標降下が回りきらず、
    # 選択イベントの追跡区間が爆発する。H0: eta^T mu = 0 の検定は y の
    # 正のスケーリングに対して不変なので、棄却率の意味は変わらない。
    y = (y - y.mean()) / y.std()
    return X, y


def _estimate_sigma2(X, y):
    """OLS 残差から分散を推定する (論文 Appendix E.1 と同式、切片分を控除)。"""
    n, p = X.shape
    y_c = y - y.mean()
    beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
    resid = y_c - X @ beta
    return float(resid @ resid / (n - p - 1))


def _build_algorithm(method, X, n, p, seed, cfg):
    """y の中心化 + 特徴選択を行うアルゴリズム関数を構築する。

    検定対象は選択された特徴集合から乱数 pick_u で 1 つ選ぶ
    (synthetic 側と同じ規則。pick_u は run_single で cfg に注入)。
    """
    if method == "marginal_screening":
        base = partial(marginal_screening, X=X, k=cfg["ms_k"],
                       pick_u=cfg["pick_u"])
    elif method == "lasso":
        base = partial(
            lasso, X=X, n_samples=n, n_features=p,
            alpha=cfg["alpha"], max_iter=cfg["max_iter"], tol=cfg["tol"],
            pick_u=cfg["pick_u"],
        )
    elif method == "cv_r2_lasso":
        rng = np.random.default_rng(seed)
        fold_indices = np.array_split(rng.permutation(n), cfg["K"])
        lambda_grid = np.logspace(cfg["lam_lo"], cfg["lam_hi"], cfg["n_lambda"])
        base = partial(
            cv_r2_lasso, X=X, n_samples=n, n_features=p,
            fold_indices=fold_indices, lambda_grid=lambda_grid,
            K=cfg["K"], max_iter=cfg["max_iter"], tol=cfg["tol"],
            pick_u=cfg["pick_u"],
        )
    else:
        raise ValueError(f"unknown method: {method}")

    def algorithm(y):
        # 切片の代替として y を中心化してから特徴選択する。
        # X は列中心化済みなので eta^T 1 = 0 となり検定統計量は中心化不変。
        return base(y - si.sum(y) / n)

    return algorithm


def _data_split_p_value(method, X_np, y_np, sigma2, seed, cfg):
    """データ分割法 (比較手法): 前半で特徴選択し、後半で古典的 z 検定を行う。

    他手法と同じ n 行を半分ずつ選択と推論に使うため、データ量はフェア。
    サブサンプルの行順はランダムなので先頭 n/2 での分割で問題ない。
    後半は半分内で再中心化してから検定する (切片相当)。分散は他手法と
    同じ全サブサンプルからの推定値 sigma2 を使い、入力条件を揃える。

    Returns
    -------
    (p_value, selected_index)
    """
    n, p = X_np.shape
    n1 = n // 2
    X1_np, X2_np = X_np[:n1], X_np[n1:]
    y1_np, y2_np = y_np[:n1], y_np[n1:]

    # 前半: 他手法と同じアルゴリズム (y 中心化込み) を 1 回実行して特徴を選ぶ
    X1 = si.siarray(X1_np)
    algorithm = _build_algorithm(method, X1, n1, p, seed, cfg)
    methods_lib.last_active_set = None
    eta1 = algorithm(si.siarray(y1_np))
    ds_active = methods_lib.last_active_set
    if ds_active is not None and len(ds_active) == 0:
        raise RuntimeError("データ分割法: 前半で特徴が選ばれなかった")
    selected = int(np.argmax(eta1.data @ X1_np))

    # 後半: 半分内で中心化し、選択に使っていないデータで OLS 係数の z 検定
    X2c = X2_np - X2_np.mean(axis=0)
    y2c = y2_np - y2_np.mean()
    eta2 = np.linalg.solve(X2c.T @ X2c, X2c.T)[selected]
    stat = float(eta2 @ y2c)
    var_stat = sigma2 * float(eta2 @ eta2)
    p_value = float(2.0 * norm.sf(abs(stat) / np.sqrt(var_stat)))
    return p_value, selected


def run_single(task):
    """1 試行: サブサンプリング -> eta 取得 -> p 値計算 (時間計測)。"""
    method, dataset, n_samples, seed, cfg = task
    set_env(parallel=True)
    set_seed(seed)
    try:
        X_np, y_np = _subsample(dataset, n_samples, seed)
        n, p = X_np.shape
        sigma2 = _estimate_sigma2(X_np, y_np)

        X = si.siarray(X_np)
        y = si.siarray(y_np)
        # 検定対象のランダム選択用の乱数 (サブサンプリングの seed と分離)
        pick_u = float(np.random.default_rng([seed, 987654321]).uniform())
        cfg = dict(cfg, pick_u=pick_u)
        algorithm = _build_algorithm(method, X, n, p, seed, cfg)

        # 観測データでのアクティブ集合 (lasso/cv のみ)。前タスクの値が
        # 残らないようリセットし、DS や推論が上書きする前に取り出す
        methods_lib.last_active_set = None
        eta = algorithm(y)
        active_set = methods_lib.last_active_set
        # 特徴が 1 つも選ばれなかった試行は検定対象が存在しないためスキップ
        # (si4pipeline でも空集合は推論エラー = 失敗扱い)
        if active_set is not None and len(active_set) == 0:
            raise RuntimeError("特徴集合が空 (選択された特徴なし)")
        # eta は (XtX)^-1 Xt の選択行なので eta・X = e_idx から選択特徴を復元
        selected_index = int(np.argmax(eta.data @ X_np))

        oc_p = si.inference(eta=eta, prob_vec=y, var=sigma2, algorithm=algorithm,
                            inference_mode="over_conditioning").p_value
        start = time.time()
        result = si.inference(eta=eta, prob_vec=y, var=sigma2, algorithm=algorithm)
        elapsed = time.time() - start

        naive_p = result.naive_p_value()
        # 比較数は選択パターン総数 2^p (log を渡す仕様)
        bonferroni_p = result.bonferroni_p_value(log_num_comparisons=p * np.log(2))

        # データ分割法 (比較手法)
        ds_p, ds_selected = _data_split_p_value(
            method, X_np, y_np, sigma2, seed, cfg
        )

        return {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "p_value": float(result.p_value),
            "oc_p_value": float(oc_p),
            "naive_p_value": float(naive_p),
            "bonferroni_p_value": float(bonferroni_p),
            "time": elapsed,
            "n_samples": n_samples,
            "n_features": p,
            "sigma2": sigma2,
            # どの設定で走った結果かを CSV 単体で検証できるように記録する
            "K": cfg["K"],
            "n_lambda": cfg["n_lambda"],
            "lam_lo": cfg["lam_lo"],
            "lam_hi": cfg["lam_hi"],
            "selected_index": selected_index,
            # lasso/cv のアクティブ集合 (係数非零の特徴、"|" 区切り。ms は空欄)
            "active_set": "" if active_set is None else "|".join(map(str, active_set)),
            "active_size": None if active_set is None else len(active_set),
            # データ分割法の結果 (選択は前半のみで行うため別に記録)
            "ds_p_value": ds_p,
            "ds_selected_index": ds_selected,
        }
    except Exception:
        # 1 試行の失敗 (数値誤差など) で全体を止めない。ログを残して次へ進む。
        print(f"[skip] method={method} dataset={dataset} n={n_samples} "
              f"seed={seed} が失敗 (次の試行へ続行)", file=sys.stderr)
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1000,
                        help="各 (データセット, n) の反復回数")
    parser.add_argument("--workers", type=int, default=1, help="並列ワーカー数")
    parser.add_argument("--methods", type=str, default="lasso",
                        help="カンマ区切りの手法名")
    parser.add_argument("--datasets", type=str, default=",".join(DATASET_NAMES),
                        help="カンマ区切りのデータセット名 (既定: 全8種)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="サブサンプルのサイズ n (論文は 100/150/200)")
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--seed_end", type=int, default=None)
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="全 seed に加える定数 (手法ごとに変えて独立性を保つ)")
    parser.add_argument("--out", type=str, default="result/realdata_results.csv")
    # lasso / cv 共通ハイパラ (synthetic と同じ既定)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Lasso 正則化係数 (sklearn と同じ "
                             "(1/(2n))RSS + alpha*|beta|_1 規約)")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--ms_k", type=int, default=3,
                        help="marginal screening が選ぶ特徴集合の大きさ "
                             "(この中から乱数で 1 つを検定する)")
    # λ グリッドは logspace(lam_lo, lam_hi, n_lambda) = {0.01, 0.032, 0.1}。
    # λ_max 付近の腕は CV に選ばれた際に係数をほぼ潰し切って特徴選択が
    # 縮退するため、予測に有効な [0.01, 0.1] に絞る (synthetic 側と同じ既定。
    # 点数を絞る理由も synthetic 側のコメント参照)。
    parser.add_argument("--n_lambda", type=int, default=3)
    parser.add_argument("--lam_lo", type=float, default=-2.0)
    parser.add_argument("--lam_hi", type=float, default=-1.0)
    args = parser.parse_args()

    cfg = {
        "alpha": args.alpha, "max_iter": args.max_iter, "tol": args.tol,
        "K": args.K, "n_lambda": args.n_lambda,
        "lam_lo": args.lam_lo, "lam_hi": args.lam_hi,
        "ms_k": args.ms_k,
    }

    methods = [m for m in args.methods.split(",") if m]
    datasets = [d for d in args.datasets.split(",") if d]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method: {m} (choices: {METHODS})")
    for d in datasets:
        if d not in DATASET_NAMES:
            raise ValueError(f"unknown dataset: {d} (choices: {DATASET_NAMES})")

    seed_start = args.seed_start if args.seed_start is not None else 0
    seed_end = args.seed_end if args.seed_end is not None else args.iter

    tasks = [
        (method, dataset, args.n_samples, seed + args.seed_offset, cfg)
        for method in methods
        for dataset in datasets
        for seed in range(seed_start, seed_end)
    ]
    print(f"methods={methods} datasets={datasets} n={args.n_samples} "
          f"seeds=[{seed_start},{seed_end})+{args.seed_offset} "
          f"-> {len(tasks)} tasks")

    records = []
    n_failed = 0
    if args.workers <= 1:
        for task in tqdm(tasks, desc="realdata"):
            rec = run_single(task)
            if rec is not None:
                records.append(rec)
            else:
                n_failed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_single, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(tasks),
                            desc="realdata"):
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

    # データセットごとの棄却率 (実データでの power 指標) を表示
    if not df.empty:
        summary = df.groupby(["method", "dataset"]).agg(
            n=("p_value", "size"),
            reject_proposed=("p_value", lambda s: float((s < 0.05).mean())),
            reject_oc=("oc_p_value", lambda s: float((s < 0.05).mean())),
            reject_bonf=("bonferroni_p_value", lambda s: float((s < 0.05).mean())),
            reject_ds=("ds_p_value", lambda s: float((s < 0.05).mean())),
            mean_time=("time", "mean"),
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
