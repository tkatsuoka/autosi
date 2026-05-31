import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autosi as si


# ============================================================================
# cv_r2_lasso: CV-R² で λ を選択し、Lasso 係数を推定する
#
# グローバル変数に依存しない形で引数化している。
# si.inference から繰り返し呼ばれる際は partial でパラメータをバインドして渡す。
# ============================================================================

def cv_r2_lasso(y, *, X, fold_indices, lambda_grid, K, n_features, n_samples,
                max_iter=1000, tol=1e-4):
    cv_r2 = []
    for lam in lambda_grid:
        fold_r2 = si.array(0.0)
        for k in range(K):
            train_idx = np.concatenate([fold_indices[j] for j in range(K) if j != k])
            val_idx = fold_indices[k]
            n_tr = len(train_idx)

            X_tr, y_tr = X[train_idx], y[train_idx]

            col_sq_mean = si.sum(X_tr ** 2, axis=0) / n_tr

            lv = lam / n_tr
            coef = si.array(np.zeros(n_features))
            for _ in range(max_iter):
                coef_old = coef.copy()
                for j in range(n_features):
                    r_j = y_tr - X_tr @ coef + X_tr[:, j] * coef[j]
                    z_j = X_tr[:, j] @ r_j / n_tr
                    if z_j > lv:
                        coef[j] = (z_j - lv) / col_sq_mean[j]
                    elif z_j < -lv:
                        coef[j] = (z_j + lv) / col_sq_mean[j]
                    else:
                        coef[j] = si.array(0.0)
                if si.sum(si.abs(coef - coef_old)) < tol:
                    break

            y_val = y[val_idx]
            y_pred = X[val_idx] @ coef
            ss_res = si.sum((y_val - y_pred) ** 2)
            ss_tot = si.sum((y_val - si.sum(y_val) / len(val_idx)) ** 2)
            fold_r2 = fold_r2 + 1.0 - ss_res / ss_tot
        cv_r2.append(fold_r2 / K)

    lam_star_idx = si.argmax(si.stack(cv_r2))
    lam_star = lambda_grid[lam_star_idx]

    col_sq_mean = si.sum(X ** 2, axis=0) / n_samples
    lambda_val = lam_star / n_samples
    coef = si.array(np.zeros(n_features))
    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            r_j = y - X @ coef + X[:, j] * coef[j]
            z_j = X[:, j] @ r_j / n_samples
            if z_j > lambda_val:
                coef[j] = (z_j - lambda_val) / col_sq_mean[j]
            elif z_j < -lambda_val:
                coef[j] = (z_j + lambda_val) / col_sq_mean[j]
            else:
                coef[j] = si.array(0.0)
        if si.sum(si.abs(coef - coef_old)) < tol:
            break

    idx = si.argsort(si.abs(coef))[-1]
    eta = (X.T @ X).inv() @ X.T
    return eta[idx]


def run_single_simulation(i, n_samples, n_features, K, lambda_grid, seed_offset):
    """単一のシミュレーション実行"""
    rng = np.random.default_rng(seed_offset + i)

    beta = np.zeros(n_features)
    X_np = rng.normal(0, 1, (n_samples, n_features))
    y_np = X_np @ beta + rng.normal(0, 1, n_samples)
    fold_indices = np.array_split(rng.permutation(n_samples), K)

    y = si.siarray(y_np)
    X = si.siarray(X_np)

    cv_r2_func = partial(
        cv_r2_lasso,
        X=X,
        fold_indices=fold_indices,
        lambda_grid=lambda_grid,
        K=K,
        n_features=n_features,
        n_samples=n_samples,
    )

    eta = cv_r2_func(y)
    result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=cv_r2_func)

    return i, result.p_value, result.naive_p_value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-initial", action="store_true",
                        help="初回単体実行をスキップして直接並列シミュレーションへ")
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=32)
    args = parser.parse_args()

    # ============================================================================
    # 設定
    # ============================================================================
    n_samples = 100
    n_features = 10
    K = 3
    lambda_grid = np.logspace(-4, 0, 10)

    rng = np.random.default_rng(0)

    # ============================================================================
    # 初回実行（動作確認）  --skip-initial で省略可能
    # ============================================================================
    if not args.skip_initial:
        print("初回実行中（時間がかかる場合は --skip-initial オプションを使用）...")
        beta = np.zeros(n_features)
        X_np = rng.normal(0, 1, (n_samples, n_features))
        y_np = X_np @ beta + rng.normal(0, 1, n_samples)
        fold_indices = np.array_split(rng.permutation(n_samples), K)

        y = si.siarray(y_np)
        X = si.siarray(X_np)

        cv_r2_func = partial(
            cv_r2_lasso,
            X=X,
            fold_indices=fold_indices,
            lambda_grid=lambda_grid,
            K=K,
            n_features=n_features,
            n_samples=n_samples,
        )

        eta = cv_r2_func(y)
        result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=cv_r2_func)
        print(f"p-value:       {result.p_value}")
        print(f"naive p-value: {result.naive_p_value}")
    else:
        print("初回実行をスキップ")

    # ============================================================================
    # 並列シミュレーション
    # ============================================================================
    n_simulations = args.n_simulations
    n_workers = args.n_workers
    print(f"並列実行開始 (ワーカー数: {n_workers}, シミュレーション数: {n_simulations})")

    p_values = [None] * n_simulations
    naive_p_values = [None] * n_simulations

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                run_single_simulation,
                i, n_samples, n_features, K, lambda_grid, 1000
            ): i
            for i in range(n_simulations)
        }

        with tqdm(total=n_simulations, desc="シミュレーション実行中") as pbar:
            for future in as_completed(futures):
                i, p_value, naive_p_value = future.result()
                p_values[i] = p_value
                naive_p_values[i] = naive_p_value
                pbar.set_postfix({"p-value": f"{p_value:.4f}", "naive": f"{naive_p_value:.4f}"})
                pbar.update(1)

    # ============================================================================
    # 可視化
    # ============================================================================
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(p_values, bins=20, range=(0, 1))
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.title("Selective p-values")
    plt.savefig("result/cv_r2_lasso_p_values_hist.pdf")
    plt.close()
    print("完了: result/cv_r2_lasso_p_values_hist.pdf")

    plt.figure()
    plt.hist(naive_p_values, bins=20, range=(0, 1))
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.title("Naive p-values")
    plt.savefig("result/cv_r2_lasso_naive_p_values_hist.pdf")
    plt.close()
    print("完了: result/cv_r2_lasso_naive_p_values_hist.pdf")
