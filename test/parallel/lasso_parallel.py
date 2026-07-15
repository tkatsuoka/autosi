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


def lasso(y, X_train, n_samples, n_features):
    alpha = si.array(0.1)
    max_iter = si.array(1000)
    tol = si.array(1e-4)
    # sklearn と同じ規約 (1/(2n))RSS + α|β|₁ の座標降下法。
    # 更新式は beta_j = S(x_j^T r / n, α) / (x_j^T x_j / n)
    col_sq_mean = si.sum(X_train ** 2, axis=0) / n_samples
    coef = si.array(np.zeros(n_features))

    for iteration in range(max_iter):
        coef_old = coef.copy()

        for j in range(n_features):
            y_pred = X_train @ coef
            residual = y - y_pred + coef[j] * X_train[:, j]
            rho = X_train[:, j] @ residual / n_samples

            if rho < -alpha:
                coef[j] = (rho + alpha) / col_sq_mean[j]
            elif rho > alpha:
                coef[j] = (rho - alpha) / col_sq_mean[j]
            else:
                coef[j] = si.array(0.0)

        diff = si.sum(si.abs(coef - coef_old))
        if diff < tol:
            break

    idx = si.argsort(si.abs(coef))[-1]
    eta = (X_train.T @ X_train).inv() @ X_train.T
    return eta[idx]


def run_single_simulation(i, n_samples, n_features, seed_offset):
    """単一のシミュレーション実行"""
    # 各プロセスでCPUを1コアのみ使用するよう制限    
    rng = np.random.default_rng(seed_offset + i)
    
    beta = np.zeros(n_features)
    X = rng.standard_normal((n_samples, n_features))
    y = X @ beta + rng.normal(0, 1, n_samples)

    y = si.siarray(y)
    X = si.siarray(X)

    split_idx = int(0.8 * n_samples)
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    eta = lasso(y_train, X_train, split_idx, n_features)
    
    # lasso関数を部分適用してアルゴリズムとして渡す
    lasso_func = partial(lasso, X_train=X_train, n_samples=split_idx, n_features=n_features)
    result = si.inference(eta=eta, prob_vec=y_train, var=1.0, algorithm=lasso_func)
    
    return i, result.p_value


if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.default_rng(0)
    
    # 初回実行
    n_samples = 100
    n_features = 5

    beta = np.zeros(n_features)
    X = np.random.randn(n_samples, n_features)
    y = X @ beta + rng.normal(0, 1, n_samples)

    y = si.array(y)
    X = si.array(X)

    split_idx = int(0.8 * n_samples)
    X_train = X[:split_idx]
    y_train = y[:split_idx]

    eta = lasso(y_train, X_train, split_idx, n_features)
    
    lasso_func = partial(lasso, X_train=X_train, n_samples=split_idx, n_features=n_features)
    result = si.inference(eta=eta, prob_vec=y_train, var=1.0, algorithm=lasso_func)
    print(f"p-value: {result.p_value}")

    # 並列シミュレーション
    n_simulations = 1000
    
    # CPUコア数を取得（必要に応じて調整）
    n_workers = 32
    print(f"並列実行開始 (ワーカー数: {n_workers})")
    
    p_values = [None] * n_simulations
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 並列実行
        futures = {
            executor.submit(run_single_simulation, i, n_samples, n_features, 1000): i
            for i in range(n_simulations)
        }
        
        # 進捗バー付きで結果を収集
        with tqdm(total=n_simulations, desc="シミュレーション実行中") as pbar:
            for future in as_completed(futures):
                i, p_value = future.result()
                p_values[i] = p_value
                pbar.set_postfix({"最新のp-value": f"{p_value:.4f}"})
                pbar.update(1)

    # 可視化
    import matplotlib.pyplot as plt
    plt.hist(p_values, bins=20, range=(0, 1))
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.savefig("result/lasso_hist.pdf")
    print("完了: result/lasso_hist.pdf")