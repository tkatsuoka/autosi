import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autosi as si


def lasso(y):
    alpha = si.array(0.1)
    max_iter = si.array(1000)
    tol = si.array(1e-4)
    
    # ===== 正規化の前処理 =====
    # 特徴量の標準化（平均0、標準偏差1）
    # X_mean = si.mean(X_train, axis=0)
    # X_std = si.std(X_train, axis=0)
    # 標準偏差が0の場合を避けるため、小さな値でクリップ
    # X_std = si.where(X_std < 1e-10, si.array(1.0), X_std)
    # 手動実装の例
    X_mean = si.sum(X_train, axis=0) / X_train.shape[0]
    X_var = si.sum((X_train - X_mean) ** 2, axis=0) / X_train.shape[0]
    X_std = si.sqrt(X_var)
    X_normalized = (X_train - X_mean) / X_std
    
    # 目的変数の中心化
    y_mean = si.mean(y)
    y_centered = y - y_mean
    
    n_samples_train = X_normalized.shape[0]
    coef = si.array(np.zeros(n_features))

    for iteration in range(max_iter):
        coef_old = coef.copy()

        for j in range(n_features):
            y_pred = X_normalized @ coef
            residual = y_centered - y_pred + coef[j] * X_normalized[:, j]
            rho = X_normalized[:, j] @ residual / n_samples_train
            lambda_val = alpha / n_samples_train

            if rho < -lambda_val:
                coef[j] = rho + lambda_val
            elif rho > lambda_val:
                coef[j] = rho - lambda_val
            else:
                coef[j] = si.array(0.0)

        diff = si.sum(si.abs(coef - coef_old))
        if diff < tol:
            print("iter", iteration + 1)
            break

    # 元のスケールに戻す（推論用）
    coef_original = coef / X_std
    
    idx = si.argsort(si.abs(coef))[-1]
    print("idx", idx)
    
    # 正規化されたデータでetaを計算
    eta = (X_normalized.T @ X_normalized).inv() @ X_normalized.T
    return eta[idx]


if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.default_rng(0)
    n_samples = 100
    n_features = 20

    beta = np.zeros(n_features)
    X = np.random.randn(n_samples, n_features)
    y = X @ beta + rng.normal(0, 1, n_samples)

    y = si.array(y)
    X = si.array(X)

    split_idx = int(0.8 * n_samples)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    eta = lasso(y_train)

    result = si.inference(eta=eta, prob_vec=y_train, var=1.0, algorithm=lasso)
    print(f"p-value: {result.p_value}")