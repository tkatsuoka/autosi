"""3 つの特徴選択手法を autosi 上で実装する。

各関数は `y` (siarray) を引数に取り、選択された変数のコントラスト
ベクトル eta を返す「selective inference アルゴリズム」である。
si.inference からは y を z 方向に摂動させながら繰り返し呼ばれ、
内部の si 演算が選択イベント (多項不等式の区間) を自動追跡する。

X など y 以外の入力は functools.partial で束縛して渡す
(test/lasso_parallel.py と同じ流儀)。
"""

import numpy as np

import autosi as si


def marginal_screening(y, X):
    """相関 (X^T y) が最大の変数を 1 つ選び、その eta を返す。"""
    corr = X.T @ y
    eta = (X.T @ X).inv() @ X.T
    idx = si.argsort(si.abs(corr))[-1]
    return eta[idx]


def _lasso_coordinate_descent(y, X, n_samples, n_features, lam, max_iter, tol):
    """座標降下法で Lasso 係数を解く (soft-thresholding)。"""
    col_sq_mean = si.sum(X ** 2, axis=0) / n_samples
    lambda_val = lam / n_samples
    coef = si.array(np.zeros(n_features))

    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            # j 番目を除いた部分残差に対する単回帰係数
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
    return coef


def lasso(y, X, n_samples, n_features, alpha=0.1, max_iter=1000, tol=1e-4):
    """固定 λ の Lasso を解き、絶対値最大の係数に対応する eta を返す。"""
    coef = _lasso_coordinate_descent(
        y, X, n_samples, n_features, alpha, max_iter, tol
    )
    idx = si.argsort(si.abs(coef))[-1]
    eta = (X.T @ X).inv() @ X.T
    return eta[idx]


def cv_r2_lasso(
    y,
    X,
    n_samples,
    n_features,
    fold_indices,
    lambda_grid,
    K=3,
    max_iter=1000,
    tol=1e-5,
):
    """K-fold CV の R^2 で λ を選択し、その Lasso で変数選択した eta を返す。"""

    # Step 1: 各 λ について K-fold CV の平均 R^2 を計算
    cv_r2 = []
    for lam in lambda_grid:
        fold_r2 = si.array(0.0)
        for k in range(K):
            train_idx = np.concatenate(
                [fold_indices[j] for j in range(K) if j != k]
            )
            val_idx = fold_indices[k]
            n_tr = len(train_idx)

            X_tr, y_tr = X[train_idx], y[train_idx]
            coef = _lasso_coordinate_descent(
                y_tr, X_tr, n_tr, n_features, lam, max_iter, tol
            )

            # バリデーションデータで R^2 を計算
            y_val = y[val_idx]
            y_pred = X[val_idx] @ coef
            ss_res = si.sum((y_val - y_pred) ** 2)
            ss_tot = si.sum((y_val - si.sum(y_val) / len(val_idx)) ** 2)
            fold_r2 = fold_r2 + 1.0 - ss_res / ss_tot
        cv_r2.append(fold_r2 / K)

    # Step 2: R^2 が最大の λ* を選択
    lam_star_idx = si.argmax(si.stack(cv_r2))
    lam_star = lambda_grid[lam_star_idx]

    # Step 3: λ* で全データに Lasso を適用し変数選択
    coef = _lasso_coordinate_descent(
        y, X, n_samples, n_features, lam_star, max_iter, tol
    )
    idx = si.argsort(si.abs(coef))[-1]
    eta = (X.T @ X).inv() @ X.T
    return eta[idx]
