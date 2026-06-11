import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import autosi as si

# ============================================================================
# 設定
# ============================================================================

rng = np.random.default_rng(0)

n_samples = 1000
n_features = 5
K = 3
lambda_grid = np.logspace(-4, 0, 10)
max_iter = 1000
tol = 1e-4 # sklearn デフォルト

beta = np.zeros(n_features)
X_np = rng.normal(0, 1, (n_samples, n_features))
y_np = X_np @ beta + rng.normal(0, 1, n_samples)

fold_indices = np.array_split(rng.permutation(n_samples), K)

y = si.siarray(y_np)
X = si.siarray(X_np)


def _lasso_coordinate_descent(y, X, n_samples, n_features, lam, max_iter, tol, coef_init=None):
    """座標降下法で Lasso 係数を解く (sklearn 準拠)"""
    col_sq_mean = si.sum(X ** 2, axis=0) / n_samples
    
    if coef_init is None:
        coef = si.array(np.zeros(n_features))
    else:
        coef = coef_init.copy()

    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            r_j = y - X @ coef + X[:, j] * coef[j]
            z_j = X[:, j] @ r_j / n_samples
            
            if z_j > lam:
                coef[j] = (z_j - lam) / col_sq_mean[j]
            elif z_j < -lam:
                coef[j] = (z_j + lam) / col_sq_mean[j]
            else:
                coef[j] = si.array(0.0)
                
        # 注意: autosi に si.max がない場合は、元の si.sum(si.abs(...)) を使用してください
        if si.max(si.abs(coef - coef_old)) < tol:
            break
            
    return coef

def cv_r2_lasso(
    y,
    X,
    n_samples,
    n_features,
    fold_indices,
    lambda_grid,
    K=3,
    max_iter=1000,
    tol=1e-4,
):
    lambda_grid = np.sort(lambda_grid)[::-1]
    
    coefs_init = [si.array(np.zeros(n_features)) for _ in range(K)]
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
            X_val, y_val = X[val_idx], y[val_idx]
            
            # 中心化 (si.mean がない場合は si.sum / length)
            y_tr_mean = si.sum(y_tr) / n_tr
            X_tr_mean = si.sum(X_tr, axis=0) / n_tr
            y_tr_centered = y_tr - y_tr_mean
            X_tr_centered = X_tr - X_tr_mean

            coef = _lasso_coordinate_descent(
                y_tr_centered, X_tr_centered, n_tr, n_features, lam, max_iter, tol, coef_init=coefs_init[k]
            )
            coefs_init[k] = coef

            intercept = y_tr_mean - X_tr_mean @ coef
            y_pred = X_val @ coef + intercept
            
            ss_res = si.sum((y_val - y_pred) ** 2)
            y_val_mean = si.sum(y_val) / len(val_idx)
            ss_tot = si.sum((y_val - y_val_mean) ** 2)
            fold_r2 = fold_r2 + (1.0 - ss_res / ss_tot)
            
        cv_r2.append(fold_r2 / K)

    lam_star_idx = si.argmax(si.stack(cv_r2))
    lam_star = lambda_grid[lam_star_idx]

    y_mean = si.sum(y) / n_samples
    X_mean = si.sum(X, axis=0) / n_samples
    y_centered = y - y_mean
    X_centered = X - X_mean
    print(lam_star_idx)
    coef_full = si.array(np.zeros(n_features))
    for lam in lambda_grid[:lam_star_idx + 1]:
        coef_full = _lasso_coordinate_descent(
            y_centered, X_centered, n_samples, n_features, lam, max_iter, tol, coef_init=coef_full
        )

    idx = si.argsort(si.abs(coef_full))[-1]
    
    # autosi の逆行列メソッドに戻す
    eta = (X.T @ X).inv() @ X.T
    
    return eta[idx]


# ============================================================================
# 推論の実行
# ============================================================================

# 1. 実際に eta を計算
eta = cv_r2_lasso(
    y=y,
    X=X,
    n_samples=n_samples,
    n_features=n_features,
    fold_indices=fold_indices,
    lambda_grid=lambda_grid,
    K=K,
    max_iter=max_iter,
    tol=tol
)

# 2. si.inference が y のみを引数にとれるようにラッパー関数を作成
def algorithm_wrapper(y_new):
    return cv_r2_lasso(
        y=y_new,
        X=X,
        n_samples=n_samples,
        n_features=n_features,
        fold_indices=fold_indices,
        lambda_grid=lambda_grid,
        K=K,
        max_iter=max_iter,
        tol=tol
    )

import time
start_time = time.time()

# 3. ラッパー関数を渡して推論を実行
result = si.inference(
    eta=eta, 
    prob_vec=y, 
    var=1.0, 
    algorithm=algorithm_wrapper, # ★ここを変更
    memoize=True
)

end_time = time.time()
print(f"Inference completed in {end_time - start_time:.2f} seconds")
print(f"p-value: {result.p_value}")