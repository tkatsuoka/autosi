import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import autosi as si


# ============================================================================
# 設定
# ============================================================================

rng = np.random.default_rng(0)

n_samples = 100
n_features = 10
K = 5                                 # CV fold 数 (degree 2K 次の選択イベント)
lambda_grid = np.logspace(-4, 0, 20)  # λ の候補グリッド
max_iter = 1000
tol = 1e-4

beta = np.zeros(n_features)
X_np = rng.normal(0, 1, (n_samples, n_features))
y_np = X_np @ beta + rng.normal(0, 1, n_samples)

fold_indices = np.array_split(rng.permutation(n_samples), K) # CV の fold インデックス

y = si.siarray(y_np)
X = si.siarray(X_np)


# ============================================================================
# numpy Lasso (座標降下法)
# ============================================================================

def _lasso_np(X_b, y_b, lam):
    n_b, p = X_b.shape
    lv = lam / n_b
    coef = np.zeros(p)
    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(p):
            r = y_b - X_b @ coef + coef[j] * X_b[:, j]
            rho = X_b[:, j] @ r / n_b
            if rho < -lv:
                coef[j] = rho + lv
            elif rho > lv:
                coef[j] = rho - lv
            else:
                coef[j] = 0.0
        if np.sum(np.abs(coef - coef_old)) < tol:
            break
    return coef


# ============================================================================
# numpy で CV (R²) を回して λ* を確定
# ============================================================================

_coef_train = {}
for li, lam in enumerate(lambda_grid):
    for k in range(K):
        test_idx = fold_indices[k]
        train_idx = np.concatenate([fold_indices[j] for j in range(K) if j != k])
        _coef_train[(li, k)] = _lasso_np(X_np[train_idx], y_np[train_idx], lam)

_cv_r2_np = np.zeros(len(lambda_grid))
for li in range(len(lambda_grid)):
    r2s = []
    for k in range(K):
        idx = fold_indices[k]
        y_t, pred = y_np[idx], X_np[idx] @ _coef_train[(li, k)]
        ss_res = np.sum((y_t - pred) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    _cv_r2_np[li] = np.mean(r2s)

_lam_star_idx = int(np.argmax(_cv_r2_np))
_lam_star = lambda_grid[_lam_star_idx]

print("optimal lambda :", _lam_star)


# ============================================================================
# CV-R² + Lasso
# ============================================================================

def cv_r2_lasso(y):
    # --- R² CV で λ 選択条件を記録 ---
    cv_r2_si = []
    for li in range(len(lambda_grid)):
        r2_sum = si.array(0.0)
        for k in range(K):
            idx = fold_indices[k]
            y_test = y[idx]
            pred = si.array(X_np[idx] @ _coef_train[(li, k)])
            y_mean = si.sum(y_test) / len(idx)
            ss_res = si.sum((y_test - pred) ** 2)
            ss_tot = si.sum((y_test - y_mean) ** 2)
            r2_sum = r2_sum + si.array(1.0) - ss_res / ss_tot
        cv_r2_si.append(r2_sum / K)

    for li in range(len(lambda_grid)):
        if li != _lam_star_idx:
            _ = (cv_r2_si[_lam_star_idx] > cv_r2_si[li])

    # --- 座標降下法 (lasso.py スタイル) ---
    lambda_val = _lam_star / n_samples
    coef = si.array(np.zeros(n_features))

    for _ in range(max_iter):
        coef_np = np.array([float(coef[j].data) for j in range(n_features)])

        for j in range(n_features):
            residual = y - X @ si.array(coef_np) + si.array(coef_np[j]) * X[:, j]
            rho = X[:, j] @ residual / n_samples

            if rho < -lambda_val:
                coef[j] = rho + lambda_val
            elif rho > lambda_val:
                coef[j] = rho - lambda_val
            else:
                coef[j] = si.array(0.0)

        coef_np_new = np.array([float(coef[j].data) for j in range(n_features)])
        if np.sum(np.abs(coef_np_new - coef_np)) < tol:
            break

    idx = si.argsort(si.abs(coef))[-1]
    eta = (X.T @ X).inv() @ X.T
    return eta[idx]


# ============================================================================
# 選択的推論
# ============================================================================

eta = cv_r2_lasso(y)
result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=cv_r2_lasso)
print(f"p-value: {result.p_value}")


# ============================================================================
# 帰無分布チェック (任意)
# ============================================================================

p_values = []
for i in range(500):
    X_np = rng.normal(0, 1, (n_samples, n_features))
    y_np = X_np @ beta + rng.normal(0, 1, n_samples)
    fold_indices = np.array_split(rng.permutation(n_samples), K)
    y = si.siarray(y_np)
    X = si.siarray(X_np)

    _coef_train = {}
    for li, lam in enumerate(lambda_grid):
        for k in range(K):
            test_idx = fold_indices[k]
            train_idx = np.concatenate([fold_indices[j] for j in range(K) if j != k])
            _coef_train[(li, k)] = _lasso_np(X_np[train_idx], y_np[train_idx], lam)
    _cv_r2_np = np.zeros(len(lambda_grid))
    for li in range(len(lambda_grid)):
        r2s = []
        for k in range(K):
            idx = fold_indices[k]
            y_t, pred = y_np[idx], X_np[idx] @ _coef_train[(li, k)]
            ss_res = np.sum((y_t - pred) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        _cv_r2_np[li] = np.mean(r2s)
    _lam_star_idx = int(np.argmax(_cv_r2_np))
    _lam_star = lambda_grid[_lam_star_idx]

    eta = cv_r2_lasso(y)
    result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=cv_r2_lasso)
    p_values.append(result.p_value)
    print(f"Iteration {i + 1}, p-value: {result.p_value}")

import matplotlib.pyplot as plt
plt.hist(p_values, bins=20, range=(0, 1))
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.savefig("result/cv_r2_lasso_p_values_hist.pdf")
