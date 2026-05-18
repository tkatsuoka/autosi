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
lam = 0.1          # Lasso の正則化パラメータ λ (固定)
max_iter = 1000
tol = 1e-4
B = 10             # サブサンプル数
pi_thr = 0.6       # 安定性の閾値 π_thr ∈ (0.5, 1)
subsample_size = n_samples // 2

# データ生成 (帰無: beta = 0)
beta = np.zeros(n_features)
X_np = rng.normal(0, 1, (n_samples, n_features))
y_np = X_np @ beta + rng.normal(0, 1, n_samples)

# サブサンプルのインデックスは y と独立に固定
subsample_indices = [
    rng.choice(n_samples, size=subsample_size, replace=False)
    for _ in range(B)
]

y = si.siarray(y_np)
X = si.siarray(X_np)


# ============================================================================
# numpy で各サブサンプルの Lasso 収束解を事前計算
# ============================================================================

def _lasso_np(X_b, y_b):
    n_b = subsample_size
    lv = lam / n_b
    coef = np.zeros(n_features)
    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
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

_coef_np_list = [
    _lasso_np(X_np[subsample_indices[b]], y_np[subsample_indices[b]])
    for b in range(B)
]


# ============================================================================
# Stability Selection
#   base learner: Lasso (座標降下法, λ 固定)
#   選択スコア: Σ_b |β̂^(b)_j|  (siarray で追跡)
#   SI 検定対象: スコア最大の変数 j* を si.argsort で選択
# ============================================================================

def stability_selection(y):
    score_si = si.array(np.zeros(n_features))

    for b in range(B):
        idx = subsample_indices[b]
        X_b = X[idx]
        y_b = y[idx]
        coef_np_b = _coef_np_list[b]
        lambda_val = lam / subsample_size

        # KKT 条件を1回だけ siarray で記録 (hybrid: 残差に numpy 定数を使用)
        coef_si = si.array(np.zeros(n_features))
        for j in range(n_features):
            r_j = y_b - X_b @ si.array(coef_np_b) + si.array(coef_np_b[j]) * X_b[:, j]
            rho_j = X_b[:, j] @ r_j / subsample_size

            if coef_np_b[j] > 0:
                _ = (rho_j > si.array(lambda_val))
                coef_si[j] = rho_j - si.array(lambda_val)
            elif coef_np_b[j] < 0:
                _ = (rho_j < -si.array(lambda_val))
                coef_si[j] = rho_j + si.array(lambda_val)
            else:
                _ = (rho_j > -si.array(lambda_val))
                _ = (rho_j < si.array(lambda_val))
                coef_si[j] = si.array(0.0)

        score_si = score_si + si.abs(coef_si)

    # j* 選択条件を si.argsort で記録
    idx = si.argsort(score_si)[-1]
    eta_mat = (X.T @ X).inv() @ X.T
    return eta_mat[idx]


# ============================================================================
# 選択的推論
# ============================================================================

# 観測データでの選択頻度を確認 (デバッグ用)
_counts = np.zeros(n_features, dtype=int)
for b in range(B):
    for j in range(n_features):
        if abs(_coef_np_list[b][j]) > 0:
            _counts[j] += 1

print("selection counts:", _counts)
print("pi_hat:", (_counts / B).round(2))
print("stable variables (pi_hat >= pi_thr):", np.where(_counts / B >= pi_thr)[0].tolist())
_scores = np.sum([np.abs(_coef_np_list[b]) for b in range(B)], axis=0)
_j_star = int(np.argmax(_scores))
print("selected variable (j*):", _j_star)

eta = stability_selection(y)
result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=stability_selection)
print("p-value:", result.p_value)


# ============================================================================
# 帰無分布チェック (任意)
# ============================================================================

# p_values = []
# for i in range(500):
#     X_np = rng.normal(0, 1, (n_samples, n_features))
#     y_np = X_np @ beta + rng.normal(0, 1, n_samples)
#     subsample_indices = [
#         rng.choice(n_samples, size=subsample_size, replace=False)
#         for _ in range(B)
#     ]
#
#     y = si.siarray(y_np)
#     X = si.siarray(X_np)
#
#     _coef_np_list = [
#         _lasso_np(X_np[subsample_indices[b]], y_np[subsample_indices[b]])
#         for b in range(B)
#     ]
#
#     eta = stability_selection(y)
#     result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=stability_selection)
#     p_values.append(result.p_value)
#     print(f"Iteration {i + 1}, p-value: {result.p_value}")
#
# import matplotlib.pyplot as plt
# plt.hist(p_values, bins=20, range=(0, 1))
# plt.xlabel("p-value")
# plt.ylabel("Frequency")
# plt.savefig("result/stability_selection_p_values_hist.pdf")
