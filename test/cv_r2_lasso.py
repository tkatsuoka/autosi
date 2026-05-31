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
K = 3
lambda_grid = np.logspace(-4, 0, 10)
max_iter = 1000
tol = 1e-4

beta = np.zeros(n_features)
X_np = rng.normal(0, 1, (n_samples, n_features))
y_np = X_np @ beta + rng.normal(0, 1, n_samples)

fold_indices = np.array_split(rng.permutation(n_samples), K)

y = si.siarray(y_np)
X = si.siarray(X_np)


# ============================================================================
# cv_r2_lasso: CV-R² で λ を選択し、Lasso 係数を推定する
#
# この関数は si.inference から繰り返し呼ばれる。
# 呼ばれるたびに引数 y が z 方向に微小摂動された siarray として渡され、
# 内部の si 演算が「y がどう変わると選択結果が変わるか」の区間を自動追跡する。
# ============================================================================

def cv_r2_lasso(y):

    # -----------------------------------------------------------------------
    # Step 1: K フォールド CV で各 λ の R² を計算
    #
    # si.array を使うことで、Lasso の各比較・R² の大小関係が
    # IntervalTracker に自動的に記録される。
    # -----------------------------------------------------------------------
    cv_r2 = []
    for lam in lambda_grid:
        fold_r2 = si.array(0.0)
        for k in range(K):
            train_idx = np.concatenate([fold_indices[j] for j in range(K) if j != k])
            val_idx = fold_indices[k]
            n_tr = len(train_idx)

            X_tr, y_tr = X[train_idx], y[train_idx]

            # 列ごとの X_j² の平均 (更新式の分母; 参照実装の col_sq_mean に対応)
            col_sq_mean = si.sum(X_tr ** 2, axis=0) / n_tr

            # 座標降下法で Lasso を解く
            lv = lam / n_tr
            coef = si.array(np.zeros(n_features))
            for _ in range(max_iter):
                coef_old = coef.copy()
                for j in range(n_features):
                    r_j = y_tr - X_tr @ coef + X_tr[:, j] * coef[j]  # 部分残差
                    z_j = X_tr[:, j] @ r_j / n_tr
                    if z_j > lv:
                        coef[j] = (z_j - lv) / col_sq_mean[j]
                    elif z_j < -lv:
                        coef[j] = (z_j + lv) / col_sq_mean[j]
                    else:
                        coef[j] = si.array(0.0)
                if si.sum(si.abs(coef - coef_old)) < tol:
                    break

            # バリデーションデータで R² を計算
            y_val = y[val_idx]
            y_pred = X[val_idx] @ coef
            ss_res = si.sum((y_val - y_pred) ** 2)
            ss_tot = si.sum((y_val - si.sum(y_val) / len(val_idx)) ** 2)
            # print(ss_res.deg, ss_tot.deg)
            # print((ss_res / ss_tot).deg)
            fold_r2 = fold_r2 + 1.0 - ss_res / ss_tot
            print(fold_r2.deg)
        cv_r2.append(fold_r2 / K)

    # -----------------------------------------------------------------------
    # Step 2: argmax で最適 λ* を選択
    #
    # si.argsort が隣接要素の大小比較を記録するため、
    # λ* の選択条件が自動的に IntervalTracker に登録される。
    # -----------------------------------------------------------------------
    # print([r2.deg for r2 in cv_r2])
    # lam_star_idx = si.argsort(si.stack(cv_r2))[-1]
    lam_star_idx = si.argmax(si.stack(cv_r2))
    lam_star = lambda_grid[lam_star_idx]

    # -----------------------------------------------------------------------
    # Step 3: λ* で全データに Lasso を適用
    # -----------------------------------------------------------------------
    col_sq_mean = si.sum(X ** 2, axis=0) / n_samples
    lambda_val = lam_star / n_samples
    coef = si.array(np.zeros(n_features))
    # print(coef.deg)
    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            r_j = y - X @ coef + X[:, j] * coef[j]  # 部分残差
            z_j = X[:, j] @ r_j / n_samples
            if z_j > lambda_val:
                coef[j] = (z_j - lambda_val) / col_sq_mean[j]
                # print(coef[j].deg)
            elif z_j < -lambda_val:
                coef[j] = (z_j + lambda_val) / col_sq_mean[j]
            else:
                coef[j] = si.array(0.0)
        if si.sum(si.abs(coef - coef_old)) < tol:
            break
    # print(coef.deg)
    # 絶対値最大の係数に対応する変数のコントラストベクトル η を返す
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

# p_values = []
# for i in range(500):
#     X_np = rng.normal(0, 1, (n_samples, n_features))
#     y_np = X_np @ beta + rng.normal(0, 1, n_samples)
#     fold_indices = np.array_split(rng.permutation(n_samples), K)
#     y = si.siarray(y_np)
#     X = si.siarray(X_np)
#
#     eta = cv_r2_lasso(y)
#     result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=cv_r2_lasso)
#     p_values.append(result.p_value)
#     print(f"Iteration {i + 1}, p-value: {result.p_value}")

# import matplotlib.pyplot as plt
# plt.hist(p_values, bins=20, range=(0, 1))
# plt.xlabel("p-value")
# plt.ylabel("Frequency")
# plt.savefig("result/cv_r2_lasso_p_values_hist.pdf")
