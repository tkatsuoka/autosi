import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import sipy as si

# ===== データの生成 =====
np.random.seed(42)
rng = np.random.default_rng(0)
n_samples = 100
n_features = 20

beta = np.zeros(n_features)
X = np.random.randn(n_samples, n_features)
y = X @ beta + rng.normal(0, 1, n_samples)

y = si.siarray(y)
X = si.siarray(X)

# 訓練データとテストデータに分割
split_idx = int(0.8 * n_samples)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# ===== ハイパーパラメータの設定 =====
alpha = 0.1  # L1正則化パラメータ
max_iter = 1000  # 最大反復回数
tol = 1e-4  # 収束判定の閾値

# ===== 係数の初期化 =====
coef = np.zeros(n_features)

# ===== Lasso回帰の学習（座標降下法） =====
print("学習開始")
for iteration in range(max_iter):
    coef_old = coef.copy()

    # 各特徴量について順番に最適化
    for j in range(n_features):
        # j番目の特徴量を除いた予測値
        y_pred = X_train @ coef
        residual = y_train - y_pred + coef[j] * X_train[:, j]

        # rhoの計算（残差とj番目の特徴量の内積）
        rho = X_train[:, j] @ residual / n_samples

        # ソフト閾値を適用してj番目の係数を更新
        lambda_val = alpha / n_samples
        # if rho < -lambda_val:
        #     coef[j] = rho + lambda_val
        # elif rho > lambda_val:
        #     coef[j] = rho - lambda_val
        # else:
        #     coef[j] = 0.0
        
        coef[j] = si.where(rho < -lambda_val, rho + lambda_val, 0) 
        coef[j] += si.where(rho > lambda_val, rho - lambda_val, 0)

    # 収束判定
    if np.sum(np.abs(coef - coef_old)) < tol:
        print(f"収束（反復回数: {iteration + 1}）")
        break

# ===== 結果の表示 =====
print(f"{coef=}")