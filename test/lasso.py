import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autosi as si

# ===== データの生成 =====

def lasso(y):
    alpha = si.array(0.1)  # L1正則化パラメータ
    max_iter = si.array(1000)  # 最大反復回数
    tol = si.array(1e-4)  # 収束判定の閾値
    coef = si.array(np.zeros(n_features))

    for iteration in range(max_iter):
        coef_old = coef.copy()

        # 各特徴量について順番に最適化
        for j in range(n_features):
            # j番目の特徴量を除いた予測値
            y_pred = X_train @ coef
            
            residual = y - y_pred + coef[j] * X_train[:, j]
            # print(residual.coef_den)
            # rhoの計算（残差とj番目の特徴量の内積）
            rho = X_train[:, j] @ residual / n_samples
            print(rho.coef_num)
            # print(X_train[:, j].coef_num)
            # print(len(rho.coef_num))
            lambda_val = alpha / n_samples

            if rho < -lambda_val:
                # print("1")
                coef[j] = rho + lambda_val
                # print(rho + lambda_val)
                # print(coef[j])
            elif rho > lambda_val:
                # print("2")
                coef[j] = rho - lambda_val
                # print(rho - lambda_val)
                # print(coef[j])
            else:
                # print("3")
                coef[j] = 0.0
                # print(coef[j])

        # 収束判定
        diff = si.sum(si.abs(coef - coef_old))
        if diff < tol:
            print(f"収束（iter: {iteration + 1}）")
            break

    idx = si.argsort(si.abs(coef))[-1]
    eta = (X_train.T @ X_train).inv() @ X_train.T
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

    # 訓練データとテストデータに分割
    split_idx = int(0.8 * n_samples)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    eta = lasso(y_train)

    result = si.inference(eta=eta, prob_vec=y_train, var=1.0, algorithm=lasso)
    print(f"p-value: {result.p_value}")

    # p_values = []

    # for i in range(1000):
    #     n_samples = 100
    #     n_features = 20

    #     beta = np.zeros(n_features)
    #     X = np.random.randn(n_samples, n_features)
    #     y = X @ beta + rng.normal(0, 1, n_samples)

    #     y = si.siarray(y)
    #     X = si.siarray(X)

    #     # 訓練データとテストデータに分割
    #     split_idx = int(0.8 * n_samples)
    #     X_train = X[:split_idx]
    #     X_test = X[split_idx:]
    #     y_train = y[:split_idx]
    #     y_test = y[split_idx:]

    #     eta = lasso(y_train)
    #     result = si.inference(eta=eta, prob_vec=y_train, var=1.0, algorithm=lasso)
    #     p_values.append(result.p_value)

    # import matplotlib.pyplot as plt
    # plt.hist(p_values, bins=20, range=(0, 1))
    # plt.xlabel("p-value")
    # plt.ylabel("Frequency")
    # plt.savefig("result/lasso_p_values_hist.pdf")
