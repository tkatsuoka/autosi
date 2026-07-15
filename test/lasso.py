import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# ===== データの生成 =====

import numpy as np
import autosi as si

def lasso(y):
    alpha = 0.1
    max_iter = 1000
    tol = 1e-4
    # 分割後の訓練データ行数を使う (グローバルの n_samples は分割前の総数)
    n_train = X_train.shape[0]
    # sklearn と同じ規約 (1/(2n))RSS + α|β|₁ の座標降下法。
    # 更新式は beta_j = S(x_j^T r / n, α) / (x_j^T x_j / n)
    col_sq_mean = si.sum(X_train ** 2, axis=0) / n_train
    coef = si.array(np.zeros(n_features))

    for iteration in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            y_pred = X_train @ coef

            residual = y - y_pred + coef[j] * X_train[:, j]
            rho = X_train[:, j] @ residual / n_train

            if rho < -alpha:
                coef[j] = (rho + alpha) / col_sq_mean[j]
            elif rho > alpha:
                coef[j] = (rho - alpha) / col_sq_mean[j]
            else:
                coef[j] = si.array(0.0)

        # 収束判定
        diff = si.sum(si.abs(coef - coef_old))
        if diff < tol:
            # print("iter", iteration + 1)
            break

    idx = si.argsort(si.abs(coef))[-1]
    eta = (X_train.T @ X_train).inv() @ X_train.T
    return eta[idx]


if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.default_rng(0)
    n_samples = 100
    n_features = 10

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

    # p_values = []

    # for i in range(1000):
    #     n_samples = 100
    #     n_features = 10

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
    #     print(f"Iteration {i + 1}, p-value: {result.p_value}")

    # import matplotlib.pyplot as plt
    # plt.hist(p_values, bins=20, range=(0, 1))
    # plt.xlabel("p-value")
    # plt.ylabel("Frequency")
    # plt.savefig("result/lasso_p_values_hist.pdf")
