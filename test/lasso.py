import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import autosi as si

# ===== データの生成 =====


def lasso(y):
    # ===== ハイパーパラメータの設定 =====
    alpha = 0.1  # L1正則化パラメータ
    max_iter = 1000  # 最大反復回数
    tol = 1e-4  # 収束判定の閾値（siarrayにしない）

    # ===== 係数の初期化 =====
    coef = si.siarray(np.zeros(n_features))

    # ===== Lasso回帰の学習（座標降下法） =====
    for iteration in range(max_iter):
        coef_old = coef.copy()

        # 各特徴量について順番に最適化
        for j in range(n_features):
            # j番目の特徴量を除いた予測値
            y_pred = X_train @ coef
            # if isinstance(y, si.RationalFunction):
            #     print("debug y:", y)
            #     print(y.coef_den)

            residual = y - y_pred + coef[j] * X_train[:, j]
            # print(y_pred.data)
            # print("debug residual:", residual.data[0])

            # rhoの計算（残差とj番目の特徴量の内積）
            rho = X_train[:, j] @ residual / n_samples
            # print("debug rho:", rho.data)
            
            # if isinstance(rho, si.RationalFunction):
            #     # print("debug rho:", rho.data())
            #     # print(rho.coef_num)
            #     print("debug", (X_train[:, j] @ residual).data)
            # else:
            #     # print("debug rho:", rho.data)
            #     print("debug", (X_train[:, j] @ residual).data)

            lambda_val = alpha / n_samples
            
            # 比較結果の.dataを使う
            if (rho < -lambda_val).data:
                coef[j] = rho + lambda_val
            elif (rho > lambda_val).data:
                coef[j] = rho - lambda_val
            else:
                coef[j] = 0.0

            # if isinstance(coef, si.RationalFunction):
            #     print("debug coef:", coef)
            #     print(coef.coef_num)
            # else:
            #     print("debug coef:", coef.data)
        
        # 収束判定
        diff = si.sum(si.abs(coef - coef_old)).data
        if diff < tol:
            print(f"収束（iter: {iteration + 1}）")
            break
    
    # 選ばれた特徴量のインデックス
    idx = si.argsort(si.abs(coef))[ -1]
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

    y = si.siarray(y)
    X = si.siarray(X)

    # 訓練データとテストデータに分割
    split_idx = int(0.8 * n_samples)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    eta = lasso(y_train)

    result = si.inference(eta=eta, prob_vec=y_train, var=1.0, algorithm=lasso)
    print(f"p-value: {result.p_value}")

    p_values = []

    for i in range(1000):
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

        eta = lasso(y_train)
        result = si.inference(eta=eta, prob_vec=y_train, var=1.0, algorithm=lasso)
        p_values.append(result.p_value)

    import matplotlib.pyplot as plt
    plt.hist(p_values, bins=20, range=(0, 1))
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.savefig("result/lasso_p_values_hist.pdf")