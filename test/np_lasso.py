import numpy as np

def lasso(y):
    alpha = 0.1  # L1正則化パラメータ
    max_iter = 1000  # 最大反復回数
    tol = 1e-4  # 収束判定の閾値（siarrayにしない）

    coef = np.zeros(n_features)
    for iteration in range(max_iter):
        coef_old = coef.copy()

        # 各特徴量について順番に最適化
        for j in range(n_features):
            # j番目の特徴量を除いた予測値
            y_pred = X_train @ coef

            residual = y - y_pred + coef[j] * X_train[:, j]

            # rhoの計算（残差とj番目の特徴量の内積）
            rho = X_train[:, j] @ residual / n_samples
            # print(rho)
            lambda_val = alpha / n_samples

            if rho < -lambda_val:
                print("1")
                coef[j] = rho + lambda_val
                # print(rho + lambda_val)
                # print(coef[j])
            elif rho > lambda_val:
                print("2")
                coef[j] = rho - lambda_val
                # print(rho - lambda_val)
                # print(coef[j])
            else:
                print("3")
                coef[j] = 0.0
                # print(coef[j])

        # 収束判定
        diff = np.sum(np.abs(coef - coef_old))
        if diff < tol:
            print(f"収束（iter: {iteration + 1}）")
            break

    idx = np.argsort(np.abs(coef))[-1]
    eta = np.linalg.inv(X_train.T @ X_train) @ X_train.T
    return eta[idx]


if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.default_rng(0)
    n_samples = 100
    n_features = 20

    beta = np.zeros(n_features)
    X = np.random.randn(n_samples, n_features)
    y = X @ beta + rng.normal(0, 1, n_samples)

    # 訓練データとテストデータに分割
    split_idx = int(0.8 * n_samples)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    eta = lasso(y_train)
