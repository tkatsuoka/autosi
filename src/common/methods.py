"""3 つの特徴選択手法を autosi 上で実装する。

各関数は `y` (siarray) を引数に取り、選択された特徴集合から乱数 pick_u で
一様に選んだ 1 変数のコントラストベクトル eta を返す「selective inference
アルゴリズム」である。si.inference からは y を z 方向に摂動させながら
繰り返し呼ばれ、内部の si 演算が選択イベント (多項不等式の区間) を自動
追跡する。

検定対象のランダム選択が valid である理由: pick_u は y と独立に固定され、
「集合の何番目を選ぶか」だけを決める。集合そのもの (ms の top-k、lasso 系
の非零パターン) は追跡済みの比較 (argsort / 座標降下の soft-threshold 分岐)
で決まるため、追跡区間の内側では選択結果が不変になる。

X など y 以外の入力は functools.partial で束縛して渡す
(test/lasso_parallel.py と同じ流儀)。
"""

import numpy as np

import autosi as si

# 直近に実行した手法の選択特徴集合 (ms は top-k、lasso/cv は係数が非零の
# 特徴のインデックス列)。アルゴリズムの戻り値は eta 1 本に固定されている
# (si.inference の要求) ため、副産物はこの変数経由で取り出す。
# si.inference の探索中にも上書きされるので、観測データでの実行直後に
# 読み取ること。
last_active_set = None


def _pick_from(indices, pick_u):
    """特徴集合 (昇順リスト) から乱数 pick_u ∈ [0,1) で 1 つ選ぶ。

    選択は集合のサイズと pick_u のみに依存する (y には依存しない)。
    """
    return indices[int(pick_u * len(indices)) % len(indices)]


def marginal_screening(y, X, k=3, pick_u=0.0):
    """|X^T y| 上位 k 個の特徴集合から乱数で 1 つ選び、その eta を返す。

    top-k 集合の同定には argsort (全順序の追跡) を使う。集合は昇順に
    並べ替えてから選ぶため、検定対象は「どの k 個が上位か」だけに依存し、
    集合内の順位の入れ替わりには依存しない。
    """
    corr = X.T @ y
    order = si.argsort(si.abs(corr))
    k = min(k, X.shape[1])
    topk = sorted(int(i) for i in order.data[-k:])
    global last_active_set
    last_active_set = topk
    eta = (X.T @ X).inv() @ X.T
    return eta[_pick_from(topk, pick_u)]


def _lasso_coordinate_descent(y, X, n_samples, n_features, lam, max_iter, tol):
    """座標降下法で Lasso 係数を解く (soft-thresholding)。

    目的関数は sklearn と同じ規約 (1/(2n))||y - X beta||^2 + lam * ||beta||_1。
    lam の意味がサンプル数 n に依存しないため、n を振る実験でも
    正則化の強さが揃う。
    """
    col_sq_mean = si.sum(X ** 2, axis=0) / n_samples
    coef = si.array(np.zeros(n_features))

    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            # j 番目を除いた部分残差に対する単回帰係数
            r_j = y - X @ coef + X[:, j] * coef[j]
            z_j = X[:, j] @ r_j / n_samples
            if z_j > lam:
                coef[j] = (z_j - lam) / col_sq_mean[j]
            elif z_j < -lam:
                coef[j] = (z_j + lam) / col_sq_mean[j]
            else:
                coef[j] = si.array(0.0)
        if si.sum(si.abs(coef - coef_old)) < tol:
            break
    return coef


def _pick_from_active(coef, n_features, pick_u):
    """Lasso 係数の非零パターン (アクティブ集合) から乱数で 1 特徴選ぶ。

    非零パターンは座標降下の soft-threshold 分岐 (追跡済み) で決まるため、
    追跡区間の内側では集合が不変で、.data から読み出してよい。
    全係数 0 の縮退解 (λ が max|x_j^T y|/n を超えた場合のみ) は保険として
    全特徴からの選択に落とす。
    """
    active = np.flatnonzero(np.abs(coef.data) > 0).tolist()
    if not active:
        active = list(range(n_features))
    global last_active_set
    last_active_set = active
    return active, _pick_from(active, pick_u)


def lasso(y, X, n_samples, n_features, alpha=0.1, max_iter=1000, tol=1e-4,
          pick_u=0.0):
    """固定 λ の Lasso を解き、アクティブ集合から乱数で選んだ eta を返す。"""
    coef = _lasso_coordinate_descent(
        y, X, n_samples, n_features, alpha, max_iter, tol
    )
    _, idx = _pick_from_active(coef, n_features, pick_u)
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
    tol=1e-4,
    pick_u=0.0,
):
    """K-fold CV の R^2 で λ を選択し、その Lasso のアクティブ集合から
    乱数で選んだ特徴の eta を返す。"""

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

    # Step 3: λ* で全データに Lasso を適用し、アクティブ集合から乱数で選ぶ
    coef = _lasso_coordinate_descent(
        y, X, n_samples, n_features, lam_star, max_iter, tol
    )
    _, idx = _pick_from_active(coef, n_features, pick_u)
    eta = (X.T @ X).inv() @ X.T
    return eta[idx]
