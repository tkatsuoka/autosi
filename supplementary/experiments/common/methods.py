"""The three feature-selection methods implemented on top of autosi.

Each function takes `y` (an autosi tracked array) and returns the contrast
vector eta of one feature drawn uniformly at random (via the fixed random
number pick_u) from the selected feature set. `si.inference` calls the
function repeatedly while perturbing y along the line Y(z) = a + b z; the
autosi operations inside automatically track the selection event (the
intervals of z defined by polynomial inequalities).

Why the random draw of the tested feature is valid: pick_u is fixed
independently of y and only decides *which position* of the selected set is
tested. The set itself (top-k for marginal screening, the nonzero pattern
for the lasso variants) is determined by tracked comparisons (argsort /
soft-threshold branches), so within a tracked interval the selection, and
hence eta, is constant.

Inputs other than y (X, hyperparameters) are bound with functools.partial.
"""

import numpy as np

import autosi as si

# Side outputs of the most recent run: the selected feature set (top-k for
# marginal screening, nonzero-coefficient indices for the lasso variants)
# and the feature chosen for testing. The algorithm's return value must be
# the single eta vector (required by si.inference), so these by-products are
# exposed through module-level variables. They are overwritten during the
# line search of si.inference, so read them right after running the
# algorithm on the observed data.
last_active_set = None
last_picked = None


def _pick_pos(indices, pick_u):
    """Position in the (sorted) selected set indicated by pick_u in [0, 1).

    Depends only on the set size and pick_u, never on y.
    """
    return int(pick_u * len(indices)) % len(indices)


def _eta_selected_model(X, active, pos):
    """Row `pos` of (X_M^T X_M)^{-1} X_M^T for the selected columns.

    The test statistic is the least-squares coefficient in the regression on
    the selected features only (the "selected model" estimand). X is a
    constant and `active` is determined by tracked events, so eta is
    constant inside each tracked interval; collecting the intervals whose
    output matches the observed one conditions exactly on {M(z) = M_obs}.
    """
    X_M = X.data[:, active]
    eta_np = np.linalg.solve(X_M.T @ X_M, X_M.T)[pos]
    return si.siarray(eta_np)


def marginal_screening(y, X, k=3, pick_u=0.0):
    """Select the k features with the largest |X^T y|, draw one at random,
    and return the eta of its selected-model regression coefficient.

    The top-k set is identified through argsort (a tracked total order).
    The set is sorted in ascending order before drawing, so the tested
    feature depends only on *which* k features are on top, not on their
    relative order within the set.
    """
    corr = X.T @ y
    order = si.argsort(si.abs(corr))
    k = min(k, X.shape[1])
    topk = sorted(int(i) for i in order.data[-k:])
    global last_active_set, last_picked
    last_active_set = topk
    pos = _pick_pos(topk, pick_u)
    last_picked = topk[pos]
    return _eta_selected_model(X, topk, pos)


def _lasso_coordinate_descent(y, X, n_samples, n_features, lam, max_iter, tol):
    """Solve the lasso by coordinate descent (soft-thresholding).

    Objective: (1/(2n)) ||y - X beta||^2 + lam * ||beta||_1, so the meaning
    of lam does not depend on the sample size n.
    """
    col_sq_mean = si.sum(X ** 2, axis=0) / n_samples
    coef = si.array(np.zeros(n_features))

    for _ in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            # Univariate regression on the partial residual excluding j
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
    """Draw one feature at random from the lasso active set.

    The nonzero pattern is decided by the (tracked) soft-threshold branches
    of coordinate descent, so it is constant inside a tracked interval and
    can be read from .data. If all coefficients are zero the selection is
    degenerate: last_active_set records the empty set (the runner skips such
    trials on the observed data, following the convention that an empty
    selection is a failed trial). During the line search the algorithm still
    has to return something, so the draw falls back to all features; the
    output then differs from the observed one and the region is excluded
    from the conditioning set.
    """
    active = np.flatnonzero(np.abs(coef.data) > 0).tolist()
    global last_active_set, last_picked
    last_active_set = active
    pick_pool = active if active else list(range(n_features))
    pos = _pick_pos(pick_pool, pick_u)
    last_picked = pick_pool[pos]
    return pick_pool, pos


def lasso(y, X, n_samples, n_features, alpha=0.1, max_iter=1000, tol=1e-4,
          pick_u=0.0):
    """Lasso at a fixed regularization strength; test one feature drawn at
    random from the active set."""
    coef = _lasso_coordinate_descent(
        y, X, n_samples, n_features, alpha, max_iter, tol
    )
    pool, pos = _pick_from_active(coef, n_features, pick_u)
    return _eta_selected_model(X, pool, pos)


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
    """Select lambda by K-fold cross-validated R^2, refit the lasso on all
    data, and test one feature drawn at random from its active set."""

    # Step 1: average K-fold validation R^2 for each candidate lambda
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

            # Validation R^2
            y_val = y[val_idx]
            y_pred = X[val_idx] @ coef
            ss_res = si.sum((y_val - y_pred) ** 2)
            ss_tot = si.sum((y_val - si.sum(y_val) / len(val_idx)) ** 2)
            fold_r2 = fold_r2 + 1.0 - ss_res / ss_tot
        cv_r2.append(fold_r2 / K)

    # Step 2: lambda with the highest CV score
    lam_star_idx = si.argmax(si.stack(cv_r2))
    lam_star = lambda_grid[lam_star_idx]

    # Step 3: lasso on the full data at lambda*, then draw from the active set
    coef = _lasso_coordinate_descent(
        y, X, n_samples, n_features, lam_star, max_iter, tol
    )
    pool, pos = _pick_from_active(coef, n_features, pick_u)
    return _eta_selected_model(X, pool, pos)
