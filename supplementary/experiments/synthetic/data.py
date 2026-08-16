"""Synthetic data generation.

The correlated setting uses the covariance Sigma[i, j] = 0.5^|i-j| (AR(1)).

- iid:  noise covariance is the identity and X is i.i.d.
- corr: both the noise and each column of X are correlated across
        observations with the AR(1) structure (adjacent observations are
        the most strongly correlated).

signal=0 gives the null setting (type I error rate); signal>0 gives the
alternative (power).

Selective inference needs the covariance of y as its `var` argument, so the
generated covariance is returned as well.
"""

import numpy as np


def _ar1_cov(n: int, rho: float = 0.5) -> np.ndarray:
    """AR(1) covariance matrix Sigma[i, j] = rho^|i-j|."""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_data(
    data_type: str,
    n_samples: int,
    n_features: int,
    seed: int,
    signal: float = 0.0,
    n_signal: int = 3,
    rho: float = 0.5,
):
    """Generate (X, y, Sigma) for one trial.

    Parameters
    ----------
    data_type : "iid" or "corr"
    n_samples : number of observations
    n_features : number of features
    seed : random seed
    signal : magnitude Delta of the true coefficients (0 = null)
    n_signal : number of leading features carrying the signal (default 3)
    rho : AR(1) correlation for the corr setting

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    Sigma : np.ndarray | float
        Covariance of y; the scalar 1.0 for iid, a matrix for corr.
    """
    rng = np.random.default_rng(seed)

    # beta = 0 under the null; otherwise Delta on the first n_signal entries
    beta = np.zeros(n_features)
    if signal != 0.0:
        k = min(n_signal, n_features)
        beta[:k] = signal

    if data_type == "iid":
        # Independent observations; the covariance is the scalar 1.0.
        X = rng.standard_normal((n_samples, n_features))
        noise = rng.standard_normal(n_samples)
        Sigma = 1.0

    elif data_type == "corr":
        # Time-series-like dependence: not only the noise but every column
        # of X follows the same AR(1) structure across rows (marginal
        # variance 1). Correlating X as well ties the two interleaved halves
        # together through the covariates, so the independence assumption of
        # data splitting genuinely fails. The proposed method treats X as a
        # fixed design and uses the exact covariance Sigma of y, so it is
        # unaffected.
        Sigma = _ar1_cov(n_samples, rho)
        L = np.linalg.cholesky(Sigma)
        X = L @ rng.standard_normal((n_samples, n_features))
        noise = rng.multivariate_normal(np.zeros(n_samples), Sigma)

    else:
        raise ValueError(f"unknown data_type: {data_type}")

    y = X @ beta + noise
    return X, y, Sigma
