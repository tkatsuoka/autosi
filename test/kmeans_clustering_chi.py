"""Chi selective inference example: testing a difference between two clusters.

A canonical use of the Chi test (:func:`autosi.inference_chi`): cluster the data
points and ask "do two clusters really have different means?". Because each point
is a ``q``-dimensional vector, the difference of cluster means ``mu_1 - mu_2`` is a
*vector*, so its norm ``||mu_1 - mu_2||`` is a chi statistic with ``q`` degrees of
freedom -- a natural multivariate counterpart to the single-contrast Norm test.

k-means fits this rational-function SI engine because its decisions are squared
Euclidean distance comparisons (``||x - mu_c||^2``), which are polynomials in the
data and hence tracked exactly along the selection line ``y(z) = a + b z``.

Under the global null (no real cluster structure) the selective p-values are
uniform on ``[0, 1]``.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

import autosi as si

# ===== Parameters =====
n_points = 40  # number of observations to cluster
dim = 2  # dimension q of each observation (= chi degrees of freedom)
n_clusters = 2  # k
max_iter = 15  # k-means iterations
cluster_a, cluster_b = 0, 1  # the two clusters whose means are compared
var = 1.0
n_trials = 1000

rng = np.random.default_rng(0)


def kmeans(y_flat):
    """Cluster the flattened data with Lloyd's k-means and return the labels.

    The data being clustered is ``y`` itself: the flat probability vector is
    reshaped to ``(n_points, dim)``. Centroids are initialised deterministically
    to the first ``n_clusters`` points so the label identities stay stable along
    the selection line (only genuine assignment flips move the selection event).
    """
    X = y_flat.reshape(n_points, dim)
    mus = [X[c] for c in range(n_clusters)]

    labels = None
    for _ in range(max_iter):
        # Assignment step: squared distance to each centroid (rational in z).
        cols = []
        for c in range(n_clusters):
            # Tile the centroid to (n_points, dim); this sidesteps an (n, q) - (q,)
            # broadcasting path in the polynomial arithmetic.
            mu_full = si.stack([mus[c]] * n_points, axis=0)
            dev = X - mu_full
            cols.append((dev * dev).sum(axis=1))
        dists = si.stack(cols, axis=1)  # (n_points, n_clusters)
        labels = si.argmin(dists, axis=1)

        # Update step: recompute each centroid as the mean of its members.
        lab = np.asarray(labels.values()).astype(int).ravel()
        mus = [
            X[lab == c].mean(axis=0) if np.any(lab == c) else mus[c]
            for c in range(n_clusters)
        ]

    return labels


def projection_of(labels):
    """Build the chi projection for the mean difference of two clusters.

    The statistic is ``||mu_a - mu_b||``. Writing the difference as a linear map
    ``D^T v`` of the data with contrast weights ``v`` (``+1/|C_a|`` on cluster a,
    ``-1/|C_b|`` on cluster b), the tested subspace is spanned, per coordinate, by
    ``v`` placed in that coordinate block of the flattened vector. Stacking the
    normalised per-coordinate directions gives a rank-``dim`` projection ``P = U U^T``
    with ``||P y|| = ||mu_a - mu_b|| / ||v||``, so ``||P y|| / sigma ~ chi_dim``.
    """
    lab = np.asarray(labels.values()).astype(int).ravel()

    v = np.zeros(n_points)
    v[lab == cluster_a] = 1.0 / np.sum(lab == cluster_a)
    v[lab == cluster_b] = -1.0 / np.sum(lab == cluster_b)
    v = v / np.linalg.norm(v)

    # Row-major flattening maps observation i, coordinate d -> index i * dim + d.
    U = np.zeros((n_points * dim, dim))
    for d in range(dim):
        U[np.arange(n_points) * dim + d, d] = v
    return U @ U.T


# ===== Single run =====
data = rng.normal(0, 1, (n_points, dim))
y = si.siarray(data.ravel())

model = kmeans(y)
result = si.inference_chi(
    projection=projection_of(model),
    prob_vec=y,
    var=var,
    algorithm=kmeans,
    model=model,
)
print("p-value", result.p_value)


# ===== Null distribution check =====
p_values = []

for i in range(n_trials):
    data = rng.normal(0, 1, (n_points, dim))  # global null: no real clusters
    y = si.siarray(data.ravel())

    model = kmeans(y)
    result = si.inference_chi(
        projection=projection_of(model),
        prob_vec=y,
        var=var,
        algorithm=kmeans,
        model=model,
    )
    p_values.append(result.p_value)
    print(f"Iteration {i + 1}, p-value: {result.p_value}")

plt.hist(p_values, bins=20, range=(0, 1))
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.savefig("result/kmeans_chi_p_values_hist.pdf")
