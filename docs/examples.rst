Examples
========

Both examples below follow the standard setting of selective inference for
feature selection: the algorithm selects a feature set, and the
least-squares coefficient of a feature in the regression on the selected
set is tested. Neither contains any inference-specific logic beyond the
final :func:`autosi.inference` call.

Lasso (coordinate descent)
--------------------------

The lasso minimizes :math:`\frac{1}{2n}\|y - X\beta\|^2 +
\lambda\|\beta\|_1` by coordinate descent. The ``if / elif / else``
branches of the soft-thresholding update are tracked automatically as the
selection event. The features with nonzero coefficients form the selected
set.

.. code-block:: python

   import numpy as np
   import autosi as asi

   rng = np.random.default_rng(0)
   n, p = 100, 10
   X_np = rng.normal(size=(n, p))
   y_np = rng.normal(size=n)

   X = asi.array(X_np)
   y = asi.array(y_np)

   lam = 0.1
   max_iter = 100
   tol = 1e-4

   def lasso(y):
       col_sq_mean = asi.sum(X**2, axis=0) / n
       coef = asi.array(np.zeros(p))
       for _ in range(max_iter):
           coef_old = coef.copy()
           for j in range(p):
               # partial residual and soft-thresholding update
               r = y - X @ coef + X[:, j] * coef[j]
               rho = X[:, j] @ r / n
               if rho > lam:
                   coef[j] = (rho - lam) / col_sq_mean[j]
               elif rho < -lam:
                   coef[j] = (rho + lam) / col_sq_mean[j]
               else:
                   coef[j] = asi.array(0.0)
           if asi.sum(asi.abs(coef - coef_old)) < tol:
               break

       active = coef != 0
       X_M = X.T[active]
       eta = (X_M @ X_M.T).inv() @ X_M  # coefficients on the selected set
       return eta[0]                    # test the first selected feature

   eta = lasso(y)
   result = asi.inference(eta=eta, prob_vec=y, var=1.0, algorithm=lasso)

   print(f"selective p = {result.p_value:.4f}, "
         f"naive p = {result.naive_p_value():.4f}")

.. code-block:: text

   selective p = 0.2349, naive p = 0.0460

Lasso tuned by cross-validated R²
---------------------------------

The practically standard pipeline: select the regularization strength
:math:`\lambda` by the average validation R² over :math:`K` folds, refit
the lasso at the chosen :math:`\lambda` on all data, and test a
coefficient on its selected feature set.

Each fold's R² is a ratio of two quantities quadratic in the line
parameter ``z``, so a :math:`K`-fold CV score is a rational function of
degree :math:`2K` over :math:`2K`, and comparing two candidates' scores
yields polynomial inequalities of degree :math:`4K` (degree 8 in this
example with :math:`K = 2`). Such selection events are beyond every
existing exact method, yet the code below simply writes the pipeline as
is — the root-finding of AutoSI handles any degree.

.. code-block:: python

   import numpy as np
   import autosi as asi

   rng = np.random.default_rng(0)
   n, p = 50, 5
   K = 2
   lambdas = [0.02, 0.1, 0.5]   # candidate grid
   max_iter = 30
   tol = 1e-3

   # Data with signal: features 0 and 1 are truly active
   X_np = rng.normal(size=(n, p))
   beta = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
   y_np = X_np @ beta + rng.normal(size=n)
   fold_indices = np.array_split(rng.permutation(n), K)

   X = asi.array(X_np)
   y = asi.array(y_np)

   def cv_lasso(y):
       def solve(X_tr, y_tr, lam, n_tr):
           # lasso by coordinate descent
           col_sq_mean = asi.sum(X_tr**2, axis=0) / n_tr
           coef = asi.array(np.zeros(p))
           for _ in range(max_iter):
               coef_old = coef.copy()
               for j in range(p):
                   r = y_tr - X_tr @ coef + X_tr[:, j] * coef[j]
                   rho = X_tr[:, j] @ r / n_tr
                   if rho > lam:
                       coef[j] = (rho - lam) / col_sq_mean[j]
                   elif rho < -lam:
                       coef[j] = (rho + lam) / col_sq_mean[j]
                   else:
                       coef[j] = asi.array(0.0)
               if asi.sum(asi.abs(coef - coef_old)) < tol:
                   break
           return coef

       # average K-fold validation R² for each candidate lambda
       scores = []
       for lam in lambdas:
           score = asi.array(0.0)
           for fold in range(K):
               tr = np.concatenate(
                   [fold_indices[j] for j in range(K) if j != fold]
               )
               va = fold_indices[fold]
               coef = solve(X[tr], y[tr], lam, len(tr))
               ss_res = asi.sum((y[va] - X[va] @ coef) ** 2)
               ss_tot = asi.sum((y[va] - asi.sum(y[va]) / len(va)) ** 2)
               score = score + 1.0 - ss_res / ss_tot
           scores.append(score / K)

       # pick the lambda with the best CV score, refit on all data
       best = asi.argmax(asi.stack(scores))
       coef = solve(X, y, lambdas[best], n)

       X_M = X.T[coef != 0]
       eta = (X_M @ X_M.T).inv() @ X_M  # coefficients on the selected set
       return eta[0]                    # test the first selected feature

   eta = cv_lasso(y)
   result = asi.inference(eta=eta, prob_vec=y, var=1.0, algorithm=cv_lasso)

   print(f"selective p = {result.p_value:.4f}")

.. code-block:: text

   selective p = 0.0348

Here a truly active feature is tested on data generated with signal, and
the selective *p*-value is small: correcting for selection does not mean
losing the ability to detect real effects.
