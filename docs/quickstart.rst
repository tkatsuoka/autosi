Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install autosi

How to use
----------

Using AutoSI is three steps, whatever the selection algorithm is:

1. wrap the observed data in tracked arrays with :func:`autosi.array`;
2. write the selection algorithm as ordinary NumPy-like code using the
   ``asi`` operations, returning the contrast vector ``eta`` of the
   quantity to test;
3. call :func:`autosi.inference` with the algorithm, ``eta``, and the
   noise (co)variance.

Example: marginal screening
---------------------------

As a minimal illustration of the workflow, the example below selects the
single feature most correlated with the response, then tests its
regression coefficient with a selective *p*-value. Marginal screening is
used here only because it is the simplest possible selection — the same
three steps handle far more complex algorithms such as the lasso and the
cross-validated lasso (see :doc:`examples`).

.. code-block:: python

   import numpy as np
   import autosi as asi

   # Data generated under the null (beta = 0)
   rng = np.random.default_rng(0)
   n, p = 100, 10
   X_np = rng.normal(size=(n, p))
   y_np = rng.normal(size=n)

   # Wrap the observed data in tracked arrays
   X = asi.array(X_np)
   y = asi.array(y_np)

   def marginal_screening(y):
       corr = X.T @ y                    # correlation with each feature
       best = asi.argmax(asi.abs(corr))  # pick the most correlated one
       x_M = X.T[best]
       eta = x_M / (x_M @ x_M)
       return eta   # contrast of the selected feature's coefficient

   eta = marginal_screening(y)
   result = asi.inference(eta=eta, prob_vec=y, var=1.0,
                          algorithm=marginal_screening)

   print(f"selective p = {result.p_value:.4f}, "
         f"naive p = {result.naive_p_value():.4f}")

.. code-block:: text

   selective p = 0.2315, naive p = 0.0403

The naive *p*-value ignores that the feature was chosen because it looked
promising in this very dataset and is spuriously small; the selective
*p*-value corrects for the selection.

How it works
------------

``asi.inference`` re-runs ``marginal_screening`` along the one-dimensional
line of datasets that the selective-inference reduction confines ``y`` to.
Every tracked operation (here ``asi.abs`` and ``asi.argmax``) records the
interval of the line parameter on which its outcome is unchanged. Sweeping
the line and collecting the intervals whose selection matches the observed
one yields the truncation region, from which the selective *p*-value
follows via the truncated normal distribution.

Writing your own algorithm
--------------------------

Any deterministic algorithm qualifies as long as it

1. performs numerical computations through the arithmetic operations of
   :class:`autosi.siarray` (``+ - * / @``, ``inv``, reductions), and
2. makes every data-dependent decision through tracked comparisons and
   selections (``< > <= >=``, :func:`autosi.abs`, :func:`autosi.max`,
   :func:`autosi.argmax`, :func:`autosi.sort`, :func:`autosi.argsort`),
   and
3. terminates for every input.

Iterative solvers with data-dependent branching are fine — see
:doc:`examples` for a coordinate-descent lasso and a lasso tuned by
cross-validated R², whose selection event is far beyond what can be
derived by hand.
