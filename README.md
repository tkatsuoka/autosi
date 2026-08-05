# AutoSI

**Automatic selective inference for algorithms written as ordinary NumPy-like code.**

Selective inference (SI) provides statistically valid *p*-values for hypotheses selected by applying an algorithm to the data. Until now, developing an SI procedure for a new algorithm required an expert to derive its *selection event* by hand. AutoSI removes this barrier: write the selection algorithm with AutoSI's tracked array operations, run it once on the observed data, and call `asi.inference` — the selection event is derived automatically, and selection events of any polynomial degree (any algorithm expressible through rational functions of the data) are supported.

**[Documentation](https://tkatsuoka.github.io/autosi/)**

## Installation

```bash
pip install autosi
```

## Quick example

Select the feature most correlated with the response, then test its regression coefficient with a selective *p*-value:

```python
import numpy as np
import autosi as asi

rng = np.random.default_rng(0)
n, p = 100, 10
X_np = rng.normal(size=(n, p))
y_np = rng.normal(size=n)

X = asi.array(X_np)
y = asi.array(y_np)

def marginal_screening(y):
    corr = X.T @ y                    # correlation with each feature
    best = asi.argmax(asi.abs(corr))  # pick the most correlated one
    x_M = X.T[best]
    return x_M / (x_M @ x_M)          # contrast of its coefficient

eta = marginal_screening(y)
result = asi.inference(eta=eta, prob_vec=y, var=1.0,
                       algorithm=marginal_screening)
print(result.p_value)
```

The same three steps (wrap the data, write the algorithm with `asi` operations, call `asi.inference`) handle far more complex procedures — including an iterative lasso and a lasso tuned by cross-validated R², whose selection event involves polynomial inequalities of degree twelve and is beyond every existing exact SI method.

## How it works

`asi.inference` re-runs the algorithm along the one-dimensional line of datasets that the SI reduction confines the response to. Every tracked operation (comparisons, `abs`, `max`, `argmax`, `sort`, ...) records the interval of the line parameter on which its outcome is unchanged. Sweeping the line and collecting the intervals whose selection matches the observed one yields the truncation region, from which an exactly valid selective *p*-value follows via the truncated normal distribution.

## Requirements

Python >= 3.11. All dependencies ([sicore](https://pypi.org/project/sicore/), NumPy, SciPy) are installed automatically with `pip install autosi`.

## License

MIT
