import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

import autosi as si

rng = np.random.default_rng(0)
beta = np.zeros(10)
X = rng.normal(0, 1, (200, 10))
y = X @ beta + rng.normal(0, 1, 200)

y = si.siarray(y)
X = si.siarray(X)

def merginal_screening(y):
    corr = X.T @ y 
    eta = (X.T @ X).inv() @ X.T
    idx = si.argsort(si.abs(corr))[-1]
    return eta[idx]

eta = merginal_screening(y)
# print(eta.data)
result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=merginal_screening)
print(result.p_value)


p_values = []

for i in range(1000):
    beta = np.zeros(10)
    X = rng.normal(0, 1, (200, 10))
    y = X @ beta + rng.normal(0, 1, 200)

    y = si.siarray(y)
    X = si.siarray(X)

    eta = merginal_screening(y)
    result = si.inference(eta=eta, prob_vec=y, var=1.0, algorithm=merginal_screening)
    p_values.append(result.p_value)
    print(f"Iteration {i + 1}, p-value: {result.p_value}")

plt.hist(p_values, bins=20)
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.savefig("result/ms_p_values_hist.pdf")