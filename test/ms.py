import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import sipy as si

rng = np.random.default_rng(0)
beta = np.zeros(10)
X = rng.normal(0, 1, (200, 10))
y = X @ beta + rng.normal(0, 1, 200)

y = si.siarray(y)
X = si.siarray(X)

corr = X.T @ y
eta = (X.T @ X).inv() @ X.T
idx = si.argsort(si.abs(corr))[-1]
# print(si.show_graph(idx))
result = si.inference(eta=eta[idx], prob_vec=y, var=1.0, model=idx)
print(result.p_value)

# 何個か試してp-valueのヒストグラムを描く
p_values = []
for i in range(10000):
    beta = np.zeros(10)
    X = rng.normal(0, 1, (200, 10))
    y = X @ beta + rng.normal(0, 1, 200)

    y = si.siarray(y)
    X = si.siarray(X)

    corr = X.T @ y
    eta = (X.T @ X).inv() @ X.T
    idx = si.argsort(si.abs(corr))[-1]  # !!!!!!!!!!!!!!!!!!!!!!!!
    # print(idx.values())
    # print(si.show_graph(idx))
    result = si.inference(eta=eta[idx], prob_vec=y, var=1.0, model=idx)
    p_values.append(result.p_value)
    print(f"Iteration {i + 1}, p-value: {result.p_value}")


plt.hist(p_values, bins=20)
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.savefig("result/p_values_hist.pdf")
