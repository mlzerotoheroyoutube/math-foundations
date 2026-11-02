"""Probability Basics: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

# Estimate probabilities by simulation and compare with exact answers
rng = np.random.default_rng(0)
n = 200_000
rolls = rng.integers(1, 7, size=(n, 4))          # four dice per trial

p_even = np.mean(rolls[:, 0] % 2 == 0)
p_even_or_gt4 = np.mean((rolls[:, 0] % 2 == 0) | (rolls[:, 0] > 4))
p_at_least_one_six = np.mean((rolls == 6).any(axis=1))

print(f"P(even)            sim={p_even:.4f}  exact={1/2:.4f}")
print(f"P(even or >4)      sim={p_even_or_gt4:.4f}  exact={2/3:.4f}")
print(f"P(>=1 six in 4)    sim={p_at_least_one_six:.4f}  exact={1-(5/6)**4:.4f}")

# --------------------------------------------------------------------------

import numpy as np

# Why we work in log space: product of many probabilities underflows
rng = np.random.default_rng(0)
probs = rng.uniform(0.01, 0.2, size=2000)   # e.g. per-token probabilities
print("naive product :", np.prod(probs))     # 0.0 (underflow)
print("log-prob sum  :", np.sum(np.log(probs)))
