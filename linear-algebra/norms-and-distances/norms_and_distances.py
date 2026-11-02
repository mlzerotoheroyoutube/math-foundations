"""Norms and Distances: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

v = np.array([3.0, 4.0])
for p in [1, 2, np.inf]:
    print(f"L{p} norm:", np.linalg.norm(v, ord=p))          # 7, 5, 4

rng = np.random.default_rng(0)
A = rng.normal(size=(50, 30))
s = np.linalg.svd(A, compute_uv=False)
print("Frobenius:", np.isclose(np.linalg.norm(A), np.sqrt((s**2).sum())),
      " spectral:", np.isclose(np.linalg.norm(A, 2), s[0]),
      " nuclear:", np.isclose(np.linalg.norm(A, "nuc"), s.sum()))

# Pairwise squared distances via one matmul
X = rng.normal(size=(500, 64)); Y = rng.normal(size=(300, 64))
D2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T
D2 = np.maximum(D2, 0)
print("matches brute force:", np.allclose(D2, ((X[:, None, :] - Y[None, :, :])**2).sum(-1)))

# Distance concentration: (max - min) / min of distances to a query shrinks with dimension
for d in [2, 10, 100, 1000]:
    P = rng.random((2000, d)); q = rng.random(d)
    dist = np.linalg.norm(P - q, axis=1)
    print(f"d={d:5d}  relative contrast={(dist.max() - dist.min()) / dist.min():.3f}")

# L1 vs L2 penalties: soft-thresholding zeroes small values, L2 only shrinks
w = np.array([-2.0, -0.3, 0.05, 0.4, 1.5]); lam = 0.5
print("L1 prox:", np.sign(w) * np.maximum(np.abs(w) - lam, 0))
print("L2 prox:", (w / (1 + 2 * lam)).round(3))

# --------------------------------------------------------------------------

import torch, torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
loss = model(torch.randn(32, 10)).pow(2).mean() * 1e4          # deliberately huge loss
loss.backward()
total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # returns pre-clip norm
after = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
print(f"grad norm before {total:.1f}, after {after:.3f}")
print("spectral norm of first layer:", torch.linalg.matrix_norm(model[0].weight, ord=2).item())
