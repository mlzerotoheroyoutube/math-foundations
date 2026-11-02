"""Matrix Inverse and Determinant: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

A = np.array([[2.0, 0.0], [0.0, 3.0]])
print("det:", np.linalg.det(A), " inverse:\n", np.linalg.inv(A))

# Determinant = area scale: image of the unit square under a shear+scale map
B = np.array([[2.0, 1.0], [0.5, 1.5]])
corners = B @ np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)
x, y = corners
shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
print("area of image:", shoelace, " det:", np.linalg.det(B))

# Small determinant != ill-conditioned; slogdet avoids underflow
I100 = 0.1 * np.eye(100)
print("det(0.1 I):", np.linalg.det(I100), " slogdet:", np.linalg.slogdet(I100), " cond:", np.linalg.cond(I100))

# Gaussian log-density via Cholesky (stable) vs scipy
from scipy.stats import multivariate_normal
rng = np.random.default_rng(0)
M = rng.normal(size=(5, 5)); Sigma = M @ M.T + 5 * np.eye(5); mu = np.zeros(5); xv = rng.normal(size=5)
L = np.linalg.cholesky(Sigma)
z = np.linalg.solve(L, xv - mu)                       # whitened residual
logp = -0.5 * z @ z - np.log(np.diag(L)).sum() - 2.5 * np.log(2 * np.pi)
print(np.isclose(logp, multivariate_normal(mu, Sigma).logpdf(xv)))

# Woodbury: inverse of diagonal + low-rank in O(n k^2)
n, k = 500, 5
d = rng.uniform(1, 2, n); U = rng.normal(size=(n, k))
Dinv = np.diag(1 / d)
wood = Dinv - Dinv @ U @ np.linalg.inv(np.eye(k) + U.T @ Dinv @ U) @ U.T @ Dinv
print("Woodbury correct:", np.allclose(wood, np.linalg.inv(np.diag(d) + U @ U.T)))

# --------------------------------------------------------------------------

import torch

# Affine coupling layer (RealNVP-style): triangular Jacobian -> log|det| is a cheap sum
torch.manual_seed(0)
D = 4
net = torch.nn.Sequential(torch.nn.Linear(D // 2, 16), torch.nn.Tanh(), torch.nn.Linear(16, D))

def coupling(x):
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    s, t = net(x1).chunk(2, dim=-1)
    y2 = x2 * torch.exp(s) + t
    return torch.cat([x1, y2], -1), s.sum(-1)        # log|det J| = sum of log-scales

x = torch.randn(D)
y, logdet = coupling(x)
J = torch.autograd.functional.jacobian(lambda v: coupling(v)[0], x)
print(torch.allclose(logdet, torch.linalg.slogdet(J).logabsdet, atol=1e-5))
