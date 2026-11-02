"""Positive Definite Matrices: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

def is_pd(A):
    try:
        np.linalg.cholesky(A); return True
    except np.linalg.LinAlgError:
        return False

print(is_pd(np.diag([2.0, 1.0])), is_pd(np.diag([2.0, -1.0])))       # bowl vs saddle
print("positive entries but not PD:", is_pd(np.array([[1.0, 2.0], [2.0, 1.0]])))

def cholesky(A):
    n = len(A); L = np.zeros_like(A, dtype=float)
    for j in range(n):
        L[j, j] = np.sqrt(A[j, j] - L[j, :j] @ L[j, :j])
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L

rng = np.random.default_rng(0)
B = rng.normal(size=(8, 5)); G = B.T @ B + 1e-3 * np.eye(5)        # Gram matrix + ridge
print("from-scratch Cholesky correct:", np.allclose(cholesky(G), np.linalg.cholesky(G)))

# Sample correlated Gaussians with Cholesky
Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
L = np.linalg.cholesky(Sigma)
samples = rng.normal(size=(100_000, 2)) @ L.T
print("empirical covariance:\n", np.cov(samples.T).round(3))

# RBF kernel: PSD in theory, numerically fragile without jitter
x = np.linspace(0, 1, 200)[:, None]
K = np.exp(-((x - x.T) ** 2) / (2 * 0.5 ** 2))
print("min eigenvalue:", np.linalg.eigvalsh(K).min(), " Cholesky ok without jitter?", is_pd(K))
print("with jitter 1e-6:", is_pd(K + 1e-6 * np.eye(200)))

# Repair an indefinite 'covariance' by clipping eigenvalues
C = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.9], [0.7, -0.9, 1.0]])
lam, Q = np.linalg.eigh(C)
C_psd = Q @ np.diag(np.clip(lam, 1e-8, None)) @ Q.T
print("eigs before:", lam.round(3), " after:", np.linalg.eigvalsh(C_psd).round(3))

# --------------------------------------------------------------------------

import torch
import torch.nn.functional as F

# Learnable PD covariance: parameterise the Cholesky factor with a positive diagonal
torch.manual_seed(0)
d = 3
raw = torch.randn(d, d, requires_grad=True)
def scale_tril(raw):
    return torch.tril(raw, -1) + torch.diag(F.softplus(torch.diagonal(raw)) + 1e-4)
target = torch.distributions.MultivariateNormal(torch.zeros(d), torch.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 0.5]]))
data = target.sample((5000,))
opt = torch.optim.Adam([raw], lr=0.05)
for _ in range(500):
    opt.zero_grad()
    nll = -torch.distributions.MultivariateNormal(torch.zeros(d), scale_tril=scale_tril(raw)).log_prob(data).mean()
    nll.backward(); opt.step()
L = scale_tril(raw).detach()
print("learned covariance:\n", (L @ L.T).round(decimals=2))
