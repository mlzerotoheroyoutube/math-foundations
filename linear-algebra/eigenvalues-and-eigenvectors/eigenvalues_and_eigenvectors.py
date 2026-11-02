"""Eigenvalues and Eigenvectors: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

A = np.array([[2.0, 0.0], [0.0, 1.0]])
vals, vecs = np.linalg.eigh(A)
print("eigenvalues:", vals, "\neigenvectors (columns):\n", vecs)

rng = np.random.default_rng(0)
M = rng.normal(size=(6, 6)); S = M @ M.T                 # symmetric PSD
lam, Q = np.linalg.eigh(S)
print("S = Q Λ Qᵀ:", np.allclose(Q @ np.diag(lam) @ Q.T, S),
      " tr = Σλ:", np.isclose(np.trace(S), lam.sum()),
      " det = Πλ:", np.isclose(np.linalg.det(S), lam.prod()))

def power_iteration(matvec, n, iters=500, seed=0):
    b = np.random.default_rng(seed).normal(size=n); b /= np.linalg.norm(b)
    for _ in range(iters):
        b = matvec(b); b /= np.linalg.norm(b)
    return b @ matvec(b), b

lam_max, v = power_iteration(lambda x: S @ x, 6)
print("power iteration:", round(lam_max, 6), " eigh max:", round(lam[-1], 6))

# Spectral radius decides whether repeated application explodes or vanishes
for scale in [0.9, 1.1]:
    W = rng.normal(size=(50, 50)); W *= scale / max(abs(np.linalg.eigvals(W)))
    h = rng.normal(size=50)
    for _ in range(100):
        h = W @ h
    print(f"spectral radius {scale}: |h| after 100 steps = {np.linalg.norm(h):.2e}")

# PageRank: dominant eigenvector of a column-stochastic link matrix
links = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=float)
P = links / links.sum(0); d = 0.85
G = d * P + (1 - d) / 4
w, V = np.linalg.eig(G)
pr = np.real(V[:, np.argmax(np.real(w))]); pr /= pr.sum()
print("PageRank:", pr.round(3))

# --------------------------------------------------------------------------

import torch

# Top Hessian eigenvalue of a small network's loss via power iteration on Hessian-vector products
torch.manual_seed(0)
X, y = torch.randn(256, 10), torch.randn(256, 1)
model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
params = list(model.parameters())
loss = torch.nn.functional.mse_loss(model(X), y)
grads = torch.autograd.grad(loss, params, create_graph=True)

v = [torch.randn_like(p) for p in params]
for _ in range(50):
    norm = torch.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / norm for vi in v]
    Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
    lam = sum((h * vi).sum() for h, vi in zip(Hv, v)).item()     # Rayleigh quotient
    v = [h.detach() for h in Hv]
print(f"sharpness λ_max ≈ {lam:.4f}  →  GD stable only for lr < {2 / lam:.3f}")
