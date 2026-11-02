"""Orthogonality and Projections: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

b = np.array([3.0, 4.0]); a = np.array([1.0, 0.0])
p = (a @ b) / (a @ a) * a
print("shadow:", p, " residual:", b - p, " dot:", p @ (b - p))

def gram_schmidt(A):
    """Modified Gram-Schmidt: returns Q (orthonormal columns) and R."""
    A = A.astype(float); m, n = A.shape
    Q = np.zeros((m, n)); R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v); Q[:, j] = v / R[j, j]
    return Q, R

rng = np.random.default_rng(0)
A = rng.normal(size=(100, 4))
Q, R = gram_schmidt(A)
print("Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(4)), " A = QR:", np.allclose(Q @ R, A))

# Least squares as projection
y = A @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.1 * rng.normal(size=100)
P = Q @ Q.T
y_hat = P @ y
print("projector idempotent & symmetric:", np.allclose(P @ P, P), np.allclose(P, P.T))
print("residual orthogonal to features:", np.abs(A.T @ (y - y_hat)).max() < 1e-10)
print("same as lstsq fit:", np.allclose(y_hat, A @ np.linalg.lstsq(A, y, rcond=None)[0]))

# Orthogonal Procrustes: recover a hidden rotation between two embedding spaces
X = rng.normal(size=(500, 8))
Qtrue, _ = np.linalg.qr(rng.normal(size=(8, 8)))
Y = X @ Qtrue.T + 0.01 * rng.normal(size=(500, 8))
U, _, Vt = np.linalg.svd(Y.T @ X)
print("rotation recovered:", np.allclose(U @ Vt, Qtrue, atol=1e-2))

# --------------------------------------------------------------------------

import torch

torch.manual_seed(0)
# Concept erasure: remove a direction from activations so a linear probe can't read it
H = torch.randn(1000, 64)
u = torch.randn(64); u = u / u.norm()
labels = (H @ u > 0).float()                      # concept encoded along u
H_erased = H - (H @ u).unsqueeze(1) * u           # (I - u u^T) h
print("max |u·h| after erasure:", (H_erased @ u).abs().max().item())

W = torch.empty(256, 256)
torch.nn.init.orthogonal_(W)
print("orthogonal init: W^T W = I ?", torch.allclose(W.T @ W, torch.eye(256), atol=1e-5))
x = torch.randn(256)
print("norm preserved:", torch.isclose((W @ x).norm(), x.norm(), atol=1e-4).item())
