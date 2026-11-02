"""Matrix Decompositions: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, cho_factor, cho_solve, qr, schur

rng = np.random.default_rng(0)
A = rng.normal(size=(5, 5))
P, L, U = lu(A)                                         # SciPy returns A = P L U
print("LU:", np.allclose(P @ L @ U, A))

# Factor once, solve many right-hand sides
lu_piv = lu_factor(A)
B = rng.normal(size=(5, 100))
print("LU reuse:", np.allclose(A @ lu_solve(lu_piv, B), B))

S = A @ A.T + 5 * np.eye(5)                             # SPD
c = cho_factor(S)
print("Cholesky solve:", np.allclose(S @ cho_solve(c, B), B))

# QR least squares vs normal equations on an ill-conditioned problem
X = np.vander(np.linspace(0, 1, 50), 12)                # badly conditioned
y = np.sin(3 * np.linspace(0, 1, 50))
Q, R = qr(X, mode="economic")
w_qr = np.linalg.solve(R, Q.T @ y)
w_ne = np.linalg.solve(X.T @ X, X.T @ y)
print(f"cond(X)={np.linalg.cond(X):.1e}  residual QR={np.linalg.norm(X @ w_qr - y):.2e}  NE={np.linalg.norm(X @ w_ne - y):.2e}")

T, Z = schur(A)
print("Schur:", np.allclose(Z @ T @ Z.T, A), " quasi-triangular T (real Schur form)")

# NMF with multiplicative updates on a non-negative "documents x words" matrix
docs = rng.poisson(lam=(rng.random((100, 3)) @ rng.random((3, 40))) * 5).astype(float)
k = 3
W, H = rng.random((100, k)), rng.random((k, 40))
for _ in range(300):
    H *= (W.T @ docs) / (W.T @ W @ H + 1e-9)
    W *= (docs @ H.T) / (W @ H @ H.T + 1e-9)
print("NMF relative error (Poisson noise sets the floor):", round(np.linalg.norm(docs - W @ H) / np.linalg.norm(docs), 3), " all non-negative:", (W >= 0).all() and (H >= 0).all())

# --------------------------------------------------------------------------

import numpy as np
from sklearn.decomposition import NMF
import torch

rng = np.random.default_rng(0)
V = rng.random((60, 4)) @ rng.random((4, 30))
model = NMF(n_components=4, init="nndsvda", max_iter=1000, random_state=0)
Wm = model.fit_transform(V)
print("sklearn NMF rel. error:", round(np.linalg.norm(V - Wm @ model.components_) / np.linalg.norm(V), 4))

# Kronecker identity used by K-FAC / Shampoo: (A ⊗ G)^{-1} vec(X) = vec(G^{-1} X A^{-1}) for symmetric A
torch.manual_seed(0)
Am = torch.randn(4, 4, dtype=torch.float64); Am = Am @ Am.T + torch.eye(4, dtype=torch.float64)
Gm = torch.randn(3, 3, dtype=torch.float64); Gm = Gm @ Gm.T + torch.eye(3, dtype=torch.float64)
grad = torch.randn(3, 4, dtype=torch.float64)                     # dL/dW, shape (out, in)
big = torch.kron(Am, Gm)                                          # 12 x 12
vecX = grad.T.reshape(-1)                                         # column-major vec
lhs = torch.linalg.solve(big, vecX)
rhs = (torch.linalg.solve(Gm, grad) @ torch.linalg.inv(Am)).T.reshape(-1)
print("Kronecker-factored solve matches:", torch.allclose(lhs, rhs))
