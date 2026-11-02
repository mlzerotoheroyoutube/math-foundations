"""Systems of Linear Equations: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

# Fruit example: 2a + b = 5, a + 3b = 5
A = np.array([[2.0, 1.0], [1.0, 3.0]]); b = np.array([5.0, 5.0])
print("apple, banana =", np.linalg.solve(A, b))   # [2, 1]

def gauss_solve(A, b):
    """Gaussian elimination with partial pivoting + back substitution."""
    A = A.astype(float).copy(); b = b.astype(float).copy(); n = len(b)
    for k in range(n - 1):
        p = k + np.argmax(np.abs(A[k:, k]))          # pivot row
        swap, back = np.array([k, p]), np.array([p, k])
        A[swap], b[swap] = A[back], b[back]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]; b[i] -= m * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x

rng = np.random.default_rng(0)
M = rng.normal(size=(50, 50)); y = rng.normal(size=50)
print("matches np.linalg.solve:", np.allclose(gauss_solve(M, y), np.linalg.solve(M, y)))

# Overdetermined: least squares = linear regression; residual is orthogonal to columns
X = np.c_[np.ones(100), rng.normal(size=(100, 2))]
t = X @ np.array([1.0, 2.0, -3.0]) + 0.1 * rng.normal(size=100)
w_ne = np.linalg.solve(X.T @ X, X.T @ t)             # normal equations
w_ls, *_ = np.linalg.lstsq(X, t, rcond=None)         # SVD-based, preferred
print(w_ls.round(3), np.allclose(w_ne, w_ls), "X^T r ≈ 0:", np.abs(X.T @ (t - X @ w_ls)).max() < 1e-9)

# Underdetermined: GD from zero converges to the minimum-norm solution A^+ b
W = rng.normal(size=(5, 20)); c = rng.normal(size=5)
x = np.zeros(20)
for _ in range(5000):
    x -= 0.01 * W.T @ (W @ x - c)
print("GD == pinv solution:", np.allclose(x, np.linalg.pinv(W) @ c, atol=1e-6))

# --------------------------------------------------------------------------

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

# Large sparse SPD system (1-D Laplacian + shift) solved with conjugate gradient
n = 10_000
L = diags([-1.0, 2.01, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
rhs = np.ones(n)
sol, info = cg(L, rhs, maxiter=2000)
print("converged:", info == 0, " residual:", np.linalg.norm(L @ sol - rhs) / np.linalg.norm(rhs))
