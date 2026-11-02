"""Matrix Multiplication: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np, time

rng = np.random.default_rng(0)
orders = np.array([[2, 1], [0, 3]])            # customers x items
facts = np.array([[3, 5], [2, 400]])           # items x (price, calories)
print(orders @ facts)                          # [[8 410] [6 1200]]

def matmul_naive(A, B):
    m, n = A.shape; n2, p = B.shape
    assert n == n2, "inner dimensions must match"
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

A = rng.normal(size=(64, 32)); B = rng.normal(size=(32, 16))
print(np.allclose(matmul_naive(A, B), A @ B))

# Outer-product view and non-commutativity
outer_sum = sum(np.outer(A[:, k], B[k, :]) for k in range(A.shape[1]))
print(np.allclose(outer_sum, A @ B))
S, T = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
print("AB == BA?", np.allclose(S @ T, T @ S))

# Parenthesisation matters
n = 1500
M1, M2, v = rng.normal(size=(n, n)), rng.normal(size=(n, n)), rng.normal(size=n)
t = time.perf_counter(); r1 = (M1 @ M2) @ v; t1 = time.perf_counter() - t
t = time.perf_counter(); r2 = M1 @ (M2 @ v); t2 = time.perf_counter() - t
print(f"(AB)x: {t1*1e3:.1f} ms   A(Bx): {t2*1e3:.2f} ms   same: {np.allclose(r1, r2)}")

# --------------------------------------------------------------------------

import torch

torch.manual_seed(0)
X = torch.randn(32, 128, requires_grad=True)   # batch of 32, 128 features
W = torch.randn(128, 64, requires_grad=True)
Y = X @ W
G = torch.randn_like(Y)                        # pretend upstream gradient dL/dY
Y.backward(G)
print(torch.allclose(W.grad, X.T @ G, atol=1e-4), torch.allclose(X.grad, G @ W.T, atol=1e-4))

# Batched matmul: (B, m, n) @ (B, n, p)
a, b = torch.randn(4, 10, 8), torch.randn(4, 8, 6)
print((a @ b).shape, torch.allclose(a @ b, torch.einsum("bmn,bnp->bmp", a, b), atol=1e-5))
