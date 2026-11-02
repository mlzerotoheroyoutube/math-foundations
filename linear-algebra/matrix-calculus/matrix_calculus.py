"""Matrix Calculus: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

rng = np.random.default_rng(0)

def num_grad(f, X, eps=1e-6):
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        E = np.zeros_like(X); E[idx] = eps
        G[idx] = (f(X + E) - f(X - E)) / (2 * eps)
    return G

# 1) Quadratic form: grad = (A + A^T) x
A = rng.normal(size=(4, 4)); x = rng.normal(size=4)
print("x^T A x:", np.allclose(num_grad(lambda v: v @ A @ v, x), (A + A.T) @ x))

# 2) MSE of a linear model: grad_W = X^T (X W - Y)
X, Y, W = rng.normal(size=(10, 3)), rng.normal(size=(10, 2)), rng.normal(size=(3, 2))
f = lambda W_: 0.5 * np.sum((X @ W_ - Y) ** 2)
print("MSE wrt W:", np.allclose(num_grad(f, W), X.T @ (X @ W - Y)))

# 3) log|det| : grad = X^{-T}
M = rng.normal(size=(4, 4)) + 4 * np.eye(4)
print("log|det|:", np.allclose(num_grad(lambda Z: np.linalg.slogdet(Z)[1], M), np.linalg.inv(M).T))

# 4) Softmax cross-entropy: grad = p - y
z = rng.normal(size=5); y = np.eye(5)[2]
def ce(z_):
    zs = z_ - z_.max()
    return -(y * (zs - np.log(np.exp(zs).sum()))).sum()
p = np.exp(z - z.max()); p /= p.sum()
print("softmax CE:", np.allclose(num_grad(ce, z), p - y))

# --------------------------------------------------------------------------

import torch

# A custom autograd function with a hand-derived backward, verified by gradcheck
class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W):
        ctx.save_for_backward(X, W)
        return X @ W
    @staticmethod
    def backward(ctx, G):
        X, W = ctx.saved_tensors
        return G @ W.T, X.T @ G            # dL/dX, dL/dW

torch.manual_seed(0)
X = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
W = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
print("gradcheck:", torch.autograd.gradcheck(LinearFn.apply, (X, W)))

# Jacobian and VJP with torch.func
from torch.func import jacrev, vjp
f = lambda v: torch.tanh(W.detach().T @ v)            # R^4 -> R^3
v = torch.randn(4, dtype=torch.float64)
J = jacrev(f)(v)                                        # (3, 4)
_, pullback = vjp(f, v)
u = torch.randn(3, dtype=torch.float64)
print("VJP = u^T J:", torch.allclose(pullback(u)[0], u @ J))
