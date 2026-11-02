"""Jacobian and Hessian: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

# Hessian of x^2 + 3y^2 by finite differences of the gradient
grad = lambda v: np.array([2 * v[0], 6 * v[1]])
def num_hessian(grad, x, h=1e-5):
    return np.column_stack([(grad(x + h * e) - grad(x - h * e)) / (2 * h) for e in np.eye(len(x))])
print("Hessian:\n", num_hessian(grad, np.array([1.0, 1.0])).round(6))

# Classify critical points by Hessian eigenvalues
for name, H in {"bowl": np.diag([2.0, 6.0]), "saddle": np.diag([2.0, -6.0])}.items():
    print(name, "eigenvalues:", np.linalg.eigvalsh(H))

# Softmax Jacobian = diag(p) - p p^T, checked numerically
z = np.array([1.0, 2.0, 0.5])
sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
p = sm(z)
J_num = np.column_stack([(sm(z + 1e-6 * e) - sm(z - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
print("softmax Jacobian ok:", np.allclose(J_num, np.diag(p) - np.outer(p, p), atol=1e-8))

# Max stable learning rate = 2 / lambda_max
H = np.diag([2.0, 6.0])
for lr in [0.3, 0.34]:
    x = np.array([1.0, 1.0])
    for _ in range(100):
        x = x - lr * H @ x
    print(f"lr={lr} (limit {2/6:.3f}): |x| = {np.linalg.norm(x):.2e}")

# --------------------------------------------------------------------------

import torch
from torch.func import jacrev, hessian

torch.manual_seed(0)
W = torch.randn(2, 3)
f = lambda x: torch.tanh(W @ x)                   # R^3 -> R^2
x = torch.randn(3)
print("Jacobian shape:", jacrev(f)(x).shape)      # (2, 3)

L = lambda v: (v ** 4).sum() + v[0] * v[1]
v = torch.randn(3)
H = hessian(L)(v)
print("Hessian symmetric:", torch.allclose(H, H.T))

# Hessian-vector product without forming H (Pearlmutter)
v.requires_grad_(True)
g = torch.autograd.grad(L(v), v, create_graph=True)[0]
u = torch.randn(3)
Hu = torch.autograd.grad(g @ u, v)[0]
print("HVP matches H @ u:", torch.allclose(Hu, H @ u, atol=1e-5))

# Hutchinson trace estimate
zs = torch.randint(0, 2, (2000, 3)).float() * 2 - 1
print("trace:", torch.trace(H).item(), " Hutchinson:", torch.einsum("ni,ij,nj->n", zs, H, zs).mean().item())
