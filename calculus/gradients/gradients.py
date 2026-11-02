"""Gradients: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

f = lambda v: v[0]**2 + 2 * v[1]**2
grad = lambda v: np.array([2 * v[0], 4 * v[1]])
p = np.array([1.0, 1.0])
print("gradient at (1,1):", grad(p), " length:", round(np.linalg.norm(grad(p)), 3))

# Steepest ascent: the gradient direction beats random unit directions
rng = np.random.default_rng(0)
g = grad(p); h = 1e-6
best_random = max((f(p + h * u) - f(p)) / h for u in (lambda U: U / np.linalg.norm(U, axis=1, keepdims=True))(rng.normal(size=(1000, 2))))
print("slope along gradient:", round((f(p + h * g / np.linalg.norm(g)) - f(p)) / h, 4), " best of 1000 random:", round(best_random, 4))

# Gradient descent on the bowl
x = np.array([1.0, 1.0])
for _ in range(100):
    x -= 0.1 * grad(x)
print("after GD:", x.round(6))

# Minibatch gradient: unbiased, variance ~ 1/B
X = rng.normal(size=(10_000, 5)); y = X @ np.arange(1.0, 6.0) + rng.normal(size=10_000)
w = np.zeros(5)
full = 2 * X.T @ (X @ w - y) / len(y)
for B in [8, 64, 512]:
    ests = []
    for _ in range(300):
        idx = rng.choice(len(y), B, replace=False)
        ests.append(2 * X[idx].T @ (X[idx] @ w - y[idx]) / B)
    ests = np.array(ests)
    print(f"B={B:4d}  bias={np.abs(ests.mean(0) - full).max():.3f}  mean var={ests.var(0).mean():.2f}")

# --------------------------------------------------------------------------

import torch

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
X, y = torch.randn(256, 20), torch.randn(256, 1)
loss = torch.nn.functional.mse_loss(model(X), y)
grads = torch.autograd.grad(loss, list(model.parameters()))
for (name, p), g in zip(model.named_parameters(), grads):
    print(f"{name:10s} param {tuple(p.shape)}  grad {tuple(g.shape)}  ‖g‖={g.norm():.4f}")
print("global grad norm:", torch.sqrt(sum(g.pow(2).sum() for g in grads)).item())

# Per-example gradients with torch.func (used by DP-SGD and data attribution)
from torch.func import functional_call, grad as fgrad, vmap
params = {k: v.detach() for k, v in model.named_parameters()}
def loss_one(params, xi, yi):
    return torch.nn.functional.mse_loss(functional_call(model, params, (xi.unsqueeze(0),)), yi.unsqueeze(0))
per_ex = vmap(fgrad(loss_one), in_dims=(None, 0, 0))(params, X, y)
print("per-example grad shape for first weight:", tuple(per_ex["0.weight"].shape))
