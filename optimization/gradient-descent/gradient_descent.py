"""Gradient Descent: code from the ML Zero to Hero explainer.

Video: https://youtu.be/hWQau4xncRo
"""

import numpy as np

# Linear regression trained with full-batch gradient descent
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + 0.1 * rng.normal(size=200)

w = np.zeros(3)
lr = 0.1
for step in range(200):
    residual = X @ w - y
    loss = np.mean(residual ** 2)
    grad = 2 * X.T @ residual / len(y)
    w -= lr * grad
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss:.4f}")

print("learned w:", w.round(3))  # ≈ [2.0, -1.0, 0.5]

# --------------------------------------------------------------------------

# The same thing in PyTorch: autograd computes the gradient for us
import torch
Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
w = torch.zeros(3, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)   # full batch => plain gradient descent
for _ in range(200):
    opt.zero_grad()
    loss = ((Xt @ w - yt) ** 2).mean()
    loss.backward()
    opt.step()
