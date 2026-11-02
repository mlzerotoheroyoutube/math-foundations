"""Linear Transformations: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

rng = np.random.default_rng(0)
A = np.array([[2.0, 1.0],
              [0.0, 1.0]])            # columns: where e1 and e2 go
print(A @ np.array([3.0, 2.0]))       # [8, 2]

# Linearity check on random vectors
x, y = rng.normal(size=2), rng.normal(size=2)
print(np.allclose(A @ (2 * x + 3 * y), 2 * (A @ x) + 3 * (A @ y)))

# Rotation preserves length; determinant = area scale
th = np.pi / 6
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
print(np.linalg.norm(R @ x).round(6) == np.linalg.norm(x).round(6), np.linalg.det(A), np.linalg.det(R).round(6))

# Stacked linear layers collapse into one matrix
W1, W2, W3 = (rng.normal(size=(4, 4)) for _ in range(3))
v = rng.normal(size=4)
print(np.allclose(W3 @ (W2 @ (W1 @ v)), (W3 @ W2 @ W1) @ v))

# SVD: rotate-scale-rotate
U, s, Vt = np.linalg.svd(A)
print("singular values:", s.round(4), np.allclose(U @ np.diag(s) @ Vt, A))

# --------------------------------------------------------------------------

import torch, torch.nn as nn

torch.manual_seed(0)
# XOR is not linearly separable: a linear model plateaus, one hidden layer + ReLU solves it
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]); y = torch.tensor([0., 1., 1., 0.])
def train(model, steps=2000):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(X).squeeze(-1), y)
        loss.backward(); opt.step()
    return loss.item()
print("linear  loss:", round(train(nn.Linear(2, 1)), 3))                                    # ~0.693
print("MLP     loss:", round(train(nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))), 3))
