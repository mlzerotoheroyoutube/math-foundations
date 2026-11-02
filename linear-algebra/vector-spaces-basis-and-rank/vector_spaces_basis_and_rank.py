"""Vector Spaces, Basis and Rank: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

rng = np.random.default_rng(0)
# Columns: maths, physics, total -> rank 2
maths, physics = rng.integers(40, 100, 6).astype(float), rng.integers(40, 100, 6).astype(float)
scores = np.c_[maths, physics, maths + physics]
print("shape:", scores.shape, " rank:", np.linalg.matrix_rank(scores))

# Null space: the 'recipe' that gives zero (1*maths + 1*physics - 1*total = 0)
U, s, Vt = np.linalg.svd(scores)
print("singular values:", s.round(6))
print("null-space direction:", (Vt[-1] / Vt[-1][0]).round(6))

# Numerical rank with noise: tiny singular values appear instead of exact zeros
noisy = scores + 1e-9 * rng.normal(size=scores.shape)
print("exact-ish rank:", np.linalg.matrix_rank(noisy), " with tol=1e-6:", np.linalg.matrix_rank(noisy, tol=1e-6))

# Low-rank ratings matrix: users x films built from 3 latent tastes
users, films, k = 200, 100, 3
R = rng.normal(size=(users, k)) @ rng.normal(size=(k, films))
s = np.linalg.svd(R, compute_uv=False)
print("rank of 200x100 ratings:", np.linalg.matrix_rank(R))
print("stable rank:", round((s**2).sum() / s[0]**2, 3),
      " effective rank:", round(np.exp(-(p := s / s.sum()) @ np.log(p + 1e-30)), 3))

# --------------------------------------------------------------------------

import torch, torch.nn as nn

class LoRALinear(nn.Module):
    """Frozen dense weight plus a trainable rank-r update B @ A."""
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False)
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))     # zero init: starts as the base model
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

torch.manual_seed(0)
layer = LoRALinear(1024, 1024, r=8)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print("trainable:", trainable, "vs full:", 1024 * 1024)       # 16384 vs 1048576
with torch.no_grad():
    layer.B.normal_()
    print("rank of update:", torch.linalg.matrix_rank(layer.B @ layer.A).item())
