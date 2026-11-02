"""Scalars, Vectors, Matrices and Tensors: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

rng = np.random.default_rng(0)
s = np.float32(3.0)                     # scalar, shape ()
v = np.array([1.0, 2.0, 3.0])           # vector, shape (3,)
M = rng.normal(size=(2, 3))             # matrix, shape (2, 3)
imgs = rng.random((32, 3, 4, 4))        # batch of 32 RGB 4x4 images (N, C, H, W)
print(s.shape, v.shape, M.shape, imgs.shape, imgs.ndim)

# Broadcasting: (2,3) + (3,) -> (2,3); per-channel normalisation of images
print((M + v).shape)
mean = imgs.mean(axis=(0, 2, 3), keepdims=True)   # shape (1, 3, 1, 1)
std = imgs.std(axis=(0, 2, 3), keepdims=True)
normed = (imgs - mean) / std
print(mean.shape, normed.mean(axis=(0, 2, 3)).round(6))

# The classic (n,) vs (n,1) bug
y = np.arange(4.0)                       # (4,)
y_pred = y.reshape(-1, 1)                # (4,1)
print((y_pred - y).shape)                # (4,4) -- silently wrong!
print((y_pred.squeeze(-1) - y).shape)    # (4,)  -- intended

# einsum makes contractions explicit: batched attention scores
B, h, T, d = 2, 4, 5, 8
Q = rng.normal(size=(B, h, T, d)); K = rng.normal(size=(B, h, T, d))
S = np.einsum("bhqd,bhkd->bhqk", Q, K)
print(S.shape, np.allclose(S, Q @ K.swapaxes(-1, -2)))

# --------------------------------------------------------------------------

import torch

torch.manual_seed(0)
x = torch.randn(8, 16, 64)                 # (B, T, D)
print(x.shape, x.dtype, x.stride())
xt = x.transpose(1, 2)                     # (B, D, T): a view with new strides
print(xt.is_contiguous())                  # False
flat = xt.reshape(8, -1)                   # reshape copies if needed; .view would fail here
print(flat.shape)
print(x.to(torch.bfloat16).element_size(), "bytes per element in bf16")
