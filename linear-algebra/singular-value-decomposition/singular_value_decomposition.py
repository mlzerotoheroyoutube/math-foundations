"""Singular Value Decomposition: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(6, 4))
U, s, Vt = np.linalg.svd(A, full_matrices=False)          # thin SVD
print("singular values:", s.round(3))
print("A = U Σ Vᵀ:", np.allclose(U @ np.diag(s) @ Vt, A))
print("σ² = eig(AᵀA):", np.allclose(np.sort(s**2), np.linalg.eigvalsh(A.T @ A)))

# Eckart-Young on a low-rank-plus-noise "image"
m, n, true_rank = 200, 150, 5
img = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n)) + 0.1 * rng.normal(size=(m, n))
U, s, Vt = np.linalg.svd(img, full_matrices=False)
for k in [1, 5, 20]:
    approx = (U[:, :k] * s[:k]) @ Vt[:k]
    err = np.linalg.norm(img - approx)
    print(f"k={k:2d} error={err:8.3f}  sqrt(sum tail σ²)={np.sqrt((s[k:]**2).sum()):8.3f}  "
          f"storage {k*(m+n+1)/(m*n):.1%}")

# PCA via SVD matches sklearn
from sklearn.decomposition import PCA
X = rng.normal(size=(300, 5)) @ rng.normal(size=(5, 5))
Xc = X - X.mean(0)
_, s, Vt = np.linalg.svd(Xc, full_matrices=False)
pca = PCA(n_components=2).fit(X)
print("explained var ratio:", (s[:2]**2 / (s**2).sum()).round(4), pca.explained_variance_ratio_.round(4))
print("components match up to sign:", np.allclose(np.abs(Vt[:2]), np.abs(pca.components_), atol=1e-8))

# --------------------------------------------------------------------------

import numpy as np, torch
from sklearn.utils.extmath import randomized_svd

rng = np.random.default_rng(0)
# Randomised SVD on a large matrix with decaying spectrum
M = rng.normal(size=(3000, 50)) @ np.diag(0.8 ** np.arange(50)) @ rng.normal(size=(50, 2000))
U, s, Vt = randomized_svd(M, n_components=10, n_oversamples=10, n_iter=2, random_state=0)
s_exact = np.linalg.svd(M, compute_uv=False)[:10]
print("randomised vs exact top-10 σ (max rel err):", np.max(np.abs(s - s_exact) / s_exact))

# Compress a linear layer: W (512x512) -> two thin layers of rank 32
torch.manual_seed(0)
W = torch.randn(512, 64) @ torch.randn(64, 512) / 8 + 0.01 * torch.randn(512, 512)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
k = 32
first, second = torch.diag(S[:k]) @ Vh[:k], U[:, :k]       # x -> second @ (first @ x)
x = torch.randn(512)
rel = torch.norm(W @ x - second @ (first @ x)) / torch.norm(W @ x)
print(f"params {first.numel() + second.numel()} vs {W.numel()}, relative output error {rel:.3f}")
