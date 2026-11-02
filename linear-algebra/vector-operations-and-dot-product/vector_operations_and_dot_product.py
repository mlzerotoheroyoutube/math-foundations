"""Vector Operations and Dot Product: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

rng = np.random.default_rng(0)
x = np.array([3.0, 4.0]); y = np.array([1.0, 2.0])
print("dot:", x @ y, "manual:", sum(a * b for a, b in zip(x, y)))
print("length of x:", np.linalg.norm(x))
cos = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
print("cosine:", round(cos, 4), "angle(deg):", round(np.degrees(np.arccos(cos)), 2))

# Near-orthogonality in high dimension: cosine of random pairs shrinks like 1/sqrt(d)
for d in [3, 100, 10_000]:
    a = rng.normal(size=(1000, d)); b = rng.normal(size=(1000, d))
    cs = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"d={d:6d}  std of cosine={cs.std():.4f}  1/sqrt(d)={1/np.sqrt(d):.4f}")

# Why attention divides by sqrt(d_k)
dk = 64
q = rng.normal(size=(10000, dk)); k = rng.normal(size=(10000, dk))
raw = np.sum(q * k, 1)
print("var raw:", raw.var().round(1), " var scaled:", (raw / np.sqrt(dk)).var().round(3))

# --------------------------------------------------------------------------

import torch
import torch.nn.functional as F

torch.manual_seed(0)
docs = F.normalize(torch.randn(1000, 384), dim=-1)    # 1000 document embeddings
query = F.normalize(torch.randn(384), dim=-1)
scores = docs @ query                                 # cosine = dot on unit vectors
top = torch.topk(scores, k=5)
print(top.indices.tolist(), top.values.round(decimals=3).tolist())
print(torch.allclose(F.cosine_similarity(docs, query.unsqueeze(0)), scores, atol=1e-6))
