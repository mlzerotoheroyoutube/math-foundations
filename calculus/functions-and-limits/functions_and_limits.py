"""Functions and Limits: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

# Limit of 1/x and of (1 + 1/n)^n
for n in [10, 1_000, 100_000]:
    print(f"n={n:>7}  1/n={1/n:.6f}  (1+1/n)^n={(1 + 1/n)**n:.6f}")
print("e =", np.e)

# Softmax temperature limits
z = np.array([2.0, 1.0, 0.5])
def softmax(z, tau=1.0):
    s = z / tau; s -= s.max()
    e = np.exp(s); return e / e.sum()
for tau in [100, 1, 0.1, 0.01]:
    print(f"tau={tau:<6} softmax={softmax(z, tau).round(4)}")

# Overflow vs stable log-sum-exp
big = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore"):
    naive = np.log(np.sum(np.exp(big)))
m = big.max(); stable = m + np.log(np.sum(np.exp(big - m)))
print("naive:", naive, " stable:", round(stable, 4))

# Sigmoid saturation: derivative vanishes for large |z|
sig = lambda t: 1 / (1 + np.exp(-t))
for t in [0, 5, 10, 20]:
    print(f"z={t:>2}  sigmoid={sig(t):.8f}  slope={sig(t) * (1 - sig(t)):.2e}")

# Cancellation near a limit: (1 - cos x)/x^2 -> 1/2
x = 1e-8
print("naive:", (1 - np.cos(x)) / x**2, " stable:", 2 * np.sin(x / 2)**2 / x**2)

# --------------------------------------------------------------------------

import torch
import torch.nn.functional as F

logits = torch.tensor([200.0, -200.0])
target = torch.tensor([0.0, 1.0])
naive = -(target * torch.log(torch.sigmoid(logits)) + (1 - target) * torch.log(1 - torch.sigmoid(logits))).mean()
stable = F.binary_cross_entropy_with_logits(logits, target)
print("naive BCE:", naive.item(), "  stable BCE:", stable.item())   # inf vs 200.0
print("fp16 max:", torch.finfo(torch.float16).max, " bf16 max:", torch.finfo(torch.bfloat16).max)
