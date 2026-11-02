"""Chain Rule: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

# Chain rule by hand: y = (3x)^2 -> dy/dx = 2u * 3 = 18x
x = 1.0
u = 3 * x; y = u**2
dy_dx = 2 * u * 3
h = 1e-6
print("chain rule:", dy_dx, " numeric:", round(((3 * (x + h))**2 - (3 * (x - h))**2) / (2 * h), 6))

# Manual backprop through a 2-layer net: L = 0.5 * ||W2 tanh(W1 x) - y||^2
rng = np.random.default_rng(0)
x = rng.normal(size=3); y = rng.normal(size=2)
W1 = rng.normal(size=(4, 3)); W2 = rng.normal(size=(2, 4))

def forward(W1, W2):
    a = W1 @ x; hdn = np.tanh(a); out = W2 @ hdn
    return 0.5 * np.sum((out - y) ** 2), (a, hdn, out)

L, (a, hdn, out) = forward(W1, W2)
d_out = out - y                       # dL/d out
dW2 = np.outer(d_out, hdn)            # dL/dW2
d_h = W2.T @ d_out                    # VJP through W2
d_a = d_h * (1 - np.tanh(a) ** 2)     # VJP through tanh
dW1 = np.outer(d_a, x)                # dL/dW1

num = np.zeros_like(W1)
for idx in np.ndindex(W1.shape):
    E = np.zeros_like(W1); E[idx] = h
    num[idx] = (forward(W1 + E, W2)[0] - forward(W1 - E, W2)[0]) / (2 * h)
print("manual backprop matches numeric:", np.allclose(dW1, num, atol=1e-6))

# Vanishing gradients: product of sigmoid derivatives (max 0.25) across depth
for depth in [5, 20, 50]:
    print(f"depth {depth}: upper bound on product of sigmoid slopes = {0.25**depth:.1e}")

# --------------------------------------------------------------------------

import torch

torch.manual_seed(0)
x = torch.randn(3, requires_grad=True)
W1, W2 = torch.randn(4, 3), torch.randn(2, 4)
y = torch.randn(2)
L = 0.5 * ((W2 @ torch.tanh(W1 @ x) - y) ** 2).sum()
L.backward()
print("autograd dL/dx:", x.grad)

# Forward mode (JVP) vs reverse mode (VJP) on f: R^3 -> R^2
from torch.func import jvp, vjp, jacrev
f = lambda v: W2 @ torch.tanh(W1 @ v)
v0 = torch.randn(3)
J = jacrev(f)(v0)                                   # (2, 3)
tangent = torch.tensor([1.0, 0.0, 0.0])
_, Jv = jvp(f, (v0,), (tangent,))                   # one column of J
_, pull = vjp(f, v0)
uTJ = pull(torch.tensor([1.0, 0.0]))[0]             # one row of J
print(torch.allclose(Jv, J[:, 0]), torch.allclose(uTJ, J[0]))
