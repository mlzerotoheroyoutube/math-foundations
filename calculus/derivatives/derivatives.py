"""Derivatives: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

f = lambda t: t**2
for h in [1, 0.1, 0.001, 1e-6]:
    print(f"h={h:<7} secant slope at t=3: {(f(3 + h) - f(3)) / h:.6f}")

# Central differences and the round-off trade-off
g = np.sin; x0 = 1.0; true = np.cos(x0)
for h in [1e-2, 1e-5, 1e-8, 1e-11]:
    fwd = (g(x0 + h) - g(x0)) / h
    cen = (g(x0 + h) - g(x0 - h)) / (2 * h)
    print(f"h={h:.0e}  forward err={abs(fwd - true):.1e}  central err={abs(cen - true):.1e}")

# Complex-step derivative: no subtractive cancellation
h = 1e-20
print("complex-step err:", abs(np.imag(np.sin(x0 + 1j * h)) / h - true))

# Sigmoid derivative identity
sig = lambda t: 1 / (1 + np.exp(-t))
xs = np.linspace(-6, 6, 13)
num = (sig(xs + 1e-6) - sig(xs - 1e-6)) / 2e-6
print("sigma' = sigma(1 - sigma):", np.allclose(num, sig(xs) * (1 - sig(xs)), atol=1e-8))

# Gradient descent on (w-3)^2 using its derivative 2(w-3)
w = 0.0
for _ in range(50):
    w -= 0.1 * 2 * (w - 3)
print("minimiser found:", round(w, 6))

# --------------------------------------------------------------------------

import torch

x = torch.linspace(-3, 3, 7, requires_grad=True)
y = torch.relu(x).sum() + torch.nn.functional.gelu(x).sum()
y.backward()
print("d/dx [ReLU + GELU]:", x.grad.round(decimals=4))

# Straight-through estimator for rounding
w = torch.tensor([0.3, 1.7, -0.6], requires_grad=True)
q = w + (torch.round(w) - w).detach()      # forward: round(w); backward: identity
(q * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
print("quantised:", q.detach(), " STE grad:", w.grad)
