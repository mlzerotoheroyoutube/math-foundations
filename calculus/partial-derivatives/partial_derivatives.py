"""Partial Derivatives: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

f = lambda s, t: 10 * s - s**2 + 3 * t
h = 1e-6
s, t = 3.0, 2.0
df_ds = (f(s + h, t) - f(s - h, t)) / (2 * h)       # nudge s only
df_dt = (f(s, t + h) - f(s, t - h)) / (2 * h)       # nudge t only
print(f"∂f/∂s={df_ds:.4f} (exact {10 - 2*s}),  ∂f/∂t={df_dt:.4f} (exact 3)")

# Logistic regression partials: (p - y) * x_j, checked numerically
rng = np.random.default_rng(0)
x = rng.normal(size=4); y = 1.0; w = rng.normal(size=4)
sig = lambda z: 1 / (1 + np.exp(-z))
loss = lambda w_: -(y * np.log(sig(w_ @ x)) + (1 - y) * np.log(1 - sig(w_ @ x)))
analytic = (sig(w @ x) - y) * x
numeric = np.array([(loss(w + h * e) - loss(w - h * e)) / (2 * h) for e in np.eye(4)])
print("analytic:", analytic.round(6), "\nnumeric: ", numeric.round(6))

# Directional derivative = gradient · u
u = np.array([1.0, 1.0]) / np.sqrt(2)
grad = np.array([10 - 2 * s, 3.0])
dir_num = (f(s + h * u[0], t + h * u[1]) - f(s - h * u[0], t - h * u[1])) / (2 * h)
print("directional:", round(dir_num, 6), " grad·u:", round(grad @ u, 6))

# --------------------------------------------------------------------------

import torch

# Saliency map: partial derivatives of a class score w.r.t. each input pixel
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
img = torch.rand(1, 1, 28, 28, requires_grad=True)
score = model(img)[0, 3]                     # logit for class 3
score.backward()
saliency = img.grad.abs().squeeze()          # |∂ score / ∂ pixel|, shape (28, 28)
print(saliency.shape, "most influential pixel:", divmod(saliency.argmax().item(), 28))

# Partial vs total derivative: L = w*u + u^2 with u = 2w
w = torch.tensor(1.5, requires_grad=True)
u = 2 * w
L = w * u + u**2
L.backward()
print("total dL/dw:", w.grad.item(), " (partial ∂L/∂w holding u fixed would be u =", u.item(), ")")
