"""Integrals: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np
from scipy import integrate

# Riemann sums converge to the true area (2) under y = x on [0, 2]
for n in [4, 100, 10_000]:
    xs = np.linspace(0, 2, n, endpoint=False)
    print(f"{n:>6} rectangles: {np.sum(xs * (2 / n)):.5f}")

# Gaussian integral via quadrature
val, err = integrate.quad(lambda t: np.exp(-t**2 / 2), -np.inf, np.inf)
print("∫exp(-x²/2) =", round(val, 6), " sqrt(2π) =", round(np.sqrt(2 * np.pi), 6))

# Monte Carlo expectation E[X^2] for X ~ N(0,1) (= 1), error shrinks like 1/sqrt(N)
rng = np.random.default_rng(0)
for N in [100, 10_000, 1_000_000]:
    s = rng.normal(size=N) ** 2
    print(f"N={N:>8}  estimate={s.mean():.4f}  std.err={s.std() / np.sqrt(N):.4f}")

# Monte Carlo works in high dimension where grids are hopeless: volume fraction of a ball in a cube
for d in [2, 10]:
    pts = rng.uniform(-1, 1, size=(1_000_000, d))
    print(f"d={d}: fraction inside unit ball ≈ {(np.sum(pts**2, 1) <= 1).mean():.5f}")

# Importance sampling for a tail probability P(X > 4), X ~ N(0,1)
from scipy.stats import norm
naive = (rng.normal(size=100_000) > 4).mean()
y = rng.normal(loc=4.5, size=100_000)
w = norm.pdf(y) / norm.pdf(y, loc=4.5)
print("naive:", naive, " importance:", round(np.mean((y > 4) * w), 8), " exact:", round(norm.sf(4), 8))

# --------------------------------------------------------------------------

import torch

# Reparameterised vs score-function gradient of E_{z~N(mu,1)}[z^2] wrt mu (true value 2*mu)
torch.manual_seed(0)
mu = torch.tensor(1.5, requires_grad=True)
dist = torch.distributions.Normal(mu, 1.0)
z = dist.rsample((100_000,))
g_path = torch.autograd.grad((z ** 2).mean(), mu)[0]
zs = dist.sample((100_000,))
g_score = torch.autograd.grad(((zs ** 2) * dist.log_prob(zs)).mean(), mu)[0]
print(f"true {2 * mu.item():.3f}  pathwise {g_path.item():.3f}  score-function {g_score.item():.3f}")

# Euler integration of an ODE dx/dt = -x (exact x(1) = e^{-1}), as in flow/diffusion samplers
for steps in [2, 10, 100]:
    x, dt = torch.tensor(1.0), 1.0 / steps
    for _ in range(steps):
        x = x + dt * (-x)
    print(f"{steps:>3} Euler steps: {x.item():.5f}  (exact {torch.exp(torch.tensor(-1.0)).item():.5f})")
