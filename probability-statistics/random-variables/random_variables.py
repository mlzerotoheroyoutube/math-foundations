"""Random Variables: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# Inverse-CDF sampling: Exponential(rate=2) from uniforms
u = rng.uniform(size=100_000)
x = -np.log(1 - u) / 2.0            # F^{-1}(u) for Exp(2)
print("sample mean", x.mean().round(4), "theory", 0.5)

# Change of variables: Y = exp(X), X ~ N(0,1)  =>  Y is log-normal
xs = rng.normal(size=100_000)
ys = np.exp(xs)
y0 = 1.5
manual_pdf = stats.norm.pdf(np.log(y0)) / y0     # p_X(g^{-1}(y)) * |d g^{-1}/dy|
print("manual pdf", round(manual_pdf, 4), "scipy", round(stats.lognorm(s=1).pdf(y0), 4))
print("empirical P(Y<1.5)", np.mean(ys < y0).round(4), "CDF", round(stats.lognorm(s=1).cdf(y0), 4))

# --------------------------------------------------------------------------

import torch

# Reparameterisation: gradients flow through a Gaussian sample
torch.manual_seed(0)
mu = torch.tensor(1.0, requires_grad=True)
log_sigma = torch.tensor(0.0, requires_grad=True)
eps = torch.randn(10_000)
z = mu + log_sigma.exp() * eps                # z ~ N(mu, sigma^2)
loss = (z ** 2).mean()                         # E[z^2] = mu^2 + sigma^2
loss.backward()
print(mu.grad.item(), log_sigma.grad.item())   # ≈ 2*mu = 2, 2*sigma^2 = 2
