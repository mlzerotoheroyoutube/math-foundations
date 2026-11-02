"""Taylor Series: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np
from math import factorial

x = 0.1
for k in range(1, 5):
    approx = sum(x**j / factorial(j) for j in range(k + 1))
    print(f"order {k}: {approx:.8f}  error {abs(approx - np.exp(x)):.1e}")

# Error shrinks like h^(k+1): halving h cuts 2nd-order error by ~8
f, df, d2f = np.sin, np.cos, lambda t: -np.sin(t)
x0 = 1.0
for h in [0.2, 0.1, 0.05]:
    err = abs(np.sin(x0 + h) - (f(x0) + df(x0) * h + 0.5 * d2f(x0) * h**2))
    print(f"h={h:<5} 2nd-order error {err:.2e}")

# log1p vs log(1+x) for tiny x
tiny = 1e-12
print("log(1+x):", np.log(1 + tiny), " log1p:", np.log1p(tiny))

# Newton = minimise the quadratic Taylor model: one step solves a quadratic exactly
A = np.array([[3.0, 1.0], [1.0, 2.0]]); b = np.array([1.0, -1.0])
grad = lambda w: A @ w - b
w = np.zeros(2)
w_newton = w - np.linalg.solve(A, grad(w))
print("Newton step:", w_newton.round(6), " exact minimiser:", np.linalg.solve(A, b).round(6))

# --------------------------------------------------------------------------

import torch
from torch.func import functional_call, jacrev

# Linearised network (first-order Taylor in parameters) vs the real network after a small update
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 256), torch.nn.Tanh(), torch.nn.Linear(256, 1))
params0 = {k: v.detach().clone() for k, v in net.named_parameters()}
x = torch.randn(8, 4)
f = lambda p: functional_call(net, p, (x,)).squeeze(-1)
J = jacrev(f)(params0)                                     # dict of Jacobians
delta = {k: 1e-2 * torch.randn_like(v) for k, v in params0.items()}
lin = f(params0) + sum((J[k].reshape(8, -1) @ delta[k].reshape(-1)) for k in params0)
real = f({k: params0[k] + delta[k] for k in params0})
print("max |real - linearised|:", (real - lin).abs().max().item(), " change size:", (real - f(params0)).abs().max().item())
