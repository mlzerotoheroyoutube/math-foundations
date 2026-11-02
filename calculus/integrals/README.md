# Integrals

**Level:** Intermediate  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> An integral adds up infinitely many infinitesimal pieces: area under a curve, total accumulated change, or an average over a continuous range. In ML, integrals are expectations, normalising constants, marginal likelihoods and posteriors; most are intractable, which is why sampling, variational inference and diffusion ODE/SDE solvers exist.

## The intuition
You drive for 2 hours at exactly 50 km/h. Distance = 50 × 2 = 100 km. Easy. But what if your speed keeps changing? Chop the trip into tiny slices of time. In each slice the speed is nearly constant, so distance ≈ speed × slice length. Add up all the slices. The thinner the slices, the more exact the total. That "add up infinitely many tiny slices" is an **integral**.

On a graph of speed over time, each slice is a thin rectangle, and adding them up gives the **area under the curve**. So "integral" and "area under a curve" mean the same thing.

Example: the area under $y = x$ from 0 to 2 is a triangle with base 2 and height 2: area $= \tfrac12\times2\times2 = 2$. Using rectangles of width 0.5 (heights at the left edges: 0, 0.5, 1, 1.5) gives $0.5\times(0 + 0.5 + 1 + 1.5) = 1.5$. With width 0.01 you'd get about 1.99. Thinner slices, closer to 2.

Integration is also the reverse of slopes: if you know the speed at every moment, integrating gives the distance travelled.

In AI, integrals answer questions like "averaging over every possible situation, how likely is this?" Probabilities spread over continuous values must add up (integrate) to exactly 1. Often the integral is too hard to compute exactly, so computers estimate it by random sampling: like guessing the area of a pond by throwing many pebbles at a field and counting how many splash.

## Core notes
- Definite integral $\int_a^bf(x)dx$ = signed area = limit of Riemann sums $\sum f(x_i)\Delta x$.
- Fundamental theorem: $\frac{d}{dx}\int_a^xf = f(x)$ and $\int_a^bf'(x)dx = f(b)-f(a)$.
- Expectation: $\mathbb{E}[g(X)] = \int g(x)p(x)dx$ (Expectation, Variance and Covariance); densities integrate to 1.
- Marginalisation: $p(x) = \int p(x,z)dz$ (Joint, Marginal and Conditional Probability).
- Gaussian integral $\int e^{-x^2/2}dx = \sqrt{2\pi}$: the source of $\mathcal{N}$'s normaliser.
- Change of variables brings in $|\det J|$ (Normalizing Flows).
- Most ML integrals are intractable: use Monte Carlo ($O(N^{-1/2})$ error, any dimension), MCMC, Laplace, or variational bounds.
- Grid quadrature cost grows exponentially with dimension (Curse of Dimensionality).
- Diffusion / flow sampling = numerically integrating an ODE/SDE (Diffusion Models, Flow Matching).
- Gradients of expectations: reparameterisation (low variance) or score function (works for discrete variables).

## The math
**Riemann definition.** For a partition $a = x_0<\cdots<x_N = b$ with mesh $\to0$ and $\xi_i\in[x_{i-1},x_i]$:
$$
\int_a^bf(x)\,dx = \lim_{N\to\infty}\sum_{i=1}^Nf(\xi_i)\,(x_i-x_{i-1}).
$$

**Fundamental theorem of calculus.** If $F' = f$ continuous on $[a,b]$, then $\int_a^bf(x)dx = F(b)-F(a)$.

**Gaussian integral.** $\displaystyle\int_{-\infty}^{\infty}e^{-x^2/2}dx = \sqrt{2\pi}$, and in $d$ dimensions $\displaystyle\int_{\mathbb{R}^d}e^{-\frac12\mathbf{x}^\top A\mathbf{x}}d\mathbf{x} = \sqrt{\frac{(2\pi)^d}{\det A}}$ for $A\succ0$.

**Expectation and Monte Carlo.** $\mathbb{E}_p[g] = \int g(\mathbf{x})p(\mathbf{x})d\mathbf{x}\approx\hat{I}_N = \frac1N\sum_{i=1}^Ng(\mathbf{x}_i)$, $\mathbf{x}_i\sim p$, with $\operatorname{Var}(\hat I_N) = \operatorname{Var}_p(g)/N$ independent of dimension.

**Importance sampling.** $\mathbb{E}_p[g] = \mathbb{E}_q\big[g(\mathbf{x})\tfrac{p(\mathbf{x})}{q(\mathbf{x})}\big]$.

**ELBO (Jensen).** $\log p(\mathbf{x}) = \log\int p(\mathbf{x},\mathbf{z})d\mathbf{z}\ge\mathbb{E}_{q(\mathbf{z})}\big[\log p(\mathbf{x},\mathbf{z})-\log q(\mathbf{z})\big]$, with gap $\operatorname{KL}(q\,\|\,p(\mathbf{z}\mid\mathbf{x}))$.

**Gradient estimators.** Pathwise: $\nabla_\theta\mathbb{E}_{\epsilon}[f(g_\theta(\epsilon))] = \mathbb{E}_\epsilon[\nabla_\theta f(g_\theta(\epsilon))]$. Score function: $\nabla_\theta\mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[f(\mathbf{x})\nabla_\theta\log p_\theta(\mathbf{x})]$.

**Probability-flow ODE sampling.** With learned velocity $\mathbf{v}_\theta$: $\mathbf{x}(0) = \mathbf{x}(1)-\int_0^1\mathbf{v}_\theta(\mathbf{x}(t),t)\,dt$, discretised by Euler $\mathbf{x}_{t-\Delta} = \mathbf{x}_t-\Delta\,\mathbf{v}_\theta(\mathbf{x}_t,t)$ (sign convention: data at $t = 0$, noise at $t = 1$).

## Run the code
```bash
pip install numpy scipy torch
python integrals.py
```

## Check yourself
<details><summary>Why is Monte Carlo integration preferred over grid quadrature in high dimensions?</summary>

A grid with $k$ points per dimension needs $k^d$ evaluations, exponential in $d$, so even $d = 20$ is infeasible. Monte Carlo error is $\sigma/\sqrt{N}$ regardless of dimension, where $\sigma^2$ is the variance of the integrand under the sampling distribution. The catch is the slow $N^{-1/2}$ rate and the need to sample from (or near) the right distribution; variance reduction (importance sampling, control variates, quasi-MC) helps.
</details>

<details><summary>Why is the marginal likelihood hard to compute, and how do VAEs get around it?</summary>

$p(\mathbf{x}) = \int p(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})d\mathbf{z}$ integrates over a high-dimensional latent space where $p(\mathbf{x}\mid\mathbf{z})$ is a neural network, so there is no closed form and naive sampling from the prior has enormous variance (most $\mathbf{z}$ explain $\mathbf{x}$ poorly). VAEs introduce an encoder $q(\mathbf{z}\mid\mathbf{x})$ and maximise the ELBO, a lower bound obtained by Jensen's inequality, whose gap is the KL to the true posterior. The reparameterisation trick makes the ELBO's gradient low-variance.
</details>

<details><summary>Compare the reparameterisation and score-function gradient estimators.</summary>

Both estimate $\nabla_\theta\mathbb{E}_{p_\theta}[f]$. Reparameterisation writes the sample as a differentiable function of parameter-free noise and differentiates through $f$: low variance, but needs a continuous, reparameterisable distribution and differentiable $f$. The score-function estimator $\mathbb{E}[f\nabla\log p_\theta]$ works for discrete variables and black-box $f$ (it's REINFORCE), but has high variance and needs baselines or control variates. In VAEs we use the former; in RL and discrete latent models the latter.
</details>

<details><summary>How is sampling from a diffusion or flow-matching model an integration problem?</summary>

The trained network defines a velocity (or score) field, and a sample is obtained by integrating the probability-flow ODE (or reverse SDE) from pure noise at $t = 1$ to data at $t = 0$. Each solver step is one network evaluation, so sample cost equals the number of integration steps. Higher-order solvers, straighter trajectories (rectified flows, flow matching with optimal-transport couplings) and distillation into few-step or one-step models (consistency models) all reduce the integration cost.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.