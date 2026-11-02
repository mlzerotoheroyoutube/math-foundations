# Random Variables

**Level:** Beginner  ·  **Topic:** Math Foundations / Probability & Statistics

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A random variable is a function that turns each random outcome into a number (or vector), described by a mass function, density, or CDF. Every input, label, weight initialisation, latent code and generated token in ML is modelled as one.

## The intuition
Imagine a carnival game: you spin a wheel, and depending on where it stops you win some number of tokens. The spin itself is random, but what you care about is the *number of tokens*. A **random variable** is exactly that: a rule that turns "what happened" into "a number".

Example: flip two coins. The possible outcomes are HH, HT, TH, TT. Let X = "number of heads". Then X can be 0 (TT), 1 (HT or TH), or 2 (HH). The chances are:

- X = 0 with probability 1/4
- X = 1 with probability 2/4 = 1/2
- X = 2 with probability 1/4

That little table is called the **distribution** of X. It always adds up to 1.

Some random variables can take any value in a range, like your exact height or the time until the next bus. Then the chance of hitting one exact value (exactly 170.000000 cm) is basically zero, so instead we use a **density** curve: the area under the curve between 165 and 175 cm tells you the chance of landing in that range.

In machine learning, the pixel brightness of a photo, the label "cat or dog", and the next word a chatbot picks are all random variables.

## Core notes
- **Discrete RV:** PMF $p(x)=P(X=x)$, $\sum_x p(x)=1$. Labels, tokens, counts.
- **Continuous RV:** PDF $p(x)\ge 0$, $\int p(x)\,dx = 1$, $P(a\le X\le b)=\int_a^b p(x)\,dx$. A density value can exceed 1.
- **CDF** $F(x)=P(X\le x)$ is non-decreasing, right-continuous, from 0 to 1; always exists.
- **Functions of RVs are RVs:** a loss $\ell(f_\theta(X),Y)$ is a random variable; training minimises its expectation (Empirical Risk Minimization).
- **Change of variables** multiplies by $|\det J|$: the engine of Normalizing Flows.
- **Reparameterisation trick:** write $Z=g(\epsilon;\theta)$ with fixed noise $\epsilon$ to backprop through sampling (Variational Autoencoders).
- **Inverse-CDF sampling:** $X = F^{-1}(U)$, $U\sim\mathrm{Unif}(0,1)$ (Sampling and Monte Carlo Methods).
- Random **vectors** are just multivariate RVs; their joint law is covered in Joint, Marginal and Conditional Probability.
- Summaries of an RV: Expectation, Variance and Covariance.

## The math
**Discrete and continuous laws.**
$$
P(X\in A) = \sum_{x\in A} p_X(x) \quad\text{or}\quad P(X\in A) = \int_A p_X(x)\,dx, \qquad F_X(x) = P(X\le x).
$$
For continuous $X$, $p_X(x) = F_X'(x)$ wherever the derivative exists.

**Change of variables.** If $Y=g(X)$ with $g:\mathbb{R}^d\to\mathbb{R}^d$ a differentiable bijection,
$$
p_Y(y) = p_X\big(g^{-1}(y)\big)\,\left|\det \frac{\partial g^{-1}(y)}{\partial y}\right|,
\qquad
\log p_Y(y) = \log p_X(x) - \log\left|\det \frac{\partial g(x)}{\partial x}\right|.
$$
Here $x=g^{-1}(y)$ and $\partial g/\partial x$ is the $d\times d$ Jacobian. Flows stack many such $g$ and sum the log-determinants.

**Probability integral transform.** If $F_X$ is continuous then $U=F_X(X)\sim\mathrm{Unif}(0,1)$; conversely $X=F_X^{-1}(U)$ has CDF $F_X$.

**Reparameterisation.** For $Z\sim\mathcal{N}(\mu,\sigma^2)$, write $Z=\mu+\sigma\epsilon$ with $\epsilon\sim\mathcal{N}(0,1)$, so
$$
\nabla_{\mu,\sigma}\,\mathbb{E}[f(Z)] = \mathbb{E}_{\epsilon}\big[\nabla_{\mu,\sigma} f(\mu+\sigma\epsilon)\big].
$$

## Run the code
```bash
pip install numpy scipy torch
python random_variables.py
```

## Check yourself
<details><summary>Can a probability density be greater than 1? Why is that not a contradiction?</summary>

Yes. A density is probability per unit length, not a probability. Only its integral over a set must lie in $[0,1]$. For example, $\mathrm{Unif}(0, 0.1)$ has density 10 on its support, and a Gaussian with $\sigma=0.01$ peaks near 40. This is also why a continuous model's log-likelihood can be positive.
</details>

<details><summary>Derive the density of Y = aX + b for a ≠ 0.</summary>

The inverse is $x=(y-b)/a$, with derivative $1/a$. By change of variables, $p_Y(y)=p_X\big((y-b)/a\big)\cdot\frac{1}{|a|}$. For $X\sim\mathcal{N}(0,1)$ this gives $\mathcal{N}(b,a^2)$. The $1/|a|$ factor keeps total mass 1 after stretching.
</details>

<details><summary>What is the reparameterisation trick and why is it needed?</summary>

We often need gradients of $\mathbb{E}_{z\sim q_\phi}[f(z)]$ with respect to $\phi$, but sampling is not differentiable. If we can write $z=g_\phi(\epsilon)$ with $\epsilon$ from a fixed distribution, the expectation becomes over $\epsilon$ and the gradient can move inside. This gives low-variance pathwise gradients and is how VAEs are trained. For discrete variables, alternatives are the score-function (REINFORCE) estimator or Gumbel-softmax relaxations.
</details>

<details><summary>How do you sample from an arbitrary 1-D distribution if you only have a uniform RNG?</summary>

If the CDF $F$ is invertible, return $F^{-1}(U)$ (inverse transform sampling). Otherwise use rejection sampling with a proposal that upper-bounds the target, or numerical inversion of a tabulated CDF. For discrete distributions, take the cumulative sum of probabilities and binary-search a uniform draw. For high-dimensional targets you move to MCMC.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.