# Taylor Series

**Level:** Intermediate  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A Taylor series approximates a smooth function near a point by a polynomial built from its derivatives there: value, plus slope × step, plus ½ curvature × step², and so on. First- and second-order Taylor models are the hidden justification for gradient descent, Newton's method, learning-rate limits, the NTK, Laplace approximations, and second-order pruning and quantisation.

## The intuition
Suppose you want to predict where a car will be one second from now. The simplest guess: "where it is now." Better: "where it is now, plus its speed × 1 second." Even better: also account for how it's speeding up or slowing down (acceleration). Each extra piece of information makes the guess more accurate, at least for short times. That's the idea of a **Taylor series**: describe a curve near a point using its value, its slope, how its slope bends, and so on.

Try it on $e^x$ near 0. The value is 1, the slope is 1, the bending is 1, and so on. The Taylor recipe gives:
$e^x \approx 1 + x + x^2/2 + x^3/6 + \dots$

At $x = 0.1$: using just $1 + x$ gives $1.1$. Adding $x^2/2$ gives $1.105$. Adding $x^3/6$ gives $1.10517$. The real answer is $1.105171\dots$ Very close, after just a few terms!

Calculators use tricks like this to compute $\sin$, $\cos$ and $e^x$. In AI, a model's error landscape is hugely complicated, but very close to where you're standing it looks like a simple slope (a straight ramp) or a simple bowl. Gradient descent trusts the "straight ramp" picture for a small step. That's also why steps must be small: the approximation is only good nearby.

## Core notes
- 1-D: $f(x+h)\approx f(x)+f'(x)h+\tfrac12f''(x)h^2+\tfrac16f'''(x)h^3+\cdots$
- n-D to second order: $f(\mathbf{x}+\mathbf{h})\approx f+\nabla f^\top\mathbf{h}+\tfrac12\mathbf{h}^\top H\mathbf{h}$ (Jacobian and Hessian).
- Must-know series: $e^x = \sum x^k/k!$; $\log(1+x) = x-x^2/2+\dots$ ($|x|<1$); $\sin x = x-x^3/6+\dots$; $(1+x)^{-1} = 1-x+x^2-\dots$
- Error of order-$k$ truncation is $O(h^{k+1})$: accurate only for small steps.
- GD = minimise linear model + $\frac{1}{2\eta}\|\mathbf{h}\|^2$; Newton = minimise quadratic model: $\mathbf{h} = -H^{-1}\nabla f$ (Gradient Descent, Second-Order Optimization).
- Quadratic model gives the max stable learning rate $2/\lambda_{\max}$.
- Linearising a network in its parameters gives the NTK picture (Neural Tangent Kernel).
- Second-order loss expansions guide pruning (OBD/OBS) and quantisation (GPTQ) (Quantization).
- Laplace approximation = second-order expansion of the log-posterior.
- Numerics: use `log1p(x)`, `expm1(x)` for small $x$; Taylor explains why.

## The math
**Taylor's theorem (1-D, Lagrange remainder).**
$$
f(x+h) = \sum_{j=0}^{k}\frac{f^{(j)}(x)}{j!}h^j+\frac{f^{(k+1)}(\xi)}{(k+1)!}h^{k+1},\quad \xi\in(x,x+h).
$$

**Multivariate second order.**
$$
f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x})+\nabla f(\mathbf{x})^\top\mathbf{h}+\tfrac12\mathbf{h}^\top\nabla^2f(\mathbf{x})\,\mathbf{h}+O(\|\mathbf{h}\|^3).
$$

**GD as a proximal first-order step.**
$$
\boldsymbol\theta_{t+1} = \arg\min_{\boldsymbol\theta}\Big\{f(\boldsymbol\theta_t)+\nabla f(\boldsymbol\theta_t)^\top(\boldsymbol\theta-\boldsymbol\theta_t)+\tfrac{1}{2\eta}\|\boldsymbol\theta-\boldsymbol\theta_t\|^2\Big\} = \boldsymbol\theta_t-\eta\nabla f(\boldsymbol\theta_t).
$$

**Newton step.** Minimising the quadratic model: $\mathbf{h}^\star = -H^{-1}\nabla f$. **Descent lemma** ($L$-smooth $f$): $f(\mathbf{x}+\mathbf{h})\le f(\mathbf{x})+\nabla f^\top\mathbf{h}+\tfrac L2\|\mathbf{h}\|^2$.

**Optimal Brain Surgeon.** At a trained minimum ($\nabla\mathcal{L}\approx0$), $\delta\mathcal{L}\approx\tfrac12\delta\mathbf{w}^\top H\delta\mathbf{w}$. Removing weight $q$ (constraint $\mathbf{e}_q^\top\delta\mathbf{w}+w_q = 0$) costs $\dfrac{w_q^2}{2[H^{-1}]_{qq}}$ with optimal compensation $\delta\mathbf{w} = -\dfrac{w_q}{[H^{-1}]_{qq}}H^{-1}\mathbf{e}_q$.

**Linearised network (NTK regime).** $f_{\boldsymbol\theta}(\mathbf{x})\approx f_{\boldsymbol\theta_0}(\mathbf{x})+J_{\boldsymbol\theta_0}(\mathbf{x})(\boldsymbol\theta-\boldsymbol\theta_0)$: a linear model in features $J_{\boldsymbol\theta_0}(\mathbf{x})$ with kernel $JJ^\top$.

**Laplace approximation.** $\log p(\boldsymbol\theta\mid\mathcal{D})\approx\log p(\hat{\boldsymbol\theta}\mid\mathcal{D})-\tfrac12(\boldsymbol\theta-\hat{\boldsymbol\theta})^\top A(\boldsymbol\theta-\hat{\boldsymbol\theta})$, $A = -\nabla^2\log p(\hat{\boldsymbol\theta}\mid\mathcal{D})$, so $p(\boldsymbol\theta\mid\mathcal{D})\approx\mathcal{N}(\hat{\boldsymbol\theta},A^{-1})$.

**Delta method.** If $\sqrt{n}(\hat\theta-\theta)\to\mathcal{N}(0,\sigma^2)$ then $\sqrt{n}(g(\hat\theta)-g(\theta))\to\mathcal{N}(0,g'(\theta)^2\sigma^2)$.

## Run the code
```bash
pip install numpy torch
python taylor_series.py
```

## Check yourself
<details><summary>Derive gradient descent and Newton's method from Taylor expansions.</summary>

First order: $f(\boldsymbol\theta+\mathbf{h})\approx f+\nabla f^\top\mathbf{h}$ is unbounded below, so add a proximity penalty $\frac{1}{2\eta}\|\mathbf{h}\|^2$; minimising gives $\mathbf{h} = -\eta\nabla f$, i.e. gradient descent. Second order: $f+\nabla f^\top\mathbf{h}+\frac12\mathbf{h}^\top H\mathbf{h}$ is minimised (when $H\succ0$) at $\mathbf{h} = -H^{-1}\nabla f$, Newton's step. Newton converges quadratically near a minimum but needs PD curvature and is expensive, hence damping and approximations.
</details>

<details><summary>Why must learning rates be small, from a Taylor perspective?</summary>

Gradient descent trusts the first-order model, whose error grows like $\frac{L}{2}\|\mathbf{h}\|^2$ for $L$-smooth losses. The descent lemma shows the loss decreases only if $\eta<2/L$; with the quadratic model, steps larger than $2/\lambda_{\max}$ overshoot and diverge along the sharpest direction. Large steps leave the region where the local model describes the loss.
</details>

<details><summary>How does GPTQ (or OBS) use a Taylor expansion?</summary>

At a trained minimum the gradient is near zero, so the loss change from perturbing weights is approximately the quadratic $\frac12\delta\mathbf{w}^\top H\delta\mathbf{w}$. OBS uses this to pick which weight to remove and how to adjust the others to compensate. GPTQ applies the same idea layer-wise with $H\propto XX^\top$ from calibration activations, quantising weights one column at a time and updating the remaining weights to absorb the error, which makes 4-bit LLM quantisation accurate.
</details>

<details><summary>What is the "lazy training" or linearised regime?</summary>

Expanding the network to first order in its parameters around initialisation gives a linear model with fixed features (the parameter Jacobian), equivalent to kernel regression with the NTK. In the infinite-width limit under standard (NTK) parameterisation, training stays in this regime and features don't move. Real networks of practical width, especially under μP-style parameterisations, learn features, so the linearisation is a useful theoretical baseline rather than a full description.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.