# Gradients

**Level:** Beginner  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> The gradient $\nabla f$ of a scalar function of many variables is the vector of all its partial derivatives. It points in the direction of steepest increase, its length is the rate of that increase, and it is perpendicular to level sets; moving against it is how virtually every model in modern ML is trained.

## The intuition
You're on a hillside with a map. At every spot, imagine an arrow painted on the ground pointing in the direction that goes **uphill fastest**, and the steeper the slope, the longer the arrow. That arrow is the **gradient**.

How do you build the arrow? Check two things separately: "how steep is it if I step east?" and "how steep is it if I step north?" Those two slopes (the partial derivatives) are the two parts of the arrow.

Example: the height of a hill is $x^2 + 2y^2$ (a bowl, really). At the spot $(1, 1)$:
- stepping east, the slope is $2x = 2$;
- stepping north, the slope is $4y = 4$.

So the gradient is the arrow $[2, 4]$: uphill is mostly north, a bit east. Its length is $\sqrt{4 + 16}\approx4.5$, which tells you how steep the steepest route is.

Two neat facts. First, if you want to go **downhill** fastest, walk the exact opposite way, $[-2, -4]$. Second, the arrow is always at right angles to the contour lines on the map (lines of equal height).

Machine learning flips the hill around: the "height" is how wrong the model is, and there are millions of directions instead of two. The computer calculates the gradient arrow, takes a small step the opposite way, and repeats. That's **gradient descent**, the training method behind almost all AI.

## Core notes
- $\nabla f(\mathbf{x}) = \big(\tfrac{\partial f}{\partial x_1},\dots,\tfrac{\partial f}{\partial x_n}\big)$; same shape as $\mathbf{x}$ (for a weight matrix, same shape as the matrix).
- Direction of **steepest ascent**; $-\nabla f$ is steepest descent; $\|\nabla f\|$ is the maximum slope.
- Perpendicular to contour lines / level sets.
- $\nabla f = \mathbf{0}$ at minima, maxima and saddles; tell them apart with the Hessian (Jacobian and Hessian).
- Directional derivative: $D_\mathbf{u}f = \nabla f\cdot\mathbf{u}$ (Vector Operations and Dot Product).
- Training loop: $\boldsymbol\theta\leftarrow\boldsymbol\theta-\eta\nabla_{\boldsymbol\theta}\mathcal{L}$ (Gradient Descent); minibatch gradients are unbiased noisy estimates (Stochastic Gradient Descent).
- Computed exactly and cheaply by Backpropagation / Automatic Differentiation.
- Must-know gradients: $\nabla(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$, $\nabla\|\mathbf{x}\|^2 = 2\mathbf{x}$, $\nabla\tfrac1n\|X\mathbf{w}-\mathbf{y}\|^2 = \tfrac2nX^\top(X\mathbf{w}-\mathbf{y})$ (Matrix Calculus).
- Monitor the global gradient norm; clip it when it spikes (Gradient Clipping).
- "Steepest" depends on the norm: sign of the gradient is steepest under $\ell_\infty$; preconditioned gradients are steepest under other geometries.

## The math
**Definition.** For $f:\mathbb{R}^n\to\mathbb{R}$,
$$
\nabla f(\mathbf{x}) = \Big(\frac{\partial f}{\partial x_1}(\mathbf{x}),\dots,\frac{\partial f}{\partial x_n}(\mathbf{x})\Big)^\top,\qquad f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x})+\nabla f(\mathbf{x})^\top\mathbf{h}+o(\|\mathbf{h}\|).
$$

**Steepest ascent.** For unit $\mathbf{u}$, $D_\mathbf{u}f = \nabla f^\top\mathbf{u}\le\|\nabla f\|_2$ (Cauchy–Schwarz), with equality iff $\mathbf{u} = \nabla f/\|\nabla f\|_2$. Under a general norm $\|\cdot\|$, the steepest-descent direction is $\arg\min_{\|\mathbf{u}\|\le1}\nabla f^\top\mathbf{u}$, which for $\ell_\infty$ is $-\operatorname{sign}(\nabla f)$ with value $-\|\nabla f\|_1$.

**Orthogonality to level sets.** If $\mathbf{c}(t)$ is a curve in $\{f = c\}$, then $\frac{d}{dt}f(\mathbf{c}(t)) = \nabla f^\top\mathbf{c}'(t) = 0$, so $\nabla f\perp\mathbf{c}'$.

**Gradient under a metric.** With inner product $\langle\mathbf{u},\mathbf{v}\rangle_M = \mathbf{u}^\top M\mathbf{v}$, $M\succ0$, the gradient is $\nabla_Mf = M^{-1}\nabla f$; natural gradient uses $M = F(\boldsymbol\theta) = \mathbb{E}\big[\nabla\log p_{\boldsymbol\theta}\nabla\log p_{\boldsymbol\theta}^\top\big]$.

**Linear regression.** $\mathcal{L}(\mathbf{w}) = \tfrac1n\|X\mathbf{w}-\mathbf{y}\|^2\Rightarrow\nabla\mathcal{L} = \tfrac2nX^\top(X\mathbf{w}-\mathbf{y})$.

**Minibatch gradient.** With batch $\mathcal{B}$ of size $B$ sampled uniformly, $\hat{\mathbf{g}} = \tfrac1B\sum_{i\in\mathcal{B}}\nabla\ell_i$ satisfies $\mathbb{E}[\hat{\mathbf{g}}] = \nabla\mathcal{L}$ and $\operatorname{Cov}(\hat{\mathbf{g}})\approx\Sigma/B$, where $\Sigma$ is the per-example gradient covariance. Gradient noise scale: $B_\text{noise} = \operatorname{tr}(\Sigma)/\|\nabla\mathcal{L}\|^2$.

**Gradient flow.** $\dot{\boldsymbol\theta}(t) = -\nabla\mathcal{L}(\boldsymbol\theta(t))$ gives $\frac{d}{dt}\mathcal{L} = -\|\nabla\mathcal{L}\|^2\le0$: loss decreases monotonically.

## Run the code
```bash
pip install numpy torch
python gradients.py
```

## Check yourself
<details><summary>Why does the gradient point in the direction of steepest ascent?</summary>

For a small step $\epsilon\mathbf{u}$ with unit $\mathbf{u}$, the first-order change is $\epsilon\nabla f^\top\mathbf{u}$. By Cauchy–Schwarz, $\nabla f^\top\mathbf{u}\le\|\nabla f\|$ with equality exactly when $\mathbf{u}$ is parallel to $\nabla f$. So among all unit directions, the gradient direction gives the largest increase, and its magnitude is that maximal rate. Note this assumes the Euclidean norm; under other norms the steepest direction changes (e.g. sign of the gradient under $\ell_\infty$).
</details>

<details><summary>Is a minibatch gradient an unbiased estimate of the full gradient? How does batch size affect it?</summary>

Yes, if examples are sampled uniformly: the expectation of the average of per-example gradients equals the full-data average. Its covariance scales roughly as $\Sigma/B$, so quadrupling the batch halves the standard deviation. Beyond the critical batch size (around the gradient noise scale $\operatorname{tr}\Sigma/\|\mathbf{g}\|^2$), the gradient is already accurate enough that larger batches give diminishing returns per example processed.
</details>

<details><summary>What does it mean that "the gradient depends on the metric"?</summary>

The derivative is a linear functional; turning it into a vector requires an inner product. Under the Euclidean inner product we get the usual partials vector; under $\langle\mathbf{u},\mathbf{v}\rangle_M$ we get $M^{-1}\nabla f$. This is why Newton's method, natural gradient and Adam can be seen as gradient descent in different geometries, and why plain gradient descent is not invariant to reparameterisation (rescaling a parameter changes its update), whereas natural gradient is.
</details>

<details><summary>Your training loss suddenly spikes. How do gradients help diagnose it?</summary>

Log the global gradient norm and per-layer norms. A spike in gradient norm just before the loss spike suggests instability: learning rate too high, a bad batch, or attention-logit growth. Mitigations: gradient clipping (e.g. at norm 1.0), lower peak learning rate or longer warmup, QK-norm or z-loss for logits, skipping bad batches, and checking for fp16 overflow. If gradients are near zero instead, look for dead units, detached graphs or saturating activations.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.