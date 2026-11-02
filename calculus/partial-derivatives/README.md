# Partial Derivatives

**Level:** Beginner  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A partial derivative measures how a function of many variables changes when you nudge just one input and hold all the others fixed. A model's loss depends on millions of parameters; training needs the partial derivative of the loss with respect to each one, and stacking them gives the gradient.

## The intuition
You're baking cookies, and how tasty they are depends on two things: the amount of sugar and the oven time. You want to know: "if I add a bit more sugar, but keep the oven time exactly the same, do the cookies get better or worse, and by how much?" That one-knob-at-a-time question is a **partial derivative**.

Here's a made-up tastiness formula: tastiness $= 10s - s^2 + 3t$, where $s$ is spoonfuls of sugar and $t$ is extra minutes in the oven.

- **Sugar only** (treat $t$ as a fixed number): the slope with respect to $s$ is $10 - 2s$. At $s = 3$ that's $4$: one more spoonful adds about 4 tastiness points.
- **Oven time only** (treat $s$ as fixed): the slope with respect to $t$ is $3$, no matter what.

At $s = 5$, the sugar slope is $10 - 10 = 0$: adding sugar no longer helps. That's the sweet spot for sugar.

A neural network is like a recipe with millions of knobs (its weights). The computer works out the partial derivative of the error for every single knob, "if I turn just this one a hair, how does the error change?", and then nudges every knob a little in the helpful direction. Do that thousands of times and the network gets good.

## Core notes
- $\partial f/\partial x_i$: differentiate w.r.t. $x_i$, treat every other variable as a constant.
- Notation: $\frac{\partial f}{\partial x}$, $f_x$, $\partial_xf$, $\partial_if$.
- Stack all partials into a vector to get the gradient $\nabla f = (\partial_1f,\dots,\partial_nf)$ (Gradients).
- Directional derivative along unit $\mathbf{u}$: $D_\mathbf{u}f = \nabla f\cdot\mathbf{u}$.
- Second partials $\partial^2f/\partial x_i\partial x_j$ form the Hessian; mixed partials commute for smooth $f$ (Jacobian and Hessian).
- Partials existing ≠ differentiable; continuous partials ⇒ differentiable.
- **Partial vs total derivative**: in a graph where $w$ affects $\mathcal{L}$ through several paths, the total derivative sums all paths (Chain Rule).
- A stationary point needs **every** partial to be zero.
- Saliency maps = partial derivatives of a class score w.r.t. input pixels (Gradient-Based Attribution).
- Autodiff frameworks compute all partials of a scalar loss in one backward pass (Automatic Differentiation).

## The math
**Definition.** For $f:\mathbb{R}^n\to\mathbb{R}$ and standard basis vector $\mathbf{e}_i$,
$$
\frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h\to0}\frac{f(\mathbf{x}+h\mathbf{e}_i)-f(\mathbf{x})}{h}.
$$

**Example.** $f(s,t) = 10s-s^2+3t$: $\partial f/\partial s = 10-2s$, $\partial f/\partial t = 3$, stationary in $s$ at $s = 5$.

**Differentiability and directional derivatives.** If $f\in C^1$, then $f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x})+\sum_i\partial_if(\mathbf{x})h_i+o(\|\mathbf{h}\|)$ and for unit $\mathbf{u}$,
$$
D_\mathbf{u}f(\mathbf{x}) = \lim_{h\to0}\frac{f(\mathbf{x}+h\mathbf{u})-f(\mathbf{x})}{h} = \nabla f(\mathbf{x})^\top\mathbf{u}.
$$

**Schwarz/Clairaut.** If $\partial^2f/\partial x_i\partial x_j$ and $\partial^2f/\partial x_j\partial x_i$ are continuous near $\mathbf{x}$, they are equal.

**Total derivative.** If $\mathcal{L} = \mathcal{L}(w, u(w), v(w))$, then
$$
\frac{d\mathcal{L}}{dw} = \frac{\partial\mathcal{L}}{\partial w}+\frac{\partial\mathcal{L}}{\partial u}\frac{du}{dw}+\frac{\partial\mathcal{L}}{\partial v}\frac{dv}{dw}.
$$

**Logistic loss partials.** With $z = \mathbf{w}^\top\mathbf{x}+b$ and $p = \sigma(z)$, $\dfrac{\partial\ell}{\partial w_j} = (p-y)x_j$, $\dfrac{\partial\ell}{\partial b} = p-y$.

**Counterexample.** $f(x,y) = \dfrac{xy}{x^2+y^2}$, $f(0,0) = 0$: $f_x(0,0) = f_y(0,0) = 0$, yet $f(t,t) = \tfrac12$ for all $t\ne0$, so $f$ is not continuous at the origin.

## Run the code
```bash
pip install numpy torch
python partial_derivatives.py
```

## Check yourself
<details><summary>What is a partial derivative, and how does it differ from a total derivative?</summary>

A partial derivative $\partial f/\partial x_i$ measures the rate of change of $f$ when only $x_i$ varies and all other arguments are held fixed. A total derivative accounts for the fact that other variables may themselves depend on $x_i$: $df/dx_i = \partial f/\partial x_i+\sum_j(\partial f/\partial y_j)(dy_j/dx_i)$. In a neural network, a weight in an early layer affects the loss only through downstream activations, so backprop computes total derivatives by summing contributions over all paths.
</details>

<details><summary>Compute the partial derivatives of the logistic loss with respect to the weights.</summary>

With $z = \mathbf{w}^\top\mathbf{x}+b$, $p = \sigma(z)$, and $\ell = -y\log p-(1-y)\log(1-p)$: $\partial\ell/\partial p = (p-y)/(p(1-p))$, $\partial p/\partial z = p(1-p)$, $\partial z/\partial w_j = x_j$. Multiplying gives $\partial\ell/\partial w_j = (p-y)x_j$ and $\partial\ell/\partial b = p-y$. The neat cancellation is why sigmoid pairs naturally with cross-entropy; averaged over data, the gradient is $X^\top(\mathbf{p}-\mathbf{y})/n$.
</details>

<details><summary>If all partial derivatives exist at a point, is the function differentiable there?</summary>

No. $f(x,y) = xy/(x^2+y^2)$ with $f(0,0) = 0$ has $f_x = f_y = 0$ at the origin (it is zero along both axes), but along the diagonal it equals $\tfrac12$, so it isn't even continuous. Differentiability requires a linear approximation that works in every direction simultaneously. A sufficient condition is that the partials exist and are continuous in a neighbourhood ($C^1$).
</details>

<details><summary>Are input-gradient saliency maps reliable explanations?</summary>

They show local sensitivity: which pixels would change the score most for an infinitesimal change. But they are noisy (ReLU networks have piecewise-constant gradients), saturate (a feature can be important yet have near-zero gradient), can be manipulated adversarially, and some sanity checks show certain saliency methods barely change when model weights are randomised. Integrated gradients, SmoothGrad and perturbation-based or Shapley methods address some issues; treat any attribution as a hypothesis to test.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.