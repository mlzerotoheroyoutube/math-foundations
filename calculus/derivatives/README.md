# Derivatives

**Level:** Beginner  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> The derivative $f'(x)$ is the instantaneous rate of change of $f$ at $x$: the slope of the tangent line, defined as a limit of rise-over-run. It tells you which way to move an input to increase or decrease the output, which is precisely what every gradient-based learning algorithm needs.

## The intuition
A car's speedometer shows how fast your position is changing *right now*. If you drive 60 km in one hour, your *average* speed is 60 km/h, but at any given moment you might be doing 30 or 90. The speedometer reading at that instant is the **derivative** of your position.

How do we get "right now" from distances and times? Measure over shorter and shorter time gaps. Say your position (in metres) after $t$ seconds is $t^2$. At $t = 3$ you're at 9 m.

- Over the next 1 second: you reach 16 m, so speed ≈ (16 − 9)/1 = 7 m/s.
- Over the next 0.1 s: you reach 9.61 m, so speed ≈ 0.61/0.1 = 6.1 m/s.
- Over the next 0.001 s: speed ≈ 6.001 m/s.

The answers close in on **6**. That's the derivative at $t = 3$. In general, the derivative of $t^2$ is $2t$, and $2\times3 = 6$.

On a graph, the derivative is the **steepness** of the curve at a point: positive means uphill, negative means downhill, zero means flat (the top of a hill or the bottom of a valley).

That's why AI uses it. Training a model means making its error as small as possible. The derivative of the error tells the computer "if you increase this setting a little, the error goes up (or down) by this much", so it knows which way to turn every knob.

## Core notes
- Definition: $f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$; slope of the tangent; best linear approximation.
- Sign: $f'>0$ increasing, $f'<0$ decreasing, $f' = 0$ stationary point (min, max, or inflection).
- Rules: $(cf)' = cf'$, $(f+g)' = f'+g'$, product $(fg)' = f'g+fg'$, quotient, and the Chain Rule $(f\circ g)' = f'(g)\,g'$.
- Must-know derivatives: $x^n\to nx^{n-1}$; $e^x\to e^x$; $\ln x\to1/x$; $\sigma(x)\to\sigma(1-\sigma)$; $\tanh\to1-\tanh^2$; $\operatorname{ReLU}\to\mathbb{1}[x>0]$.
- Second derivative = curvature; $f''>0$ is convex (bowl), used in Newton's method (Second-Order Optimization).
- Kinks (ReLU, $|x|$, hinge): use **subgradients**; frameworks pick a convention at the kink.
- Non-differentiable discrete ops (argmax, sampling, rounding) need surrogates: straight-through, Gumbel-softmax, REINFORCE.
- Finite differences are for **checking** gradients, never for training large models (cost $O(n)$ per gradient).
- Gradient Descent uses the derivative to step downhill: $\theta\leftarrow\theta-\eta f'(\theta)$.

## The math
**Definition.**
$$
f'(x) = \frac{df}{dx} = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}.
$$

**Linearisation.** $f$ is differentiable at $x$ iff $f(x+h) = f(x)+f'(x)\,h+r(h)$ with $r(h)/h\to0$.

**Rules.** Product $(fg)' = f'g+fg'$; quotient $(f/g)' = (f'g-fg')/g^2$; chain $(f\circ g)'(x) = f'(g(x))\,g'(x)$.

**Mean value theorem.** If $f$ is continuous on $[a,b]$ and differentiable on $(a,b)$, there is $c\in(a,b)$ with $f(b)-f(a) = f'(c)(b-a)$. Consequence: $|f'|\le L$ everywhere $\Rightarrow f$ is $L$-Lipschitz.

**Subgradient** of convex $f$ at $x$: any $g$ with $f(y)\ge f(x)+g(y-x)$ for all $y$; the set is $\partial f(x)$, e.g. $\partial|x|(0) = [-1,1]$, $\partial\operatorname{ReLU}(0) = [0,1]$.

**Finite-difference error.** Taylor expansion gives
$$
\frac{f(x+h)-f(x)}{h} = f'(x)+\tfrac{h}{2}f''(\xi),\qquad \frac{f(x+h)-f(x-h)}{2h} = f'(x)+\tfrac{h^2}{6}f'''(\xi),
$$
and with machine precision $\epsilon$ the total error of the central difference is about $\tfrac{h^2}{6}|f'''|+\tfrac{\epsilon|f|}{h}$, minimised at $h\sim\epsilon^{1/3}$.

**Straight-through estimator.** Forward $y = \operatorname{round}(x)$; backward pretend $\frac{dy}{dx} = 1$ (identity), so gradients pass through quantisation.

## Run the code
```bash
pip install numpy torch
python derivatives.py
```

## Check yourself
<details><summary>Derive the derivative of the sigmoid and explain its consequence for deep networks.</summary>

$\sigma(x) = 1/(1+e^{-x})$, so $\sigma'(x) = e^{-x}/(1+e^{-x})^2 = \sigma(x)(1-\sigma(x))$. Its maximum is $0.25$ at $x = 0$, and it decays to 0 for large $|x|$. In a deep network, backprop multiplies one such factor per layer, so gradients shrink by at least a factor of 4 per sigmoid layer and vanish entirely when units saturate. This is why ReLU-family activations, careful initialisation and normalisation replaced sigmoids in hidden layers.
</details>

<details><summary>ReLU isn't differentiable at 0. Why can we still train with gradient descent?</summary>

ReLU is continuous and piecewise linear; at 0 any value in $[0,1]$ is a valid subgradient, and frameworks simply use 0 (or 1). Pre-activations landing exactly on 0 is a measure-zero event in floating point, so the choice rarely matters. Theoretically, Clarke subdifferentials and the "conservative field" framework show that SGD using autodiff outputs on such piecewise-smooth programs still converges to generalised stationary points.
</details>

<details><summary>How do you numerically check a gradient, and what step size do you use?</summary>

Use central differences $(f(x+h\mathbf{e}_i)-f(x-h\mathbf{e}_i))/(2h)$ per coordinate and compare to the analytical gradient via relative error $\|g_a-g_n\|/(\|g_a\|+\|g_n\|)$, aiming for about $10^{-7}$ or better in float64. The truncation error is $O(h^2)$ and round-off error is $O(\epsilon/h)$, so $h\approx10^{-5}$ to $10^{-6}$ is a good choice in float64; float32 is too imprecise for tight checks. Check only a small random subset of coordinates for large models, and avoid points near kinks.
</details>

<details><summary>How do you backpropagate through a non-differentiable operation like rounding or sampling?</summary>

Options: the straight-through estimator (use the true op forward but the identity's derivative backward), common in quantisation-aware training; continuous relaxations like Gumbel-softmax for categorical sampling, annealing temperature towards discrete; the reparameterisation trick for continuous distributions ($z = \mu+\sigma\epsilon$); and score-function (REINFORCE) estimators $\nabla\mathbb{E}[f] = \mathbb{E}[f\nabla\log p]$, unbiased but high-variance, used in RL and discrete latent models. Each trades bias against variance.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.