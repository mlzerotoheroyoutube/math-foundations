# Gradient Descent

**Level:** Beginner  ·  **Topic:** Math Foundations / Optimization

▶️ **Watch the video:** https://youtu.be/hWQau4xncRo

> Repeatedly nudge the parameters a small step in the direction that lowers the loss fastest (the negative gradient). It is the engine underneath almost every trained model, from linear regression to GPT.

## The intuition
Imagine you are standing on a foggy hillside at night and want to get to the lowest point of the valley. You can't see the whole landscape, but you can feel the ground under your feet. So you feel which way slopes downhill most steeply, take one step that way, and repeat. Eventually you end up at the bottom (or at least in *a* dip).

That's gradient descent. The "hill" is a **loss**: a number that says how wrong your model is. The "position" is the model's settings (its **parameters**). The "slope under your feet" is the **gradient**, which calculus gives us for free. The **step size** (called the *learning rate*) matters a lot: tiny steps take forever; giant steps make you leap over the valley and land on the other hillside, maybe even higher than before.

Concrete example: you're guessing a number $w$ and your error is $(w-3)^2$. At $w=0$ the slope is $-6$ (downhill is to the right). With a step size of $0.1$ you move to $w = 0 - 0.1\times(-6) = 0.6$. Next slope is $-4.8$, so you move to $1.08$… and you creep toward $3$, the best answer.

## Core notes
- Update rule: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$; the gradient points uphill, so we subtract it.
- "Batch" GD uses the **whole dataset** per step; Stochastic Gradient Descent uses a sample, which is what deep learning actually runs.
- The **learning rate** $\eta$ is the single most important hyperparameter. Too small: slow. Too large: oscillation or divergence.
- On convex problems GD reaches the global minimum; on neural nets it reaches *a* good region, which turns out to be enough.
- Convergence speed depends on **conditioning**: elongated valleys (high $\kappa$) cause zig-zagging, which feature scaling, Momentum and Nesterov, or Adam and Adaptive Optimizers fix.
- Gradients are computed with Backpropagation / Automatic Differentiation; GD itself is only the update step.
- Stop when the gradient norm is tiny, validation loss stops improving (Early Stopping), or the budget runs out.
- GD has an **implicit regulariser** effect: which minimum it picks matters for generalisation (Implicit Regularization).

## The math
**Update rule.** For loss $\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)$:
$$
\theta_{t+1} = \theta_t - \eta\, \nabla_\theta \mathcal{L}(\theta_t)
$$

**Why the negative gradient is steepest descent.** By first-order Taylor expansion, for a unit direction $u$,
$$
\mathcal{L}(\theta + \epsilon u) \approx \mathcal{L}(\theta) + \epsilon\, \nabla \mathcal{L}(\theta)^\top u,
$$
and by Cauchy–Schwarz $\nabla\mathcal{L}^\top u \ge -\|\nabla \mathcal{L}\|$, with equality at $u = -\nabla\mathcal{L}/\|\nabla\mathcal{L}\|$.

**Descent lemma** (for $L$-smooth $f$, i.e. $\|\nabla f(x)-\nabla f(y)\| \le L\|x-y\|$):
$$
f(\theta_{t+1}) \le f(\theta_t) - \eta\left(1 - \tfrac{L\eta}{2}\right)\|\nabla f(\theta_t)\|^2,
$$
so any $\eta < 2/L$ guarantees decrease, and $\eta = 1/L$ gives $f(\theta_{t+1}) \le f(\theta_t) - \frac{1}{2L}\|\nabla f(\theta_t)\|^2$.

**Example: linear regression with MSE.** $\mathcal{L}(w) = \frac{1}{n}\|Xw - y\|^2 \Rightarrow \nabla \mathcal{L} = \frac{2}{n}X^\top(Xw - y)$. Here $L = \frac{2}{n}\lambda_{\max}(X^\top X)$ and the convergence rate depends on $\kappa(X^\top X)$.

## Run the code
```bash
pip install numpy torch
python gradient_descent.py
```

## Check yourself
<details><summary>Why do we move in the direction of the *negative* gradient?</summary>

The gradient points in the direction of steepest *increase* of the function. By a first-order Taylor expansion, the change in loss for a small step $\epsilon u$ is $\epsilon \nabla \mathcal{L}^\top u$, which is most negative when $u$ points opposite to the gradient (Cauchy–Schwarz).
</details>

<details><summary>What happens if the learning rate is too high or too low? How do you pick it?</summary>

Too low: very slow convergence and possibly getting stuck on plateaus. Too high: oscillation and divergence (loss explodes or becomes NaN). On a quadratic, divergence occurs when $\eta > 2/\lambda_{\max}$. In practice, run a log-spaced sweep or an LR range test, choose the largest value that trains stably, and add warmup + decay.
</details>

<details><summary>Batch GD vs SGD vs mini-batch: trade-offs?</summary>

Batch GD uses exact gradients but costs a full pass per step and doesn't scale. SGD uses one example: cheap and noisy, and the noise can help escape sharp minima but needs a decaying LR. Mini-batch (32–4096+) balances variance against hardware parallelism and is the standard in deep learning.
</details>

<details><summary>Does gradient descent find the global minimum of a neural network loss?</summary>

No guarantee: the loss is non-convex. In practice, overparameterised networks have many good minima connected by low-loss paths, saddle points (not bad local minima) are the main obstacle in high dimensions, and noise from SGD helps escape them. Theory (NTK, overparameterisation) gives guarantees only in restricted regimes.
</details>

<details><summary>Why does feature scaling speed up gradient descent?</summary>

Unscaled features make the loss surface an elongated ellipse (a high condition number). GD zig-zags across the narrow direction while crawling along the long one. Standardising features makes the contours closer to circular, so the gradient points more directly at the minimum.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.