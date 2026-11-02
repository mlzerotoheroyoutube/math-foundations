# Chain Rule

**Level:** Beginner  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> The derivative of a composition is the product of the derivatives of its pieces: $(f\circ g)'(x) = f'(g(x))\,g'(x)$; with many variables, multiply Jacobians and sum over all paths. A neural network is a long composition of simple functions, so the chain rule, applied efficiently from the output backwards, is exactly backpropagation.

## The intuition
Think of gears. A small gear turns a medium gear, which turns a big gear. If the medium gear spins **3 times** for each turn of the small one, and the big gear spins **2 times** for each turn of the medium one, then the big gear spins $3\times2 = 6$ times for each turn of the small gear. Rates of change in a chain **multiply**. That's the chain rule.

A number example. You earn money from selling lemonade: each cup earns 2 dollars. Each lemon makes 3 cups. So each extra lemon earns $3\times2 = 6$ dollars. If money depends on cups, and cups depend on lemons, then "money per lemon" = "cups per lemon" × "money per cup".

With formulas: let $u = 3x$ and $y = u^2$. How fast does $y$ change when $x$ changes? $y$ changes $2u$ per unit of $u$, and $u$ changes 3 per unit of $x$, so $y$ changes $2u\times3 = 6u = 18x$ per unit of $x$. At $x = 1$, that's 18.

A neural network is a very long chain: input → layer 1 → layer 2 → … → error. To find how the error changes when you tweak a weight deep inside, the computer multiplies the rates of change along the chain, starting from the error and walking backwards. That backwards walk is called **backpropagation**, and it's how every modern AI learns.

## Core notes
- Single variable: $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$; derivative of the outer at the inner, times derivative of the inner.
- Multivariable: $\frac{\partial y}{\partial x_j} = \sum_\ell\frac{\partial y}{\partial u_\ell}\frac{\partial u_\ell}{\partial x_j}$: sum over every path.
- Vector form: Jacobians multiply, $J_{f\circ g} = J_f\,J_g$ (Jacobian and Hessian).
- **Reverse mode** (backprop): multiply from the output end with VJPs; one pass gives all partials of a scalar loss.
- **Forward mode**: multiply from the input end with JVPs; efficient when inputs are few.
- Backprop stores forward activations to evaluate local derivatives: memory grows with depth × batch × width (Gradient Checkpointing).
- Long products of Jacobians explode or vanish: the root of Vanishing and Exploding Gradients.
- Residual connections add an identity path so the chain doesn't shrink to zero (Residual Connections).
- Sigmoid + cross-entropy, softmax + cross-entropy: chain rule produces the clean gradient $p-y$.
- Reparameterisation trick = chain rule through a sampling step (Variational Autoencoders).

## The math
**Single variable.** If $y = f(u)$ and $u = g(x)$:
$$
\frac{dy}{dx} = f'(g(x))\,g'(x).
$$

**Multivariable.** For $\mathbf{u} = g(\mathbf{x})\in\mathbb{R}^m$ and $y = f(\mathbf{u})\in\mathbb{R}$:
$$
\frac{\partial y}{\partial x_j} = \sum_{\ell=1}^m\frac{\partial y}{\partial u_\ell}\frac{\partial u_\ell}{\partial x_j}, \qquad \nabla_\mathbf{x}y = J_g(\mathbf{x})^\top\nabla_\mathbf{u}y .
$$

**Deep composition.** For $\mathbf{h}_\ell = f_\ell(\mathbf{h}_{\ell-1})$, $\ell = 1..L$, and scalar loss $\mathcal{L}(\mathbf{h}_L)$:
$$
\frac{\partial\mathcal{L}}{\partial\mathbf{h}_0} = J_1^\top J_2^\top\cdots J_L^\top\,\nabla_{\mathbf{h}_L}\mathcal{L},\qquad J_\ell = \frac{\partial\mathbf{h}_\ell}{\partial\mathbf{h}_{\ell-1}} .
$$
Reverse mode evaluates this right-to-left as a sequence of VJPs, never forming any $J_\ell$ explicitly. Norm bound: $\big\|\partial\mathcal{L}/\partial\mathbf{h}_0\big\|\le\prod_\ell\|J_\ell\|_2\,\|\nabla_{\mathbf{h}_L}\mathcal{L}\|$.

**Residual block.** $\mathbf{h}_\ell = \mathbf{h}_{\ell-1}+F_\ell(\mathbf{h}_{\ell-1})\Rightarrow J_\ell = I+\partial F_\ell$, so the gradient contains an identity term that passes through all layers unattenuated.

**Reparameterisation.** If $\mathbf{z} = \boldsymbol\mu_\theta+\boldsymbol\sigma_\theta\odot\boldsymbol\epsilon$ with $\boldsymbol\epsilon\sim\mathcal{N}(0,I)$:
$$
\nabla_\theta\,\mathbb{E}_{\boldsymbol\epsilon}\big[f(\mathbf{z})\big] = \mathbb{E}_{\boldsymbol\epsilon}\Big[\Big(\frac{\partial\mathbf{z}}{\partial\theta}\Big)^\top\nabla_\mathbf{z}f(\mathbf{z})\Big].
$$

**Itô's lemma** (for intuition): for $dX_t = \mu\,dt+\sigma\,dW_t$, $df(X_t) = \big(f'\mu+\tfrac12f''\sigma^2\big)dt+f'\sigma\,dW_t$; the extra $\tfrac12f''\sigma^2$ term is what distinguishes stochastic calculus.

## Run the code
```bash
pip install numpy torch
python chain_rule.py
```

## Check yourself
<details><summary>Explain backpropagation in terms of the chain rule. Why is it efficient?</summary>

A network computes a loss through a chain of operations; the chain rule says the derivative of the loss w.r.t. any intermediate is the product of local derivatives along the paths to the loss, summed over paths. Backprop evaluates these products starting from the loss and moving backwards, so each node's adjoint $\partial L/\partial u$ is computed once and reused by everything upstream. Because the output is a scalar, one reverse pass gives the partials for all $N$ parameters at a cost of a few forward passes, versus $N$ passes for finite differences or forward mode.
</details>

<details><summary>When would you prefer forward-mode autodiff over reverse mode?</summary>

Forward mode computes Jacobian-vector products, one input direction per pass, so it is efficient when the function has few inputs and many outputs, e.g. sensitivities with respect to a handful of hyperparameters, or computing a full Jacobian of $\mathbb{R}^3\to\mathbb{R}^{1000}$. It also needs no stored tape, saving memory. Forward-over-reverse is the standard way to get Hessian-vector products. For a scalar loss of millions of parameters, reverse mode is vastly cheaper.
</details>

<details><summary>How does the chain rule explain vanishing and exploding gradients, and how do residual connections help?</summary>

The gradient w.r.t. early layers is a product of per-layer Jacobians. If their norms are consistently below 1 (e.g. saturated sigmoids with slope at most 0.25) the product decays exponentially with depth; if above 1, it explodes. A residual block computes $\mathbf{h}+F(\mathbf{h})$, whose Jacobian is $I+\partial F$; expanding the product gives a sum that always includes the identity path, so gradients reach early layers without being multiplied by many small factors.
</details>

<details><summary>What is the reparameterisation trick, and why is it a chain-rule trick?</summary>

To differentiate $\mathbb{E}_{z\sim q_\theta}[f(z)]$ w.r.t. $\theta$, we can't backprop through a random sampling node. Writing $z = \mu_\theta+\sigma_\theta\epsilon$ with parameter-free noise $\epsilon\sim\mathcal{N}(0,1)$ moves the randomness to an input, so $z$ becomes a differentiable function of $\theta$ and the chain rule gives $\nabla_\theta f = f'(z)\,\partial z/\partial\theta$ inside the expectation. It yields low-variance gradient estimates and is the key to training VAEs; the alternative score-function estimator works for discrete variables but has higher variance.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.