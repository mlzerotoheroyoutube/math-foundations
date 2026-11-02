# Matrix Calculus

**Level:** Advanced  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Matrix calculus is the bookkeeping for derivatives when inputs and outputs are vectors or matrices: gradients, Jacobians, and identities like $\nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x}) = (A+A^\top)\mathbf{x}$ and $\partial\mathcal{L}/\partial W = X^\top G$. It is what you need to derive backprop rules by hand, check autograd, write custom kernels and read optimisation papers.

## The intuition
You already know the slope of a simple curve: for $y = x^2$, if $x$ grows a tiny bit, $y$ grows about $2x$ times as much. Matrix calculus asks the same question when there are *lots* of inputs and outputs at once, organised in grids.

Think of a mixing desk at a concert, with 100 sliders and 10 speakers. Matrix calculus is a cheat-sheet that tells you: "if I nudge slider number 37 up a hair, how much louder does each speaker get?" Doing that for every slider and every speaker gives a whole table of "nudge effects".

Tiny example. Your score is $3a + 4b$ (two sliders $a$ and $b$). Nudge $a$ by 0.01 and the score rises by 0.03; nudge $b$ by 0.01 and it rises by 0.04. Put those effects together: $[3, 4]$. That list is called the **gradient**, and it tells you which way to push the sliders to raise the score fastest.

A neural network has millions of sliders (its weights), arranged in grids. To train it, the computer needs the "nudge effect" of every weight on the error, all at once. Matrix calculus gives compact formulas, like a grid times another grid, so a computer can calculate millions of nudge effects in one go instead of one at a time.

## Core notes
- Gradient of a scalar w.r.t. a matrix has the **same shape** as the matrix; check shapes after every derivation.
- Differentials first: write $df = \operatorname{tr}(G^\top dX)$, then $\nabla_Xf = G$.
- Key identities: $\nabla_\mathbf{x}(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$; $\nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x}) = (A+A^\top)\mathbf{x}$; $\nabla_\mathbf{x}\|A\mathbf{x}-\mathbf{b}\|^2 = 2A^\top(A\mathbf{x}-\mathbf{b})$.
- Linear layer $Y = XW$: $\nabla_W = X^\top G$, $\nabla_X = GW^\top$ (Matrix Multiplication, Backpropagation).
- Trace tricks: $\operatorname{tr}(ABC) = \operatorname{tr}(CAB)$, $\nabla_X\operatorname{tr}(AX) = A^\top$, $\nabla_X\operatorname{tr}(X^\top AX) = (A+A^\top)X$.
- $\nabla_X\log|\det X| = X^{-\top}$; $d(X^{-1}) = -X^{-1}dX\,X^{-1}$ (Matrix Inverse and Determinant).
- Softmax + cross-entropy: $\nabla_\mathbf{z} = \mathbf{p}-\mathbf{y}$ (Softmax Function, Cross-Entropy).
- Reverse-mode autodiff = chained VJPs; forward mode = chained JVPs (Automatic Differentiation).
- Always verify hand-derived gradients with finite differences or `torch.autograd.gradcheck` (in float64).
- Layout conventions differ between sources; pick one (this vault: gradients have the shape of the variable, Jacobians are $m\times n$ for $\mathbb{R}^n\to\mathbb{R}^m$).

## The math
**Gradient via differential.** For scalar $f(X)$, $X\in\mathbb{R}^{m\times n}$:
$$
df = \sum_{ij}\frac{\partial f}{\partial X_{ij}}dX_{ij} = \operatorname{tr}\big(G^\top dX\big)\ \Longleftrightarrow\ \nabla_Xf = G\in\mathbb{R}^{m\times n}.
$$

**Quadratic form.** $d(\mathbf{x}^\top A\mathbf{x}) = d\mathbf{x}^\top A\mathbf{x}+\mathbf{x}^\top A\,d\mathbf{x} = \mathbf{x}^\top(A^\top+A)\,d\mathbf{x}$, so $\nabla_\mathbf{x} = (A+A^\top)\mathbf{x}$ and the Hessian is $A+A^\top$.

**Linear layer backward.** For $Y = XW$ ($X\in\mathbb{R}^{B\times n}$, $W\in\mathbb{R}^{n\times p}$) and $G = \partial\mathcal{L}/\partial Y$: $d\mathcal{L} = \operatorname{tr}(G^\top dY) = \operatorname{tr}(G^\top X\,dW)+\operatorname{tr}(G^\top dX\,W)$, giving
$$
\nabla_W\mathcal{L} = X^\top G,\qquad \nabla_X\mathcal{L} = GW^\top .
$$

**Softmax cross-entropy.** With $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$, $p_i = e^{z_i}/\sum_je^{z_j}$, the Jacobian is $\partial\mathbf{p}/\partial\mathbf{z} = \operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$, and for $\ell = -\sum_iy_i\log p_i$ with $\sum_iy_i = 1$:
$$
\nabla_\mathbf{z}\ell = \mathbf{p}-\mathbf{y}.
$$

**Gaussian log-likelihood w.r.t. covariance.** For $\ell(\Sigma) = -\tfrac12\log\det\Sigma-\tfrac12\mathbf{x}^\top\Sigma^{-1}\mathbf{x}$:
$$
\nabla_\Sigma\ell = -\tfrac12\Sigma^{-1}+\tfrac12\Sigma^{-1}\mathbf{x}\mathbf{x}^\top\Sigma^{-1},
$$
which, summed over data and set to zero, gives the MLE $\hat\Sigma = \tfrac1N\sum_n\mathbf{x}_n\mathbf{x}_n^\top$ (Maximum Likelihood Estimation).

**Implicit function theorem.** If $F(\mathbf{z}^\star,\theta) = \mathbf{0}$ and $\partial_\mathbf{z}F$ is invertible, then $\dfrac{\partial\mathbf{z}^\star}{\partial\theta} = -\big(\partial_\mathbf{z}F\big)^{-1}\partial_\theta F$.

**Vectorisation identity.** $\operatorname{vec}(AXB) = (B^\top\otimes A)\operatorname{vec}(X)$, so the Jacobian of $X\mapsto AXB$ is $B^\top\otimes A$.

## Run the code
```bash
pip install numpy torch
python matrix_calculus.py
```

## Check yourself
<details><summary>Derive $\nabla_\mathbf{w}$ of the ridge objective and solve it.</summary>

$f(\mathbf{w}) = \|X\mathbf{w}-\mathbf{y}\|^2+\lambda\|\mathbf{w}\|^2$. The differential is $2(X\mathbf{w}-\mathbf{y})^\top X\,d\mathbf{w}+2\lambda\mathbf{w}^\top d\mathbf{w}$, so $\nabla f = 2X^\top(X\mathbf{w}-\mathbf{y})+2\lambda\mathbf{w}$. Setting it to zero gives $(X^\top X+\lambda I)\mathbf{w} = X^\top\mathbf{y}$, which has a unique solution for $\lambda>0$ because the matrix is positive definite. The Hessian $2(X^\top X+\lambda I)\succ0$ confirms it is the global minimum.
</details>

<details><summary>Show that the gradient of softmax cross-entropy w.r.t. logits is $\mathbf{p}-\mathbf{y}$.</summary>

Write $\ell = -\sum_iy_iz_i+\log\sum_je^{z_j}$ (using $\sum_iy_i = 1$). Then $\partial\ell/\partial z_k = -y_k+e^{z_k}/\sum_je^{z_j} = p_k-y_k$. This is why the softmax and the loss are fused in frameworks: the combined gradient is simple and numerically stable, whereas backpropagating through a separate softmax Jacobian $\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$ and a separate log invites overflow and cancellation.
</details>

<details><summary>For $Y = XW$, what are $\partial\mathcal{L}/\partial W$ and $\partial\mathcal{L}/\partial X$, and how do you remember them?</summary>

$\partial\mathcal{L}/\partial W = X^\top G$ and $\partial\mathcal{L}/\partial X = GW^\top$, where $G = \partial\mathcal{L}/\partial Y$. Mnemonic: shapes force the answer. $W$ is $n\times p$, $X$ is $B\times n$, $G$ is $B\times p$; the only product of $X$ and $G$ giving $n\times p$ is $X^\top G$, and the only product of $G$ and $W$ giving $B\times n$ is $GW^\top$. Formally they follow from $d\mathcal{L} = \operatorname{tr}(G^\top dY)$ and cyclic trace.
</details>

<details><summary>How would you differentiate through the solution of an optimisation problem or a fixed point?</summary>

Use the implicit function theorem rather than unrolling the solver. If $\mathbf{z}^\star(\theta)$ satisfies $F(\mathbf{z}^\star,\theta) = \mathbf{0}$ (e.g. $F = \nabla_\mathbf{z}$ of an inner objective, or $\mathbf{z}-f(\mathbf{z},\theta)$ for a fixed point), then $\partial\mathbf{z}^\star/\partial\theta = -(\partial_\mathbf{z}F)^{-1}\partial_\theta F$. In practice you never form the inverse: in the backward pass you solve a linear system with the upstream gradient using CG or a fixed-point iteration. Memory is constant in the number of solver iterations, which is the key advantage over backprop through unrolled steps.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.