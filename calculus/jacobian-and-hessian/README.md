# Jacobian and Hessian

**Level:** Intermediate  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> The Jacobian collects all first partial derivatives of a vector-valued function (how every output responds to every input); the Hessian collects all second partial derivatives of a scalar function (its curvature). Jacobians drive backprop, normalizing flows and robustness analysis; Hessians govern learning-rate limits, sharpness, saddle points and second-order optimisers.

## The intuition
Imagine a sound mixer with 3 sliders (inputs) and 2 speakers (outputs). The **Jacobian** is a table with one row per speaker and one column per slider. Each entry answers: "if I push *this* slider up a hair, how much louder does *that* speaker get?" So it's a 2 × 3 table of sensitivities.

The **Hessian** answers a different question, about a single output (say, how wrong a model is). The gradient tells you the slope. The Hessian tells you how the slope itself **changes**: is the ground curving up like a bowl, curving down like a hilltop, or curving up one way and down another like a saddle? It's the "bendiness" of the landscape.

Example: the height $x^2 + 3y^2$. Its slope in $x$ is $2x$, which changes by 2 for each step in $x$; its slope in $y$ is $6y$, which changes by 6 per step in $y$. So the Hessian is the table with 2 and 6 on the diagonal (and 0 elsewhere). Both positive: it's a bowl, steeper in the $y$ direction.

Why should AI care? If the valley is very steep and narrow in one direction (a big Hessian number), taking big steps makes you bounce from wall to wall and fly off. The biggest curvature number tells you the largest safe step size. And the Jacobian's table, multiplied layer after layer, is literally how backpropagation passes error signals through a network.

## Core notes
- Jacobian of $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$: $m\times n$ matrix of first partials; row $i$ is $\nabla f_i^\top$.
- Hessian of $f:\mathbb{R}^n\to\mathbb{R}$: $n\times n$ symmetric matrix of second partials; the Jacobian of $\nabla f$.
- Chain rule: Jacobians multiply (Chain Rule); backprop = chained VJPs (Backpropagation).
- At a critical point: $H\succ0$ minimum, $H\prec0$ maximum, indefinite saddle (Positive Definite Matrices).
- Eigenvalues of $H$ = curvatures along principal directions; $\eta<2/\lambda_{\max}$ for stable GD; $\kappa = \lambda_{\max}/\lambda_{\min}$ sets convergence speed.
- Never form $H$ for big models: use Hessian-vector products (cost about 2 gradients).
- Gauss–Newton $J^\top H_\ell J$ and Fisher are PSD curvature approximations used by K-FAC, Shampoo, Laplace.
- Flows need $\log|\det J|$ (Normalizing Flows, Matrix Inverse and Determinant).
- NTK $= J_\theta J_\theta^\top$ (Neural Tangent Kernel).
- Sharpness (top Hessian eigenvalue) relates to training stability and, debatably, generalisation (Loss Landscapes).

## The math
**Jacobian.** For $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$,
$$
J_\mathbf{f}(\mathbf{x}) = \begin{pmatrix}\frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_n}\\ \vdots&\ddots&\vdots\\ \frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}\end{pmatrix},\qquad \mathbf{f}(\mathbf{x}+\mathbf{h}) = \mathbf{f}(\mathbf{x})+J_\mathbf{f}(\mathbf{x})\mathbf{h}+o(\|\mathbf{h}\|).
$$

**Hessian.** For $f:\mathbb{R}^n\to\mathbb{R}$, $H_{ij} = \dfrac{\partial^2f}{\partial x_i\partial x_j}$, $H = J_{\nabla f}$, and
$$
f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x})+\nabla f^\top\mathbf{h}+\tfrac12\mathbf{h}^\top H\mathbf{h}+o(\|\mathbf{h}\|^2).
$$

**Hessian of a composition.** For $\mathcal{L}(\boldsymbol\theta) = \ell(F(\boldsymbol\theta))$ with $J = \partial F/\partial\boldsymbol\theta$:
$$
\nabla^2\mathcal{L} = \underbrace{J^\top\nabla^2\ell\,J}_{\text{Gauss–Newton}}+\sum_k\frac{\partial\ell}{\partial F_k}\nabla^2F_k .
$$

**Hessian-vector product.** $H\mathbf{v} = \nabla_{\boldsymbol\theta}\big(\nabla_{\boldsymbol\theta}\mathcal{L}^\top\mathbf{v}\big)$, exact, at roughly twice the cost of one gradient.

**GD stability on the quadratic model.** Near a minimum with Hessian $H$, GD error evolves as $(I-\eta H)^t$; convergence requires $\eta<2/\lambda_{\max}(H)$.

**Hutchinson trace estimator.** For $\mathbf{z}$ with i.i.d. Rademacher entries, $\mathbb{E}[\mathbf{z}^\top H\mathbf{z}] = \operatorname{tr}(H)$.

**Softmax Jacobian.** $\mathbf{p} = \operatorname{softmax}(\mathbf{z})\Rightarrow J = \operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$ (symmetric PSD, rank $\le n-1$), which is also the Hessian of $\operatorname{logsumexp}(\mathbf{z})$ and of cross-entropy w.r.t. logits.

**NTK.** $\Theta(\mathbf{x},\mathbf{x}') = J_\theta f(\mathbf{x})\,J_\theta f(\mathbf{x}')^\top$.

## Run the code
```bash
pip install numpy torch
python jacobian_and_hessian.py
```

## Check yourself
<details><summary>How do you compute a Hessian-vector product for a neural network without forming the Hessian?</summary>

Compute the gradient $\mathbf{g} = \nabla\mathcal{L}$ while keeping the computational graph, take the dot product $\mathbf{g}^\top\mathbf{v}$ with a fixed vector, and backpropagate again: $\nabla(\mathbf{g}^\top\mathbf{v}) = H\mathbf{v}$. The cost is roughly two backward passes and memory similar to training. Alternatively, forward-over-reverse (a JVP of the gradient function) is often faster. HVPs feed power iteration, Lanczos, conjugate gradient and Hutchinson trace estimates.
</details>

<details><summary>What is the Gauss–Newton matrix and why is it preferred to the Hessian in practice?</summary>

For $\mathcal{L} = \ell(F(\theta))$, the Hessian splits into $J^\top\nabla^2\ell J$ plus a term weighted by the loss residual times the network's second derivatives. The Gauss–Newton matrix keeps only the first term. It is positive semidefinite whenever $\ell$ is convex in the outputs, so preconditioned steps are descent directions, and it equals the Fisher for common losses. The dropped term vanishes when residuals are small, so GN approximates the Hessian well near good fits.
</details>

<details><summary>How does the Hessian determine the maximum learning rate?</summary>

Locally the loss is approximately quadratic with Hessian $H$; gradient descent multiplies the error component along eigenvector $i$ by $(1-\eta\lambda_i)$ per step. Stability requires $|1-\eta\lambda_i|<1$ for all $i$, i.e. $\eta<2/\lambda_{\max}$. In deep learning full-batch GD self-tunes to this boundary (edge of stability), and warmup can be seen as giving sharpness time to fall before the learning rate gets large.
</details>

<details><summary>What is the Jacobian's role in normalizing flows and in adversarial robustness?</summary>

In flows, the change-of-variables formula adds $\log|\det J|$ of the invertible map to the base log-density, so architectures make $J$ triangular for an $O(d)$ log-det. In robustness, the input–output Jacobian's spectral norm is the local Lipschitz constant: a small perturbation $\boldsymbol\delta$ changes outputs by about $J\boldsymbol\delta$, so large Jacobian norms mean adversarial vulnerability; Jacobian regularisation penalises $\|J\|_F^2$.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.