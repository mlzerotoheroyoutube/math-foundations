# Positive Definite Matrices

**Level:** Intermediate  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A symmetric matrix $A$ is positive definite if $\mathbf{x}^\top A\mathbf{x}>0$ for every non-zero $\mathbf{x}$, equivalently all its eigenvalues are positive. Such matrices behave like positive numbers: they define bowls (convex quadratics), valid covariances, valid kernels and valid distance metrics, and they have a cheap, stable "square root" (Cholesky).

## The intuition
Think of a bowl. Wherever you put a marble inside it, the marble rolls down to the single lowest point in the middle. A **positive definite** matrix is the maths recipe for a perfect bowl shape.

With ordinary numbers it's simple: the curve $y = 3x^2$ is a bowl (because 3 is positive), $y = -3x^2$ is an upside-down bowl (a hill), and $y = 0\cdot x^2$ is flat. With two or more directions, the "3" becomes a small table of numbers, a matrix, and we want to know: is the shape a bowl in *every* direction?

Example: the matrix with 2 and 1 on the diagonal and 0 elsewhere gives height $2x^2 + 1y^2$. Plug in any point except the centre, like $(1, -1)$: $2 + 1 = 3$, positive. It's a bowl: positive definite. Now try $2x^2 - 1y^2$: at $(0, 1)$ the height is $-1$, so in the $y$ direction it curves downwards. That's a saddle, like a horse saddle or a Pringle crisp: not positive definite.

Why it matters: when a computer trains a model, it's rolling a marble down a landscape of "error". If the landscape is a perfect bowl, there's one best answer and you are guaranteed to find it. Positive definite tables also describe how spread out data is (for example, how height and weight vary together), and these can never be negative.

## Core notes
- Definition: symmetric $A$ with $\mathbf{x}^\top A\mathbf{x}>0$ for all $\mathbf{x}\ne\mathbf{0}$ (PD) or $\ge0$ (PSD).
- Tests: all eigenvalues $>0$ ⇔ Cholesky succeeds ⇔ leading principal minors $>0$ ⇔ $A = B^\top B$ with $B$ full column rank.
- Any $X^\top X$ (Gram matrix) is PSD; covariances are PSD (Expectation, Variance and Covariance).
- PD ⇒ invertible, $\det>0$, inverse also PD, diagonal entries $>0$.
- Quadratic $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top A\mathbf{x}-\mathbf{b}^\top\mathbf{x}$ with $A\succ0$ is a bowl with unique minimiser $A^{-1}\mathbf{b}$.
- Hessian PSD everywhere ⇔ convex; PD at a critical point ⇒ strict local minimum; indefinite ⇒ saddle (Jacobian and Hessian).
- Valid kernel ⇔ PSD Gram matrices (Mercer) (Kernel Methods and the Kernel Trick).
- **Cholesky** $A = LL^\top$: half the cost of LU, used for sampling Gaussians, GP inference, log-dets (Matrix Decompositions).
- Add **jitter** $\epsilon I$ (e.g. $10^{-6}$) when Cholesky fails on a kernel or covariance matrix.
- Damping $H+\lambda I$ makes curvature matrices PD in Newton-type and Levenberg–Marquardt steps.

## The math
**Definitions.** For symmetric $A$: $A\succ0\iff\mathbf{x}^\top A\mathbf{x}>0\ \forall\mathbf{x}\ne\mathbf{0}$; $A\succeq0\iff\mathbf{x}^\top A\mathbf{x}\ge0\ \forall\mathbf{x}$.

**Eigenvalue characterisation.** With $A = Q\Lambda Q^\top$ and $\mathbf{y} = Q^\top\mathbf{x}$,
$$
\mathbf{x}^\top A\mathbf{x} = \sum_{i=1}^n\lambda_iy_i^2,
$$
so $A\succ0\iff\lambda_i>0$ for all $i$, and $\lambda_{\min}\|\mathbf{x}\|^2\le\mathbf{x}^\top A\mathbf{x}\le\lambda_{\max}\|\mathbf{x}\|^2$.

**Cholesky.** $A\succ0\iff A = LL^\top$ with $L$ lower triangular, $L_{ii}>0$. Entries:
$$
L_{jj} = \sqrt{A_{jj}-\textstyle\sum_{k<j}L_{jk}^2},\qquad L_{ij} = \frac{1}{L_{jj}}\Big(A_{ij}-\textstyle\sum_{k<j}L_{ik}L_{jk}\Big),\ i>j .
$$

**Sampling a Gaussian.** If $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$ and $\Sigma = LL^\top$, then $\boldsymbol{\mu}+L\mathbf{z}\sim\mathcal{N}(\boldsymbol{\mu},\Sigma)$, since $\operatorname{Cov}(L\mathbf{z}) = LL^\top$.

**Convexity.** For $f\in C^2$: $f$ convex $\iff\nabla^2f(\mathbf{x})\succeq0\ \forall\mathbf{x}$; $\mu$-strongly convex and $L$-smooth $\iff\mu I\preceq\nabla^2f\preceq LI$, giving condition number $\kappa = L/\mu$ (Convergence of Gradient Methods).

**Mercer / kernel validity.** $k$ is PSD iff $\sum_{i,j}c_ic_jk(\mathbf{x}_i,\mathbf{x}_j)\ge0$ for all finite sets $\{\mathbf{x}_i\}$ and $\mathbf{c}\in\mathbb{R}^N$; then $k(\mathbf{x},\mathbf{x}') = \langle\phi(\mathbf{x}),\phi(\mathbf{x}')\rangle_\mathcal{H}$ for some feature map into an RKHS $\mathcal{H}$.

**Nearest PSD matrix** (Frobenius, symmetric input $A = Q\Lambda Q^\top$): $A_+ = Q\max(\Lambda,0)Q^\top$.

**Schur complement.** $\begin{pmatrix}A&B\\B^\top&C\end{pmatrix}\succ0\iff A\succ0$ and $C-B^\top A^{-1}B\succ0$; the Schur complement is the conditional covariance of a Gaussian.

## Run the code
```bash
pip install numpy torch
python positive_definite_matrices.py
```

## Check yourself
<details><summary>Give three equivalent definitions of a positive definite matrix.</summary>

For symmetric $A$: (1) $\mathbf{x}^\top A\mathbf{x}>0$ for every non-zero $\mathbf{x}$; (2) all eigenvalues are strictly positive; (3) $A$ has a Cholesky factorisation $LL^\top$ with positive diagonal, equivalently $A = B^\top B$ for some $B$ with full column rank. Sylvester's criterion (all leading principal minors positive) is a fourth. In code, attempting Cholesky is the cheapest test.
</details>

<details><summary>Why must a covariance matrix be positive semidefinite?</summary>

For any fixed vector $\mathbf{w}$, $\mathbf{w}^\top\Sigma\mathbf{w} = \operatorname{Var}(\mathbf{w}^\top X)$, the variance of a scalar random variable, which cannot be negative. It is strictly positive definite unless some linear combination of the variables is constant (perfectly collinear features), in which case $\Sigma$ is singular. Sample covariances with fewer samples than dimensions are always singular.
</details>

<details><summary>How does positive definiteness relate to convexity and to local minima?</summary>

A twice-differentiable function is convex iff its Hessian is PSD everywhere, and strictly convex if PD. At a critical point (zero gradient), a PD Hessian guarantees a strict local minimum, a negative definite one a maximum, and an indefinite one a saddle point. In deep learning the Hessian is typically indefinite during training, which is why plain Newton steps can move uphill and why PSD approximations (Gauss–Newton, Fisher) or damping are used.
</details>

<details><summary>Your GP code crashes with "matrix not positive definite". What do you do?</summary>

The kernel matrix is PD in exact arithmetic but numerically has tiny negative eigenvalues, usually because inputs are nearly duplicated or the lengthscale is large relative to point spacing. Add jitter $\epsilon I$ to the diagonal (starting around $10^{-6}$ times the average diagonal, increasing if needed) or increase the noise variance. Also check for duplicate inputs, use float64, and make sure the kernel itself is a valid PSD kernel (some ad hoc similarity functions are not).
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.