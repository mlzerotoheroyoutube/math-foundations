# Eigenvalues and Eigenvectors

**Level:** Intermediate  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> An eigenvector of a matrix is a direction the matrix only stretches (never turns); the stretch factor is its eigenvalue: $A\mathbf{v} = \lambda\mathbf{v}$. Eigen-analysis tells you the "natural axes" of a linear map, which is how ML finds principal components, analyses curvature of loss landscapes, ranks web pages, clusters graphs and diagnoses exploding gradients.

## The intuition
Imagine stretching a piece of dough with a rolling pin that only pushes left and right. Draw some arrows on the dough first. After rolling, most arrows have turned: an arrow pointing up-and-right now points more to the right. But two kinds of arrows keep their direction: arrows pointing exactly left–right just get **longer**, and arrows pointing exactly up–down **don't change at all**. Those special arrows are the dough's **eigenvectors** ("eigen" is German for "own" or "characteristic"). How much each one gets stretched is its **eigenvalue**.

Tiny example: the matrix that doubles x and keeps y the same. The arrow `[1, 0]` becomes `[2, 0]`: same direction, twice as long, eigenvalue 2. The arrow `[0, 1]` stays `[0, 1]`: eigenvalue 1. But `[1, 1]` becomes `[2, 1]`, which points in a new direction, so it is not an eigenvector.

Why do we care? Repeating a transformation many times makes the direction with the biggest eigenvalue take over. If the biggest eigenvalue is above 1, things blow up; below 1, they shrink to nothing. That's exactly why some neural networks "explode" or "forget" when signals pass through many layers. Google's original ranking of web pages was also an eigenvector: the page-importance scores that stay the same after one more round of "follow a random link".

## Core notes
- Definition: $A\mathbf{v} = \lambda\mathbf{v}$, $\mathbf{v}\ne\mathbf{0}$. Find $\lambda$ from $\det(A-\lambda I) = 0$, then $\mathbf{v}$ from $\ker(A-\lambda I)$.
- $\sum\lambda_i = \operatorname{tr}A$, $\prod\lambda_i = \det A$ (Matrix Inverse and Determinant).
- **Symmetric** matrices (covariances, Hessians, kernels, Laplacians): real eigenvalues, orthogonal eigenvectors, $A = Q\Lambda Q^\top$.
- Powers: $A^k = V\Lambda^kV^{-1}$; long-run behaviour is dominated by $|\lambda_{\max}|$ (spectral radius $\rho(A)$).
- $\rho(A)>1$: repeated application explodes; $\rho(A)<1$: it decays. This is the RNN gradient story (Recurrent Neural Networks, Backpropagation Through Time).
- **Positive definite** ⇔ all eigenvalues $>0$ for symmetric matrices (Positive Definite Matrices).
- PCA directions = eigenvectors of the covariance; variance explained = eigenvalues (Principal Component Analysis).
- Hessian eigenvalues = curvatures; $\lambda_{\max}$ caps the stable learning rate ($\eta<2/\lambda_{\max}$) (Gradient Descent).
- Eigenvalues of non-symmetric real matrices can be complex (rotations).
- SVD is the eigen-decomposition of $A^\top A$ and $AA^\top$, and works for any matrix (Singular Value Decomposition).

## The math
**Definition and characteristic polynomial.**
$$
A\mathbf{v} = \lambda\mathbf{v},\ \mathbf{v}\ne\mathbf{0} \iff \det(A-\lambda I) = 0, \qquad p_A(\lambda) = \det(A-\lambda I).
$$

**Diagonalisation.** If $A$ has $n$ independent eigenvectors $V = [\mathbf{v}_1\cdots\mathbf{v}_n]$, then $A = V\Lambda V^{-1}$, $\Lambda = \operatorname{diag}(\lambda_i)$, and $A^k = V\Lambda^kV^{-1}$, $e^{A} = Ve^{\Lambda}V^{-1}$.

**Spectral theorem.** If $A = A^\top\in\mathbb{R}^{n\times n}$, then $A = Q\Lambda Q^\top = \sum_i\lambda_i\mathbf{q}_i\mathbf{q}_i^\top$ with $Q$ orthogonal and $\lambda_i\in\mathbb{R}$.

**Rayleigh quotient / Courant–Fischer.** For symmetric $A$ with $\lambda_1\ge\cdots\ge\lambda_n$:
$$
\lambda_1 = \max_{\mathbf{x}\ne0}\frac{\mathbf{x}^\top A\mathbf{x}}{\mathbf{x}^\top\mathbf{x}},\qquad \lambda_k = \max_{\dim S = k}\ \min_{\mathbf{x}\in S\setminus\{0\}}\frac{\mathbf{x}^\top A\mathbf{x}}{\mathbf{x}^\top\mathbf{x}} .
$$
PCA's first component maximises variance $\mathbf{w}^\top\Sigma\mathbf{w}$ subject to $\|\mathbf{w}\| = 1$, so it is the top eigenvector of $\Sigma$.

**GD on a quadratic.** For $f(\boldsymbol{\theta}) = \tfrac12\boldsymbol{\theta}^\top H\boldsymbol{\theta}$ with $H = Q\Lambda Q^\top$, GD gives $\boldsymbol{\theta}_t = Q(I-\eta\Lambda)^tQ^\top\boldsymbol{\theta}_0$: component $i$ evolves as $(1-\eta\lambda_i)^t$, convergent iff $0<\eta<2/\lambda_{\max}$, and the slowest rate is set by $\lambda_{\min}$, hence the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$.

**Power iteration convergence.** If $|\lambda_1|>|\lambda_2|\ge\cdots$, the angle between $\mathbf{b}_k$ and $\mathbf{v}_1$ shrinks as $O(|\lambda_2/\lambda_1|^k)$.

**Weyl and Davis–Kahan** (symmetric $A$, perturbation $E$): $|\lambda_i(A+E)-\lambda_i(A)|\le\|E\|_2$; $\sin\angle(\mathbf{v}_i,\tilde{\mathbf{v}}_i)\lesssim\|E\|_2/\delta_i$ where $\delta_i$ is the eigen-gap.

**Graph Laplacian.** $L = D-W$ is PSD with $\mathbf{x}^\top L\mathbf{x} = \tfrac12\sum_{ij}W_{ij}(x_i-x_j)^2$; the multiplicity of eigenvalue 0 equals the number of connected components.

## Run the code
```bash
pip install numpy torch
python eigenvalues_and_eigenvectors.py
```

## Check yourself
<details><summary>Why are the eigenvalues of a symmetric matrix real and its eigenvectors orthogonal?</summary>

For $A\mathbf{v} = \lambda\mathbf{v}$ with possibly complex $\mathbf{v}$, compute $\bar{\mathbf{v}}^\top A\mathbf{v} = \lambda\|\mathbf{v}\|^2$; taking the conjugate transpose and using $A = A^\top$ real gives the same quantity equals $\bar\lambda\|\mathbf{v}\|^2$, so $\lambda = \bar\lambda$. For distinct eigenvalues $\lambda_1\ne\lambda_2$: $\lambda_1\mathbf{v}_2^\top\mathbf{v}_1 = \mathbf{v}_2^\top A\mathbf{v}_1 = (A\mathbf{v}_2)^\top\mathbf{v}_1 = \lambda_2\mathbf{v}_2^\top\mathbf{v}_1$, forcing $\mathbf{v}_2^\top\mathbf{v}_1 = 0$. Within a repeated eigenvalue's eigenspace one can choose an orthonormal basis. This is why covariance and Hessian analyses are so clean.
</details>

<details><summary>How do Hessian eigenvalues relate to the maximum stable learning rate?</summary>

Near a minimum the loss is approximately quadratic, $\tfrac12\boldsymbol{\delta}^\top H\boldsymbol{\delta}$. In the eigenbasis of $H$, GD multiplies each component by $(1-\eta\lambda_i)$ per step, which shrinks only if $|1-\eta\lambda_i|<1$, i.e. $\eta<2/\lambda_i$. So the largest eigenvalue (sharpness) caps the learning rate at $2/\lambda_{\max}$, while the smallest eigenvalue determines how slowly the flattest direction converges. In deep learning, full-batch GD tends to drive sharpness up to exactly $2/\eta$ (edge of stability) and then hover there.
</details>

<details><summary>How would you find the top eigenvalue of a matrix too large to store?</summary>

Use power iteration or Lanczos, which only require matrix-vector products. For a neural network's Hessian, compute Hessian-vector products via double backpropagation (Pearlmutter's trick) at roughly the cost of two gradient evaluations, normalise, repeat, and read off the Rayleigh quotient. Lanczos converges faster and gives several extremal eigenvalues; stochastic Lanczos quadrature even estimates the whole spectral density.
</details>

<details><summary>What is the relationship between eigendecomposition and SVD?</summary>

For any $A$, $A^\top A = V\Sigma^2V^\top$ and $AA^\top = U\Sigma^2U^\top$: the right and left singular vectors are eigenvectors of these Gram matrices, and singular values are square roots of their eigenvalues. For symmetric PSD matrices the two decompositions coincide. SVD exists for every (even rectangular) matrix and always has orthonormal factors, while eigendecomposition needs a square, diagonalisable matrix and may be complex and non-orthogonal for non-symmetric matrices.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.