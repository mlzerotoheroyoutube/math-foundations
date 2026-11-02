# Norms and Distances

**Level:** Beginner  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A norm measures the size of a vector (or matrix); a distance measures how far apart two things are, usually as the norm of their difference. Choosing L1 vs L2 vs cosine vs spectral norm silently decides what your model treats as "big" or "similar": it drives regularisation, nearest-neighbour search, gradient clipping and robustness guarantees.

## The intuition
How far is it from your house to the shop, if the shop is 3 blocks east and 4 blocks north?

- A **bird** flies straight there: by Pythagoras, √(3² + 4²) = √25 = **5** blocks. That's the **L2** or **Euclidean** distance, "as the crow flies".
- A **taxi** has to follow the streets: 3 + 4 = **7** blocks. That's the **L1** or **Manhattan** distance, named after New York's grid.
- A **chess king** moves one square in any direction including diagonally, so it needs only the larger of the two: **4** moves. That's the **L∞** or **max** distance.

Same trip, three different answers, and each is "correct" for a different traveller. A **norm** is just "the size of an arrow" measured in one of these ways, and a **distance** is the size of the arrow from one point to the other.

Why does machine learning care? Suppose a computer compares songs by numbers like tempo and loudness. Which songs count as "close" depends on which ruler you use. And when we want a model to stay simple, we punish it for having big settings, and *how* we measure "big" changes the result: the taxi ruler (L1) tends to switch many settings off completely, the bird ruler (L2) tends to make them all small but not zero.

## Core notes
- $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$ (Euclidean), $\|\mathbf{x}\|_1 = \sum|x_i|$ (Manhattan), $\|\mathbf{x}\|_\infty = \max|x_i|$ (Chebyshev).
- Unit balls: L2 sphere, L1 diamond (corners on axes → sparsity), L∞ cube.
- **L2 penalty** (ridge, weight decay) shrinks all weights smoothly; **L1 penalty** (lasso) drives many exactly to zero (Ridge, Lasso and Elastic Net, Regularization).
- **Squared** L2 is used in losses (MSE) because it is smooth and gives linear gradients; plain L2 is used in distances.
- **Cosine distance** ignores magnitude; equals half squared L2 distance on unit vectors (Vector Operations and Dot Product).
- **Mahalanobis** $\sqrt{(\mathbf{x}-\mathbf{y})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$ accounts for feature scales and correlations (Positive Definite Matrices).
- Matrix norms: Frobenius (entry-wise L2), spectral $\sigma_{\max}$ (max stretch, Lipschitz constant), nuclear $\sum\sigma_i$ (rank surrogate).
- Gradient Clipping rescales the gradient when its global L2 norm exceeds a threshold (commonly 1.0 for LLMs).
- Always scale features before using distances (Feature Scaling and Normalization); otherwise the feature with the largest units dominates.
- In high dimensions all pairwise distances look alike (Curse of Dimensionality).

## The math
**Norm axioms.** For all $\mathbf{x},\mathbf{y}$ and scalars $a$: $\|\mathbf{x}\|\ge0$ with equality iff $\mathbf{x} = \mathbf{0}$; $\|a\mathbf{x}\| = |a|\|\mathbf{x}\|$; $\|\mathbf{x}+\mathbf{y}\|\le\|\mathbf{x}\|+\|\mathbf{y}\|$.

**$\ell_p$ norms.** $\|\mathbf{x}\|_p = \big(\sum_{i=1}^n|x_i|^p\big)^{1/p}$ for $p\ge1$, $\|\mathbf{x}\|_\infty = \max_i|x_i|$. Equivalences:
$$
\|\mathbf{x}\|_\infty\le\|\mathbf{x}\|_2\le\|\mathbf{x}\|_1\le\sqrt{n}\,\|\mathbf{x}\|_2\le n\,\|\mathbf{x}\|_\infty .
$$

**Hölder and dual norms.** $|\mathbf{x}^\top\mathbf{y}|\le\|\mathbf{x}\|_p\|\mathbf{y}\|_q$ with $\tfrac1p+\tfrac1q = 1$. Hence $\max_{\|\boldsymbol{\delta}\|_\infty\le\epsilon}\mathbf{w}^\top\boldsymbol{\delta} = \epsilon\|\mathbf{w}\|_1$, attained at $\boldsymbol{\delta} = \epsilon\operatorname{sign}(\mathbf{w})$: the FGSM step.

**Matrix norms.** For $A$ with singular values $\sigma_1\ge\sigma_2\ge\cdots$:
$$
\|A\|_2 = \max_{\mathbf{x}\ne\mathbf{0}}\frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_1,\qquad \|A\|_F = \sqrt{\textstyle\sum_{ij}A_{ij}^2} = \sqrt{\textstyle\sum_i\sigma_i^2},\qquad \|A\|_* = \textstyle\sum_i\sigma_i .
$$
Sub-multiplicativity $\|AB\|\le\|A\|\|B\|$ gives the network Lipschitz bound $\operatorname{Lip}(f)\le\prod_\ell\|W_\ell\|_2\cdot\prod_\ell\operatorname{Lip}(\sigma_\ell)$.

**Why L1 gives sparsity.** The proximal operator of $\lambda|\cdot|$ is soft-thresholding:
$$
\operatorname{prox}_{\lambda|\cdot|}(v) = \operatorname{sign}(v)\max(|v|-\lambda, 0),
$$
which maps every $|v|\le\lambda$ exactly to 0; the L2 prox $v/(1+2\lambda)$ only shrinks.

**Mahalanobis distance.** $d_\Sigma(\mathbf{x},\mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{y})} = \|L^{-1}(\mathbf{x}-\mathbf{y})\|_2$ with $\Sigma = LL^\top$: Euclidean distance after whitening.

**Gradient clipping by global norm.** $\mathbf{g}\leftarrow\mathbf{g}\cdot\min\big(1, \tfrac{c}{\|\mathbf{g}\|_2}\big)$ for threshold $c$.

## Run the code
```bash
pip install numpy torch
python norms_and_distances.py
```

## Check yourself
<details><summary>Why does L1 regularisation produce sparse solutions while L2 does not?</summary>

Geometrically, the L1 ball is a diamond with corners on the coordinate axes; the level sets of a smooth loss typically first touch it at a corner, where some coordinates are exactly zero. The L2 ball is round, so the touching point generically has all coordinates non-zero. Analytically, the proximal step for L1 is soft-thresholding, which sets any coordinate with magnitude below $\lambda$ to exactly zero, while the L2 step only multiplies by a factor less than 1. Equivalently, the L1 subgradient at 0 is the whole interval $[-\lambda,\lambda]$, so zero is stable unless the loss gradient exceeds $\lambda$.
</details>

<details><summary>What is the spectral norm of a weight matrix and why does it matter?</summary>

It is the largest singular value, $\sigma_{\max}(W)$, the maximum factor by which $W$ can stretch any input vector, i.e. the Lipschitz constant of the linear layer under L2. The product of layer spectral norms bounds how much a network's output can change for a small input change, which links to adversarial robustness, GAN discriminator stability (spectral normalisation divides $W$ by $\sigma_{\max}$), exploding activations, and norm-based generalisation bounds. It is cheaply estimated by a few steps of power iteration.
</details>

<details><summary>How do you compute all pairwise Euclidean distances between two sets of vectors efficiently?</summary>

Expand $\|\mathbf{x}-\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^\top\mathbf{y}$. Compute the squared norms of each row once, and all cross terms as one matrix product $XY^\top$, which runs at BLAS/GPU speed with $O((n+m)d)$ extra memory rather than materialising an $n\times m\times d$ tensor. Clamp the result at zero to remove small negative values from floating-point cancellation, and take the square root only if needed.
</details>

<details><summary>Why is Euclidean distance unreliable in high dimensions, and what do you do about it?</summary>

For many data distributions, as dimension grows the distances from a query to its nearest and farthest neighbours become nearly equal relative to their size (distance concentration), so "nearest" carries little information. Irrelevant or noisy features also add variance to every distance. Remedies: feature selection, dimensionality reduction (PCA), learning task-specific embeddings or metrics, using cosine on normalised learned representations, and approximate nearest-neighbour indexes designed for embedding spaces.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.