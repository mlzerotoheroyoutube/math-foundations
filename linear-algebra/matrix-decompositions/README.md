# Matrix Decompositions

**Level:** Advanced  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A matrix decomposition rewrites a matrix as a product of simpler, structured factors (triangular, orthogonal, diagonal, low-rank, non-negative). Picking the right one turns expensive or unstable computations into cheap, stable ones: LU and Cholesky for solving, QR for least squares, eigen/Schur/SVD for spectra, and NMF or tensor decompositions for interpretable latent structure.

## The intuition
Factoring numbers makes hard problems easy: 391 looks messy, but 391 = 17 × 23 tells you everything about it. Matrix decompositions do the same for tables of numbers: they split one complicated matrix into a product of simple ones that are easy to work with.

Different jobs need different splits, like different tools in a toolbox:

- **LU** splits a matrix into a "lower staircase" times an "upper staircase" (triangle-shaped tables with zeros on one side). Triangles are easy: you solve the first equation, plug that in to solve the next, and so on, like knocking over dominoes.
- **QR** splits it into a "pure rotation" times a staircase. Rotations never blow up rounding errors, so it's the safe choice for fitting lines to data.
- **Cholesky** is a special, twice-as-fast LU for "bowl-shaped" matrices, a bit like taking a square root.
- **Eigen** and **SVD** split it into "turn, stretch, turn back", revealing the matrix's natural directions.
- **NMF** splits a table of only positive numbers into positive parts, like splitting a smoothie into the fruits that went in.

Example: a table of how often 1000 words appear in 500 news articles can be split by NMF into "articles × 10 topics" and "10 topics × words". One topic might weight "goal, match, team" highly: sport. The computer discovered the topics on its own just by factoring the table.

## Core notes
- **LU** ($PA = LU$): general square solves and determinants; $\tfrac23n^3$.
- **Cholesky** ($LL^\top$): SPD matrices only; twice as fast as LU; Gaussian sampling, GP posteriors, log-dets.
- **QR** ($A = QR$): least squares, orthonormal bases, Gram–Schmidt done right (Orthogonality and Projections).
- **Eigen** ($Q\Lambda Q^\top$) for symmetric; **Schur** ($QTQ^*$) for general: spectra, matrix functions ($e^A$, $A^{1/2}$).
- **SVD** ($U\Sigma V^\top$): rank, best low-rank approximation, pseudo-inverse, conditioning.
- **NMF** ($X\approx WH$, non-negative): parts-based, interpretable topics, spectra, audio sources.
- **Tensor**: CP (sum of rank-1 tensors), Tucker (core × factor matrices), tensor-train (chain of 3-way cores).
- Rule: **factor once, solve many** right-hand sides in $O(n^2)$ each.
- Exploit structure first: diagonal, banded, sparse, Toeplitz/circulant (FFT), Kronecker, low-rank-plus-diagonal.
- Low-rank and Kronecker factorisations are the backbone of PEFT and modern preconditioners (Parameter-Efficient Fine-Tuning, Modern Optimizers).

## The math
**LU with partial pivoting.** $PA = LU$ with $P$ a permutation, $L$ unit lower triangular ($|L_{ij}|\le1$), $U$ upper triangular. Solving $A\mathbf{x} = \mathbf{b}$: $L\mathbf{y} = P\mathbf{b}$ (forward), $U\mathbf{x} = \mathbf{y}$ (back), and $\det A = \det(P)\prod_iU_{ii}$ (with $\det P = \pm1$).

**QR and least squares.** For $A\in\mathbb{R}^{m\times n}$, $m\ge n$, full rank, thin QR $A = QR$ with $Q^\top Q = I_n$, $R$ upper triangular:
$$
\min_\mathbf{x}\|A\mathbf{x}-\mathbf{b}\|_2 \;\Longrightarrow\; R\mathbf{x} = Q^\top\mathbf{b}, \qquad \kappa(R) = \kappa(A)\ \text{(not squared)}.
$$

**Schur.** Every $A\in\mathbb{C}^{n\times n}$ satisfies $A = QTQ^*$ with $Q$ unitary and $T$ upper triangular holding the eigenvalues on its diagonal; $A$ normal $\iff T$ diagonal.

**NMF (Frobenius objective) and multiplicative updates** (Lee and Seung):
$$
\min_{W\ge0,H\ge0}\|X-WH\|_F^2,\qquad H\leftarrow H\odot\frac{W^\top X}{W^\top WH},\qquad W\leftarrow W\odot\frac{XH^\top}{WHH^\top},
$$
with element-wise products and divisions; each update does not increase the objective.

**CP decomposition** of an order-3 tensor $\mathcal{X}\in\mathbb{R}^{I\times J\times K}$:
$$
\mathcal{X}\approx\sum_{r=1}^R\lambda_r\,\mathbf{a}_r\otimes\mathbf{b}_r\otimes\mathbf{c}_r,\qquad \mathcal{X}_{ijk}\approx\sum_r\lambda_ra_{ir}b_{jr}c_{kr}.
$$
**Tucker:** $\mathcal{X}\approx\mathcal{G}\times_1A\times_2B\times_3C$ with core $\mathcal{G}\in\mathbb{R}^{R_1\times R_2\times R_3}$.

**Kronecker-factored curvature (K-FAC).** For a linear layer with input activations $\mathbf{a}$ and output gradients $\mathbf{g}$, the Fisher block is approximated as $F\approx\mathbb{E}[\mathbf{a}\mathbf{a}^\top]\otimes\mathbb{E}[\mathbf{g}\mathbf{g}^\top] = A\otimes G$, and since $(A\otimes G)^{-1} = A^{-1}\otimes G^{-1}$, the preconditioned gradient is $G^{-1}\,\nabla_W\,A^{-1}$: two small inverses instead of one huge one.

## Run the code
```bash
pip install numpy scikit-learn scipy torch
python matrix_decompositions.py
```

## Check yourself
<details><summary>When would you use LU, Cholesky, QR and SVD to solve a linear problem?</summary>

Cholesky when the matrix is symmetric positive definite (covariances, kernel matrices, $X^\top X+\lambda I$): it is twice as fast as LU and needs no pivoting. LU with partial pivoting for general square systems. QR for overdetermined least squares, because it avoids squaring the condition number as the normal equations do. SVD when the matrix may be rank-deficient or very ill-conditioned, or you need the minimum-norm solution, at a higher constant cost.
</details>

<details><summary>How does NMF differ from SVD/PCA, and why might you prefer it?</summary>

SVD gives the optimal low-rank approximation with orthogonal factors that mix positive and negative values, so components cancel each other and are hard to interpret. NMF constrains both factors to be non-negative, so data are explained as additive combinations of parts (topics, facial parts, spectral templates). NMF is non-convex, has no closed form, depends on initialisation and is not unique, but its factors are often more interpretable for count or intensity data.
</details>

<details><summary>Explain how K-FAC uses a Kronecker factorisation to make natural gradient tractable.</summary>

The Fisher matrix for a layer with $d_\text{in}\times d_\text{out}$ weights is $(d_\text{in}d_\text{out})^2$ in size, impossible to invert. K-FAC approximates the layer's Fisher block as $A\otimes G$, with $A$ the covariance of the layer inputs and $G$ the covariance of the backpropagated output gradients, assuming independence between them. Because $(A\otimes G)^{-1} = A^{-1}\otimes G^{-1}$, preconditioning reduces to $G^{-1}\nabla_WA^{-1}$, which needs only $d_\text{in}\times d_\text{in}$ and $d_\text{out}\times d_\text{out}$ inverses. Shampoo uses a related Kronecker structure with matrix roots.
</details>

<details><summary>Why is the CP decomposition harder than the matrix SVD?</summary>

For matrices, rank is easy to compute and truncated SVD is optimal (Eckart–Young). For order-3+ tensors, determining CP rank is NP-hard, the best rank-$R$ approximation may not exist (the set of rank-$\le R$ tensors is not closed, leading to degenerate solutions with diverging components), and there is no closed-form algorithm: ALS and gradient methods can converge slowly or to local optima. On the plus side, CP is often unique under mild conditions (Kruskal), which is what makes it useful for identifying latent-variable models.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.