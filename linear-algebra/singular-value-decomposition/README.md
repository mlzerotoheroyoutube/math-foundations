# Singular Value Decomposition

**Level:** Intermediate  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Every matrix, of any shape, factors as $A = U\Sigma V^\top$: a rotation, then a stretch along perpendicular axes, then another rotation. Keeping only the largest stretches gives the best possible low-rank approximation (Eckart–Young), which is why SVD sits underneath PCA, recommender systems, latent semantic analysis, LoRA, model compression, pseudo-inverses and spectral normalisation.

## The intuition
Imagine any stretchy transformation of a rubber sheet: squash, slant, flip, whatever. The SVD says you can *always* break it into three simple moves:

1. **Turn** the sheet (a rotation).
2. **Stretch** it along the left–right and up–down directions by some amounts (no turning, just pulling).
3. **Turn** it again.

The stretch amounts are called **singular values**, and they are listed from biggest to smallest. A circle drawn on the sheet always ends up as an ellipse: the singular values are the lengths of the ellipse's long and short axes.

Now the magic for data. A photo is a big grid of numbers, say 1000 × 1000 = one million numbers. The SVD writes it as a sum of simple "layers", each one just a column pattern times a row pattern, weighted by a singular value. The first few layers carry most of the picture: the big shapes and shading. Later layers only add fine speckle. Keep the top 50 layers and you store about 50 × (1000 + 1000 + 1) ≈ 100,000 numbers, a tenth of the original, and the picture still looks almost the same.

That's the same trick Netflix-style recommenders use: a huge table of people × films is mostly explained by a handful of hidden "taste" layers.

## Core notes
- $A = U\Sigma V^\top$ exists for **every** matrix; $U$, $V$ orthogonal, $\Sigma$ diagonal non-negative, sorted descending.
- Geometry: rotate ($V^\top$) → scale axes ($\Sigma$) → rotate ($U$). The unit sphere maps to an ellipsoid with semi-axes $\sigma_i\mathbf{u}_i$.
- Outer-product form $A = \sum_i\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$: a weighted sum of rank-1 "layers".
- **Rank** = number of non-zero singular values (Vector Spaces, Basis and Rank).
- **Eckart–Young**: truncating to the top $k$ terms is the best rank-$k$ approximation; Frobenius error $= \sqrt{\sum_{i>k}\sigma_i^2}$.
- Relation to eigen: $A^\top A = V\Sigma^2V^\top$, $AA^\top = U\Sigma^2U^\top$ (Eigenvalues and Eigenvectors).
- $\sigma_1 = \|A\|_2$ (Lipschitz constant); $\sigma_1/\sigma_r$ = condition number; pseudo-inverse $A^+ = V\Sigma^+U^\top$ (Matrix Inverse and Determinant).
- PCA: SVD of centred $X$; principal directions = columns of $V$, variances $= \sigma_i^2/(n-1)$ (Principal Component Analysis).
- Storage of rank-$k$ approximation: $k(m+n+1)$ numbers instead of $mn$.
- Large matrices: `randomized_svd`, `TruncatedSVD`, `torch.svd_lowrank` instead of full SVD.

## The math
**Existence.** For $A\in\mathbb{R}^{m\times n}$ of rank $r$:
$$
A = U\Sigma V^\top = \sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top,\qquad U^\top U = I_m,\ V^\top V = I_n,\ \sigma_1\ge\cdots\ge\sigma_r>0 .
$$
Here $\mathbf{u}_i$ (left singular vectors) span $\operatorname{col}A$, $\mathbf{v}_i$ (right singular vectors) span $\operatorname{row}A$, and $A\mathbf{v}_i = \sigma_i\mathbf{u}_i$, $A^\top\mathbf{u}_i = \sigma_i\mathbf{v}_i$.

**Variational characterisation.**
$$
\sigma_1 = \max_{\|\mathbf{x}\| = \|\mathbf{y}\| = 1}\mathbf{y}^\top A\mathbf{x} = \|A\|_2,\qquad \sigma_k = \min_{\operatorname{rank}B<k}\|A-B\|_2 .
$$

**Eckart–Young–Mirsky.** With $A_k = \sum_{i=1}^k\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$,
$$
\min_{\operatorname{rank}B\le k}\|A-B\|_F = \|A-A_k\|_F = \Big(\sum_{i>k}\sigma_i^2\Big)^{1/2},\qquad \min_{\operatorname{rank}B\le k}\|A-B\|_2 = \sigma_{k+1}.
$$

**Proof sketch of the link to eigenvalues.** $A^\top A = V\Sigma^\top U^\top U\Sigma V^\top = V(\Sigma^\top\Sigma)V^\top$ is symmetric PSD, so its spectral decomposition supplies $V$ and $\sigma_i^2$; then define $\mathbf{u}_i = A\mathbf{v}_i/\sigma_i$, which are orthonormal because $\mathbf{u}_i^\top\mathbf{u}_j = \mathbf{v}_i^\top A^\top A\mathbf{v}_j/(\sigma_i\sigma_j) = \delta_{ij}$.

**PCA via SVD.** For centred $X\in\mathbb{R}^{n\times d}$, $X = U\Sigma V^\top$ gives sample covariance $\tfrac{1}{n-1}X^\top X = V\tfrac{\Sigma^2}{n-1}V^\top$; scores are $XV_k = U_k\Sigma_k$; explained-variance ratio of component $i$ is $\sigma_i^2/\sum_j\sigma_j^2$.

**Pseudo-inverse and ridge.** $A^+ = V\Sigma^+U^\top$ with $\Sigma^+_{ii} = 1/\sigma_i$ for $\sigma_i>0$. Ridge regression shrinks each component by a filter factor:
$$
\mathbf{w}_\lambda = \sum_i\frac{\sigma_i}{\sigma_i^2+\lambda}(\mathbf{u}_i^\top\mathbf{y})\,\mathbf{v}_i ,
$$
so small-$\sigma$ (noisy) directions are damped most.

**Randomised SVD error** (Halko et al.): with oversampling $p\ge2$, $\mathbb{E}\|A-QQ^\top A\|_F\le\big(1+\tfrac{k}{p-1}\big)^{1/2}\big(\sum_{j>k}\sigma_j^2\big)^{1/2}$.

## Run the code
```bash
pip install numpy scikit-learn
python singular_value_decomposition.py
```

## Check yourself
<details><summary>What is the SVD and how does it differ from eigendecomposition?</summary>

SVD factors any $m\times n$ matrix as $U\Sigma V^\top$ with orthogonal $U$, $V$ and non-negative diagonal $\Sigma$. Eigendecomposition $V\Lambda V^{-1}$ requires a square matrix, may not exist (defective matrices), and can have complex values and non-orthogonal eigenvectors. SVD always exists, is numerically stable, and uses two different orthonormal bases for input and output spaces. They coincide for symmetric PSD matrices, and in general the singular values are square roots of the eigenvalues of $A^\top A$.
</details>

<details><summary>State the Eckart–Young theorem and give two ML uses.</summary>

The best rank-$k$ approximation of $A$ in Frobenius or spectral norm is obtained by keeping the top $k$ singular triplets; the errors are $\sqrt{\sum_{i>k}\sigma_i^2}$ and $\sigma_{k+1}$ respectively. Uses: PCA keeps the top-$k$ directions of centred data, giving the subspace with minimum reconstruction error; model compression replaces a weight matrix with a product of two thin matrices from its truncated SVD. It also justifies LSA for text and low-rank denoising.
</details>

<details><summary>How is PCA computed with SVD, and why is that better than eigendecomposing the covariance?</summary>

Centre $X$, compute the thin SVD $X = U\Sigma V^\top$; principal directions are the columns of $V$, component variances are $\sigma_i^2/(n-1)$, and projected scores are $U\Sigma$. Forming $X^\top X$ squares the condition number, so small-variance components lose precision, and it costs an extra $O(nd^2)$ product. SVD on $X$ directly avoids both and is what scikit-learn does.
</details>

<details><summary>You need the top 50 singular vectors of a 10M × 100k sparse matrix. How?</summary>

Full SVD is infeasible. Use a truncated method that only needs matrix-vector products: randomised SVD (sketch with a Gaussian test matrix, a couple of power iterations, QR, small SVD) or Lanczos/ARPACK (`scipy.sparse.linalg.svds`, `TruncatedSVD`). Keep the matrix sparse throughout, oversample by about 10, and use power iterations if the spectrum decays slowly. Distributed implementations (Spark MLlib) follow the same pattern.
</details>

<details><summary>Why can SVD backward passes be numerically unstable in PyTorch?</summary>

The gradient of the singular vectors involves terms $1/(\sigma_i^2-\sigma_j^2)$, which blow up when two singular values are close or equal. Gradients of singular values alone are stable ($\partial\sigma_i/\partial A = \mathbf{u}_i\mathbf{v}_i^\top$). Workarounds: differentiate only through singular values or the projector onto a well-separated subspace, add small perturbations, use power iteration with a fixed number of steps, or use Newton–Schulz iterations for the polar factor.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.