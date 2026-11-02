# Matrix Inverse and Determinant

**Level:** Intermediate  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> The inverse $A^{-1}$ undoes a linear map; the determinant $\det A$ is the factor by which the map scales volume (and its sign says whether it flips orientation). A matrix is invertible exactly when its determinant is non-zero. In ML the log-determinant is everywhere: Gaussian likelihoods, normalizing flows, and Bayesian model evidence.

## The intuition
Think of a matrix as a machine that stretches and twists a picture drawn on a rubber sheet. The **inverse** is the "undo" button: run the picture through the machine and then through the inverse, and you get the original picture back. Doubling everything is undone by halving everything. Rotating 30° left is undone by rotating 30° right.

The **determinant** is one number that tells you how much the machine changes areas. Draw a 1-by-1 square. If the machine turns it into a shape with area 6, the determinant is 6. If the shape comes out mirror-flipped, the determinant is negative, like −6.

Example: the matrix that stretches width by 2 and height by 3 turns the unit square into a 2-by-3 rectangle, area 6, so the determinant is 2 × 3 = 6. Its inverse shrinks width by half and height by a third.

What if the determinant is **0**? Then the machine squashes the square flat into a line, area 0. You can't undo that: once a whole square has collapsed onto a line, you can't tell which point came from where. So: **determinant 0 means no undo button.** That's why it matters in machine learning: some methods need to undo a transformation (for example, to turn generated images back into random noise), and they only work when the determinant is never zero.

## Core notes
- $AA^{-1} = A^{-1}A = I$; exists only for square matrices with $\det A\ne0$.
- $2\times2$: $\begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$, $\det = ad-bc$.
- $\det A$ = signed volume scale factor = product of eigenvalues (Eigenvalues and Eigenvectors); $\det(AB) = \det A\det B$, $\det A^\top = \det A$, $\det A^{-1} = 1/\det A$.
- $(AB)^{-1} = B^{-1}A^{-1}$; $(A^\top)^{-1} = (A^{-1})^\top$; orthogonal $Q^{-1} = Q^\top$; diagonal inverse = reciprocal entries.
- Use **log-det** via `slogdet` or Cholesky ($\log\det\Sigma = 2\sum_i\log L_{ii}$), never `log(det(...))`.
- Near-singularity is measured by the **condition number**, not by $|\det|$ (Norms and Distances).
- **Pseudo-inverse** $A^+ = V\Sigma^+U^\top$ handles non-square/singular $A$ (Singular Value Decomposition).
- **Woodbury identity** turns an $n\times n$ inverse into a $k\times k$ one for low-rank updates: key in sparse Gaussian Processes and Kalman filters.
- ML homes: Multivariate Gaussian density, Normalizing Flows, Gaussian Processes evidence, QDA, Newton steps.

## The math
**Inverse.** $A^{-1}$ is the unique matrix with $AA^{-1} = A^{-1}A = I_n$. Via the adjugate, $A^{-1} = \frac{1}{\det A}\operatorname{adj}(A)$ (theory only).

**Determinant (Leibniz).**
$$
\det A = \sum_{\pi\in S_n}\operatorname{sgn}(\pi)\prod_{i=1}^n A_{i,\pi(i)},
$$
where $S_n$ is the set of permutations of $\{1,\dots,n\}$ and $\operatorname{sgn}(\pi) = \pm1$ its parity. Equivalently $\det A = \prod_i\lambda_i$ (eigenvalues, with multiplicity).

**Change of variables.** If $\mathbf{z} = f(\mathbf{x})$ is a diffeomorphism with Jacobian $J_f = \partial f/\partial\mathbf{x}$,
$$
p_X(\mathbf{x}) = p_Z(f(\mathbf{x}))\,\big|\det J_f(\mathbf{x})\big|, \qquad \log p_X(\mathbf{x}) = \log p_Z(f(\mathbf{x})) + \log\big|\det J_f(\mathbf{x})\big|.
$$

**Gaussian log-density** with mean $\boldsymbol{\mu}$ and covariance $\Sigma\succ0$ in $d$ dimensions:
$$
\log\mathcal{N}(\mathbf{x};\boldsymbol{\mu},\Sigma) = -\tfrac12(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) - \tfrac12\log\det\Sigma - \tfrac{d}{2}\log2\pi .
$$

**Woodbury identity.** For $A\in\mathbb{R}^{n\times n}$, $U\in\mathbb{R}^{n\times k}$, $C\in\mathbb{R}^{k\times k}$, $V\in\mathbb{R}^{k\times n}$:
$$
(A+UCV)^{-1} = A^{-1} - A^{-1}U\big(C^{-1}+VA^{-1}U\big)^{-1}VA^{-1},
$$
which costs a $k\times k$ inverse when $A^{-1}$ is cheap (e.g. diagonal).

**Derivatives** (Matrix Calculus): $\partial\log|\det A|/\partial A = A^{-\top}$; $\;d(A^{-1}) = -A^{-1}(dA)A^{-1}$; Jacobi's formula $d\det A = \det A\,\operatorname{tr}(A^{-1}dA)$.

**Continuous flows (instantaneous change of variables).** For $\dot{\mathbf{z}} = f(\mathbf{z},t)$: $\dfrac{d\log p(\mathbf{z}(t))}{dt} = -\operatorname{tr}\Big(\dfrac{\partial f}{\partial\mathbf{z}}\Big)$.

## Run the code
```bash
pip install numpy scipy torch
python matrix_inverse_and_determinant.py
```

## Check yourself
<details><summary>What does the determinant represent geometrically, and when is a matrix invertible?</summary>

$|\det A|$ is the factor by which the linear map scales volumes; the sign indicates whether orientation is preserved. $\det A = 0$ means the columns are linearly dependent, so the map collapses space onto a lower-dimensional subspace and information is lost; hence no inverse exists. Equivalent conditions for invertibility: full rank, trivial null space, all eigenvalues non-zero, all singular values positive.
</details>

<details><summary>Why do normalizing flows need cheap Jacobian determinants, and how do they get them?</summary>

The flow's log-likelihood is $\log p_Z(f(\mathbf{x})) + \log|\det J_f(\mathbf{x})|$, which must be computed for every training example. A general $d\times d$ determinant costs $O(d^3)$, impossible for images with $d$ in the hundreds of thousands. Coupling and autoregressive layers make the Jacobian triangular, so the determinant is the product of the diagonal, $O(d)$. Glow's 1×1 convolutions use an LU parameterisation; continuous flows replace the log-det with a trace estimated stochastically.
</details>

<details><summary>How do you compute $\log\det\Sigma$ for a 5000×5000 covariance matrix stably?</summary>

Compute the Cholesky factor $\Sigma = LL^\top$ (adding small jitter to the diagonal if needed), then $\log\det\Sigma = 2\sum_i\log L_{ii}$. Computing `det` directly would overflow or underflow to inf or 0 at this size. Cholesky costs about $\tfrac13 n^3$ FLOPs and the same factor is reused for the quadratic form via triangular solves. For much larger matrices, stochastic Lanczos quadrature estimates log-det using only matrix-vector products.
</details>

<details><summary>A matrix has determinant $10^{-50}$. Is it nearly singular?</summary>

Not necessarily. $\det(0.3 I_{100})\approx 5\times10^{-53}$, yet this matrix has condition number 1 and is perfectly well-behaved. The determinant scales as $c^n$ under $A\mapsto cA$, so its magnitude says little about singularity. The right diagnostic is the condition number $\sigma_{\max}/\sigma_{\min}$ or the smallest singular value relative to the largest.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.