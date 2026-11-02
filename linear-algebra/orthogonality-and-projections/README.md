# Orthogonality and Projections

**Level:** Intermediate  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Two vectors are orthogonal when their dot product is zero; projecting a vector onto a subspace finds the closest point in that subspace, leaving an error that is orthogonal to it. This one idea gives least squares, PCA, Gram–Schmidt/QR, orthogonal initialisation, and many interpretability and model-editing tricks.

## The intuition
Stand in the sun at noon and hold a stick at a slant. Its **shadow** on the ground is its **projection**: the part of the stick that lies along the ground. The sunlight hits the ground straight down, at a right angle, and that's the key: the line from the tip of the stick to the tip of its shadow is **perpendicular** (at 90°) to the ground. In maths, "perpendicular" is called **orthogonal**, and it means the dot product is zero.

Example with numbers. A stick from the origin to the point `(3, 4)`, with the ground being the x-axis. The shadow is `(3, 0)`. The leftover piece from shadow to tip is `(0, 4)`, which points straight up, at right angles to the ground. Check: `(3, 0) · (0, 4) = 0`. Of all the points on the ground, the shadow `(3, 0)` is the one closest to the tip of the stick.

That "closest point" idea is how computers fit a line through messy data. The data can't all lie exactly on one line, so we find the "shadow" of the data on the set of all possible lines. The leftover errors are at right angles to everything the line could explain, meaning nothing more can be squeezed out of them. Directions at right angles are also handy because they don't interfere: moving east never changes how far north you are.

## Core notes
- Orthogonal: $\mathbf{u}^\top\mathbf{v} = 0$. Orthonormal: orthogonal and unit length.
- Projection of $\mathbf{b}$ onto a unit vector $\mathbf{u}$: $(\mathbf{u}^\top\mathbf{b})\,\mathbf{u}$; onto any $\mathbf{a}$: $\frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\mathbf{a}$.
- Projector onto $\operatorname{col}(A)$: $P = A(A^\top A)^{-1}A^\top$; with orthonormal $Q$: $P = QQ^\top$. Properties: $P^2 = P$, $P^\top = P$.
- Complement projector $I - P$ removes the component in the subspace ("projecting out").
- **Least squares = projection**: fitted values $\hat{\mathbf{y}} = P\mathbf{y}$, residual $\perp$ every feature (Linear Regression, Systems of Linear Equations).
- **Orthogonal matrices** preserve lengths and angles; inverse is the transpose; condition number 1.
- **Gram–Schmidt** → QR decomposition; use Householder or modified GS in practice (Matrix Decompositions).
- PCA projects data onto the top-$k$ orthonormal directions of variance.
- Orthogonal initialisation and orthogonality constraints stabilise deep and recurrent nets (Weight Initialization, Vanishing and Exploding Gradients).
- Concept erasure / steering: project activations off (or along) a learned direction (Mechanistic Interpretability).

## The math
**Projection onto a vector.** For $\mathbf{a}\ne\mathbf{0}$,
$$
\operatorname{proj}_{\mathbf{a}}(\mathbf{b}) = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a} = \frac{\mathbf{a}\mathbf{a}^\top}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{b}.
$$

**Projection onto a subspace.** For $A\in\mathbb{R}^{m\times n}$ with full column rank, the orthogonal projector onto $\operatorname{col}(A)$ is
$$
P = A(A^\top A)^{-1}A^\top = QQ^\top \quad (A = QR),
$$
with $P^2 = P$, $P^\top = P$, $\operatorname{rank}P = \operatorname{tr}P = n$.

**Projection theorem.** $\mathbf{p} = P\mathbf{b}$ uniquely solves $\min_{\mathbf{s}\in\operatorname{col}A}\|\mathbf{b}-\mathbf{s}\|_2$, and $\mathbf{b}-\mathbf{p}\perp\operatorname{col}A$. Pythagoras: $\|\mathbf{b}\|^2 = \|P\mathbf{b}\|^2 + \|(I-P)\mathbf{b}\|^2$. In regression this is $\text{TSS} = \text{ESS} + \text{RSS}$ (after centring), giving $R^2 = \text{ESS}/\text{TSS}$.

**Orthogonal matrices.** $Q^\top Q = I\Rightarrow\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$, $\langle Q\mathbf{x},Q\mathbf{y}\rangle = \langle\mathbf{x},\mathbf{y}\rangle$, $\kappa_2(Q) = 1$, all singular values equal 1.

**Gram–Schmidt.**
$$
\mathbf{v}_j = \mathbf{a}_j - \sum_{i<j}(\mathbf{q}_i^\top\mathbf{a}_j)\,\mathbf{q}_i, \qquad \mathbf{q}_j = \mathbf{v}_j/\|\mathbf{v}_j\|_2, \qquad R_{ij} = \mathbf{q}_i^\top\mathbf{a}_j .
$$

**Orthogonal Procrustes.** $\arg\min_{Q^\top Q = I}\|QA-B\|_F = UV^\top$ where $BA^\top = U\Sigma V^\top$.

**Concept erasure along a unit direction $\mathbf{u}$.** $\mathbf{h}' = (I-\mathbf{u}\mathbf{u}^\top)\mathbf{h}$, so $\mathbf{u}^\top\mathbf{h}' = 0$ for every activation $\mathbf{h}$.

## Run the code
```bash
pip install numpy torch
python orthogonality_and_projections.py
```

## Check yourself
<details><summary>Show that least squares is an orthogonal projection.</summary>

The least-squares fit minimises $\|\mathbf{y}-X\mathbf{w}\|$ over all vectors $X\mathbf{w}$ in the column space of $X$, which is exactly the "closest point in a subspace" problem. The optimality condition $X^\top(\mathbf{y}-X\mathbf{w}) = \mathbf{0}$ says the residual is orthogonal to every column. Substituting $\mathbf{w} = (X^\top X)^{-1}X^\top\mathbf{y}$ gives $\hat{\mathbf{y}} = X(X^\top X)^{-1}X^\top\mathbf{y} = H\mathbf{y}$, where the "hat matrix" $H$ is symmetric and idempotent, i.e. an orthogonal projector. Its diagonal entries are the leverages used in regression diagnostics.
</details>

<details><summary>What properties characterise an orthogonal projection matrix?</summary>

$P^2 = P$ (projecting twice changes nothing) and $P^\top = P$ (symmetric, which makes the residual orthogonal rather than oblique). Its eigenvalues are only 0 and 1, its trace equals its rank (the subspace dimension), and $I-P$ is the projector onto the orthogonal complement. Any such $P$ can be written $QQ^\top$ for an orthonormal basis $Q$ of its range.
</details>

<details><summary>Why initialise weights orthogonally, and when does it help?</summary>

An orthogonal matrix has all singular values equal to 1, so it preserves the norm of forward activations and backward gradients exactly. In deep linear networks and RNNs (where the same matrix is applied many times), this prevents exponential growth or decay of signals; Saxe et al. showed it yields depth-independent learning speed in deep linear nets. For ReLU networks with normalisation and residual connections the benefit over scaled Gaussian init is smaller, but orthogonal init remains common for RNNs and RL policy networks.
</details>

<details><summary>How would you remove a concept (e.g. gender) linearly from embeddings, and what are the limits?</summary>

Fit a linear probe or compute a difference-of-means direction $\mathbf{u}$ for the concept, then apply $(I-\mathbf{u}\mathbf{u}^\top)$ to every embedding so the component along $\mathbf{u}$ is zero. Iterate (INLP) or use closed-form methods (LEACE) to remove all linearly decodable information. Limits: non-linear probes may still recover the concept; correlated directions leak information; erasure can damage useful features; and it doesn't change what the downstream model was trained on.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.