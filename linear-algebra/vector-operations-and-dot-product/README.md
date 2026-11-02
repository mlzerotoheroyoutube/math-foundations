# Vector Operations and Dot Product

**Level:** Beginner  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Vectors add component-wise and scale by numbers; the dot product multiplies matching components and sums them, giving one number that measures how aligned two vectors are. It is the single most executed operation in ML: every neuron, every attention score and every similarity search is a dot product.

## The intuition
Picture a vector as an arrow: "walk 3 steps east and 4 steps north" is the arrow `[3, 4]`. **Adding** two arrows means doing one walk and then the other: `[3, 4] + [1, -2] = [4, 2]`. **Scaling** means walking further in the same direction: `2 × [3, 4] = [6, 8]`.

The **dot product** answers "how much do these two arrows point the same way?" You multiply matching entries and add: `[3, 4] · [1, 2] = 3×1 + 4×2 = 11`. A big positive number means "mostly the same direction", zero means "at right angles, unrelated", and negative means "opposite directions".

Here's a real use. A music app describes you by how much you like `[pop, rock, jazz]` = `[5, 1, 0]`. Song A is `[4, 0, 1]` and song B is `[0, 5, 3]`. Your score for A is `5×4 + 1×0 + 0×1 = 20`; for B it's `5×0 + 1×5 + 0×3 = 5`. Recommend song A. That's genuinely how many recommendation systems and search engines rank things, just with hundreds of numbers instead of three.

The **length** of an arrow comes from Pythagoras: `[3, 4]` has length √(9 + 16) = 5.

## Core notes
- Addition and scaling are element-wise; together they make linear combinations $\sum_i c_i\mathbf{v}_i$, the core idea of Vector Spaces, Basis and Rank.
- Dot product: $\mathbf{x}\cdot\mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$. Cost $O(n)$.
- Sign tells direction: $>0$ acute angle, $=0$ orthogonal (Orthogonality and Projections), $<0$ obtuse.
- **Cosine similarity** = dot product of unit-normalised vectors; range $[-1, 1]$; the default for comparing Embeddings.
- **Element-wise (Hadamard) product** $\mathbf{x}\odot\mathbf{y}$ returns a vector, not a scalar: used in gating (LSTM, SwiGLU) and masking.
- **Outer product** $\mathbf{x}\mathbf{y}^\top$ returns a rank-1 matrix: appears in weight-gradient formulas and LoRA and QLoRA updates.
- A linear layer is a stack of dot products: row $i$ of $W$ dotted with $\mathbf{x}$ (Matrix Multiplication).
- Scaled dot-product attention divides by $\sqrt{d_k}$ to keep logit variance near 1.
- In high dimensions random vectors are nearly orthogonal; this is a feature for hashing and superposition, a nuisance for kNN (Curse of Dimensionality).

## The math
**Operations.** For $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$, $c\in\mathbb{R}$: $(\mathbf{x}+\mathbf{y})_i = x_i+y_i$, $(c\mathbf{x})_i = c\,x_i$, $(\mathbf{x}\odot\mathbf{y})_i = x_iy_i$.

**Dot product (algebraic and geometric).**
$$
\mathbf{x}^\top\mathbf{y} = \sum_{i=1}^{n} x_i y_i = \|\mathbf{x}\|_2\,\|\mathbf{y}\|_2\cos\theta
$$
where $\theta$ is the angle between the vectors. The geometric form follows from the law of cosines applied to $\|\mathbf{x}-\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^\top\mathbf{y}$.

**Properties.** Symmetric $\mathbf{x}^\top\mathbf{y} = \mathbf{y}^\top\mathbf{x}$; bilinear $(a\mathbf{x}+b\mathbf{z})^\top\mathbf{y} = a\,\mathbf{x}^\top\mathbf{y} + b\,\mathbf{z}^\top\mathbf{y}$; positive definite $\mathbf{x}^\top\mathbf{x} > 0$ for $\mathbf{x}\ne\mathbf{0}$.

**Cauchy–Schwarz.** $|\mathbf{x}^\top\mathbf{y}| \le \|\mathbf{x}\|\|\mathbf{y}\|$. Proof sketch: $0 \le \|\mathbf{x} - t\mathbf{y}\|^2$ is a quadratic in $t$ with non-positive discriminant.

**Cosine similarity and distance link.** For unit vectors, $\|\mathbf{x}-\mathbf{y}\|^2 = 2 - 2\cos\theta$, so ranking by cosine equals ranking by Euclidean distance.

**Why attention scales by $\sqrt{d_k}$.** If $q_i, k_i$ are independent with mean 0 and variance 1,
$$
\operatorname{Var}\Big(\sum_{i=1}^{d_k} q_i k_i\Big) = \sum_{i=1}^{d_k}\operatorname{Var}(q_i)\operatorname{Var}(k_i) = d_k,
$$
so $\mathbf{q}^\top\mathbf{k}/\sqrt{d_k}$ has unit variance.

**Gradient.** $\nabla_{\mathbf{x}}(\mathbf{w}^\top\mathbf{x}) = \mathbf{w}$: the weight vector *is* the direction of steepest increase of a linear unit (Gradients).

## Run the code
```bash
pip install numpy torch
python vector_operations_and_dot_product.py
```

## Check yourself
<details><summary>What does the dot product measure geometrically, and how does cosine similarity differ?</summary>

$\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$: it combines the lengths of both vectors with how aligned they are. Cosine similarity divides out the lengths, leaving only the angle, so it ranges in $[-1,1]$ and is invariant to rescaling either vector. Use dot product when magnitude carries information (e.g. confidence or popularity in a recommender), cosine when only direction should matter. On L2-normalised vectors they coincide.
</details>

<details><summary>Why is attention scaled by $1/\sqrt{d_k}$?</summary>

If query and key entries are roughly independent with zero mean and unit variance, the dot product is a sum of $d_k$ terms each with variance 1, so its variance is $d_k$. For $d_k = 128$ the logits have standard deviation about 11, pushing the softmax into a near one-hot regime where gradients vanish. Dividing by $\sqrt{d_k}$ restores unit variance. Some modern models add QK-normalisation for the same stability reason.
</details>

<details><summary>Prove that two random high-dimensional vectors are nearly orthogonal.</summary>

Take $\mathbf{x},\mathbf{y}$ with i.i.d. $\mathcal{N}(0,1)$ entries. $\mathbf{x}^\top\mathbf{y}$ has mean 0 and variance $d$, while $\|\mathbf{x}\|^2$ and $\|\mathbf{y}\|^2$ concentrate around $d$. Hence $\cos\theta \approx \mathbf{x}^\top\mathbf{y}/d$, which has standard deviation about $1/\sqrt{d}$. At $d = 10{,}000$ the typical cosine is around 0.01, so random directions are almost orthogonal. This underlies random projections, locality-sensitive hashing and the superposition hypothesis.
</details>

<details><summary>What is the gradient of $\mathbf{w}^\top\mathbf{x}$ with respect to $\mathbf{w}$ and to $\mathbf{x}$, and why does it matter?</summary>

$\nabla_{\mathbf{w}} = \mathbf{x}$ and $\nabla_{\mathbf{x}} = \mathbf{w}$. The first means a linear layer's weight gradient is the input scaled by the upstream error, which is why weight gradients are outer products $\boldsymbol{\delta}\mathbf{x}^\top$ in Backpropagation. The second underlies saliency maps and adversarial examples: perturbing the input along $\mathbf{w}$ changes the output fastest (Adversarial Examples and Robustness).
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.