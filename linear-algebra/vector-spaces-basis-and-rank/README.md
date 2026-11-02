# Vector Spaces, Basis and Rank

**Level:** Intermediate  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A vector space is a set closed under adding and scaling; a basis is a minimal set of directions that can build every vector in it; and the rank of a matrix is the number of genuinely independent directions its columns span. Rank is how ML measures "how much information" a matrix really carries, which is why low-rank structure powers PCA, LoRA, recommender systems and compression.

## The intuition
Imagine a paint shop with three base colours: red, yellow and blue. By mixing different amounts you can make thousands of colours. The set of all colours you can mix is like a **vector space**, and the base paints are a **basis**: every colour is a recipe of "this much red, this much yellow, this much blue".

Now suppose someone hands you a fourth tube labelled "orange". It doesn't add anything new, because orange is just red plus yellow. In maths words, orange is **not independent** of the others. A basis is the smallest set of ingredients that still makes everything, with no wasted tubes. The number of tubes in a basis is the **dimension**: 3 here.

**Rank** is the same idea for a table of numbers. Take a table where each row is a student and each column is a test score: maths, physics, and "maths plus physics" (the total). The table has 3 columns, but the third is just the first two added, so only 2 columns carry real information. Its rank is 2.

Machine learning loves this. Netflix has a giant table of millions of users × thousands of films, but people's tastes are driven by a few hidden ingredients (likes action, likes comedy, likes old films…). So that huge table is "low rank", and that's what lets the computer guess which films you'd like.

## Core notes
- **Span** = all linear combinations; **independent** = no vector is a combination of the others; **basis** = independent + spanning; **dimension** = basis size.
- Coordinates are relative to a basis: the same vector has different coordinates in different bases.
- **Rank** = dim of column space = dim of row space = number of non-zero singular values = number of pivots.
- **Rank–nullity**: $\operatorname{rank}A + \dim\ker A = n$ (number of columns).
- Full rank: $\operatorname{rank} = \min(m,n)$. Rank-deficient matrices make $A^\top A$ singular (Systems of Linear Equations).
- Outer product $\mathbf{u}\mathbf{v}^\top$ has rank 1; a rank-$r$ matrix is a sum of $r$ of them.
- $W\in\mathbb{R}^{d\times d}$ rank $r$ can be stored as $UV^\top$ with $2dr$ numbers instead of $d^2$: the idea behind LoRA and QLoRA and compression.
- In floating point use **numerical rank** with a tolerance; `np.linalg.matrix_rank` does this via SVD.
- Collinear features (dummy-variable trap, duplicated columns) = rank deficiency (Feature Selection).
- Embeddings occupying a low-rank subspace signal **dimensional collapse** (Self-Supervised Learning).

## The math
**Linear independence.** $\mathbf{v}_1,\dots,\mathbf{v}_k$ are independent iff $\sum_{i=1}^k c_i\mathbf{v}_i = \mathbf{0}\Rightarrow c_1 = \cdots = c_k = 0$.

**Basis and coordinates.** If $B = [\mathbf{b}_1\cdots\mathbf{b}_n]$ is a basis of $\mathbb{R}^n$, every $\mathbf{x}$ has unique coordinates $\mathbf{c} = B^{-1}\mathbf{x}$ with $\mathbf{x} = B\mathbf{c}$.

**Rank and the fundamental theorem.** For $A\in\mathbb{R}^{m\times n}$ with rank $r$:
$$
\dim\operatorname{col}A = \dim\operatorname{row}A = r,\quad \dim\ker A = n-r,\quad \dim\ker A^\top = m-r,
$$
$$
\mathbb{R}^n = \operatorname{row}A\oplus\ker A,\qquad \mathbb{R}^m = \operatorname{col}A\oplus\ker A^\top \quad(\text{orthogonal direct sums}).
$$

**Rank via SVD.** $A = \sum_{i=1}^{r}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ with $\sigma_1\ge\cdots\ge\sigma_r>0$.

**Rank inequalities.** $\operatorname{rank}(AB)\le\min\{\operatorname{rank}A,\operatorname{rank}B\}$; $\operatorname{rank}(A+B)\le\operatorname{rank}A+\operatorname{rank}B$; $\operatorname{rank}(A^\top A) = \operatorname{rank}A$.

**Low-rank parameterisation (LoRA).** For a frozen $W_0\in\mathbb{R}^{d\times k}$,
$$
W = W_0 + \tfrac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k),
$$
so trainable parameters drop from $dk$ to $r(d+k)$ and $\operatorname{rank}(W-W_0)\le r$.

**Softmax bottleneck.** With hidden states $H\in\mathbb{R}^{N\times d}$ and output embeddings $E\in\mathbb{R}^{V\times d}$, the logit matrix $HE^\top$ has rank $\le d$, while the true log-probability matrix over $N$ contexts and $V$ tokens may need rank up to $\min(N,V)$.

**Effective rank.** With $p_i = \sigma_i/\sum_j\sigma_j$, $\operatorname{erank}(A) = \exp\big(-\sum_i p_i\log p_i\big)$.

## Run the code
```bash
pip install numpy torch
python vector_spaces_basis_and_rank.py
```

## Check yourself
<details><summary>State the rank–nullity theorem and give an ML interpretation.</summary>

For $A\in\mathbb{R}^{m\times n}$, $\operatorname{rank}A + \dim\ker A = n$. Interpreting $A$ as a design matrix with $n$ features: if the rank is less than $n$, there is a non-trivial null space, i.e. directions in weight space that change nothing about the predictions. Then the least-squares solution is not unique, coefficients are unidentifiable, and you need regularisation or feature removal. In an overparameterised network the analogous "flat directions" are why many parameter settings give the same function.
</details>

<details><summary>Why does LoRA work, and what does the rank $r$ control?</summary>

Empirically, the weight change needed to adapt a pretrained model to a downstream task has low intrinsic dimension, so it can be well approximated by a rank-$r$ matrix $BA$. Parameterising $\Delta W = BA$ with $r\ll d$ cuts trainable parameters and optimizer state from $d^2$ to $2dr$ per matrix, and the update can be merged into $W_0$ at inference with no latency cost. $r$ bounds the rank (expressivity) of the update; small $r$ (8–16) suffices for style or format adaptation, while larger shifts (new languages, heavy domain shift) benefit from larger $r$ or full fine-tuning.
</details>

<details><summary>How do you determine the rank of a matrix numerically?</summary>

Compute the singular values and count those above a tolerance, typically $\max(m,n)\cdot\epsilon_\text{machine}\cdot\sigma_{\max}$; this is what `numpy.linalg.matrix_rank` does. Gaussian elimination is unreliable because rounding turns exact zeros into tiny pivots. For a more robust continuous measure, use stable rank $\|A\|_F^2/\|A\|_2^2$ or effective rank based on the entropy of the singular values.
</details>

<details><summary>What is dimensional collapse in self-supervised learning?</summary>

The learned embeddings, though living in $\mathbb{R}^d$, span only a low-dimensional subspace: the covariance of embeddings has only a few significant eigenvalues. It wastes capacity and hurts downstream linear-probe performance; complete collapse (all embeddings identical) is the extreme case. Contrastive negatives, variance/covariance regularisers (VICReg), redundancy reduction (Barlow Twins), and predictor/stop-gradient tricks counteract it; monitor the effective rank of the embedding covariance to detect it.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.