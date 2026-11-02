# Scalars, Vectors, Matrices and Tensors

**Level:** Beginner  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Scalars are single numbers, vectors are lists of numbers, matrices are grids, and tensors are grids with any number of axes. Every piece of data and every learned parameter in ML lives in one of these containers, and most bugs are shape bugs.

## The intuition
Think of containers for numbers, getting bigger each time.

A **scalar** is one number, like the temperature outside: `21`. A **vector** is a list of numbers in a fixed order, like a student's scores in three subjects: `[90, 75, 82]`. The order matters: position 1 is always maths, position 2 always English. A **matrix** is a grid, like a spreadsheet: one row per student, one column per subject. Three students and three subjects make a 3-by-3 grid with 9 numbers. A **tensor** is the same idea with more directions. A colour photo is a tensor: height × width × 3 colour channels (red, green, blue). A small 4×4 photo holds 4 × 4 × 3 = 48 numbers. A batch of 32 photos adds one more direction: 32 × 4 × 4 × 3.

Why should you care? Computers that "learn" store *everything* this way. A word can be turned into a vector of 768 numbers that captures its meaning. A neural network's knowledge is stored in huge matrices of numbers called weights. The **shape** (how many numbers along each direction) is like the label on a box: if you try to combine two boxes whose labels don't match, the program crashes, and figuring out shapes is half of the job.

## Core notes
- **Order / ndim** = number of axes: scalar 0, vector 1, matrix 2, tensor $\ge 3$. "Rank" in PyTorch/TF means ndim, not matrix rank.
- Notation: scalars $a$, vectors $\mathbf{x}$ (lower-case bold, column by default), matrices $A$ (upper-case), tensors $\mathcal{X}$.
- Canonical ML shapes: tabular $(N, D)$; images $(N, C, H, W)$ in PyTorch, $(N, H, W, C)$ in TF/JAX; text $(B, T, d_\text{model})$; attention scores $(B, h, T, T)$.
- **Broadcasting**: align shapes from the right; a size-1 axis stretches to match. Powerful and the top source of silent bugs.
- **Transpose** swaps axes; $(A^\top)_{ij} = A_{ji}$. `permute`/`transpose` usually return views with new strides, not copies.
- **Reshape/view** changes shape but not data order; requires contiguity for `view` in PyTorch.
- dtype matters: fp32 (4 bytes), bf16/fp16 (2), fp8/int8 (1). Memory = numel × bytes (Memory Estimation for Training and Inference, Mixed Precision Training).
- Every model is a composition of operations on tensors; Embeddings are rows of a matrix, weights are matrices, activations are tensors.
- Learn `einsum`: it states contractions explicitly and documents shapes.

## The math
**Containers.** $a \in \mathbb{R}$, $\mathbf{x} \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$, $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$ with entries $\mathcal{X}_{i_1 i_2 \dots i_k}$.

**Transpose.** $(A^\top)_{ij} = A_{ji}$, with $(AB)^\top = B^\top A^\top$.

**Einstein summation.** A repeated index is summed. Matrix product: $C_{ik} = \sum_j A_{ij}B_{jk}$, written `einsum("ij,jk->ik")`. Batched attention scores:
$$
S_{bhqk} = \sum_{d} Q_{bhqd}\, K_{bhkd}
$$
where $b$ is batch, $h$ head, $q,k$ query/key positions, $d$ head dimension.

**Outer (tensor) product.** For $\mathbf{u}\in\mathbb{R}^m, \mathbf{v}\in\mathbb{R}^n$: $(\mathbf{u}\otimes\mathbf{v})_{ij} = u_i v_j$, a rank-1 matrix $\mathbf{u}\mathbf{v}^\top$. An order-3 rank-1 tensor is $(\mathbf{a}\otimes\mathbf{b}\otimes\mathbf{c})_{ijk} = a_i b_j c_k$.

**Memory.** A tensor with shape $(d_1,\dots,d_k)$ and $b$ bytes per element occupies $b\prod_i d_i$ bytes. In row-major (C) layout the stride of axis $i$ is $\prod_{j>i} d_j$ elements, and the flat index is $\sum_i i_i \cdot \text{stride}_i$.

**Frobenius inner product.** $\langle A, B\rangle_F = \sum_{ij} A_{ij}B_{ij} = \operatorname{tr}(A^\top B)$, the natural dot product for matrices (Norms and Distances).

## Run the code
```bash
pip install numpy torch
python scalars_vectors_matrices_and_tensors.py
```

## Check yourself
<details><summary>What is the difference between a tensor's "rank" in PyTorch and the rank of a matrix?</summary>

In frameworks, "rank" (better: ndim or order) is the number of axes: a matrix always has ndim 2. Matrix rank is the dimension of the column space, i.e. the number of linearly independent columns, which can be anything from 0 to $\min(m,n)$. A 1000×1000 matrix of all ones has ndim 2 but matrix rank 1. Tensor rank (CP rank) is a third concept: the minimal number of rank-1 outer products summing to the tensor.
</details>

<details><summary>Explain broadcasting and give a bug it can cause.</summary>

Broadcasting compares shapes from the trailing axis backwards; two axes are compatible if equal or one of them is 1, and size-1 axes are virtually repeated. It lets you add a bias of shape `(D,)` to activations `(B, T, D)` without copying. The classic bug: predictions of shape `(N,1)` minus targets `(N,)` broadcasts to `(N,N)`, so an MSE loss averages $N^2$ wrong differences and training silently fails. Fix with explicit `squeeze`/`reshape` and shape assertions.
</details>

<details><summary>How much memory does a fp32 activation tensor of shape (8, 2048, 4096) take? In bf16?</summary>

Number of elements is $8 \times 2048 \times 4096 = 67{,}108{,}864 \approx 6.7\times 10^7$. At 4 bytes that is about 268 MB (256 MiB); at 2 bytes in bf16 about 134 MB. Multiply by the number of saved activations per layer and number of layers to see why Gradient Checkpointing and Mixed Precision Training matter.
</details>

<details><summary>Why does `x.transpose(0,1).view(-1)` fail in PyTorch while `reshape` works?</summary>

`transpose` returns a view with permuted strides, so the data is no longer laid out contiguously in the new axis order. `view` requires a compatible stride layout and refuses to copy. `reshape` returns a view when possible and otherwise copies into contiguous memory. Calling `.contiguous()` first also fixes it.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.