# Matrix Multiplication

**Level:** Beginner  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Multiplying matrix $A$ ($m\times n$) by $B$ ($n\times p$) gives an $m\times p$ matrix whose $(i,j)$ entry is the dot product of row $i$ of $A$ with column $j$ of $B$. It composes linear maps, and it is where more than 90% of the FLOPs of modern neural networks are spent.

## The intuition
Imagine a café. A table lists how many of each item two customers ordered:

| | coffee | muffin |
|---|---|---|
| Ana | 2 | 1 |
| Ben | 0 | 3 |

A second table lists the price and the calories of each item:

| | price | calories |
|---|---|---|
| coffee | 3 | 5 |
| muffin | 2 | 400 |

Ana's bill is `2×3 + 1×2 = 8`. Her calories are `2×5 + 1×400 = 410`. Ben's bill is `0×3 + 3×2 = 6`, calories `0×5 + 3×400 = 1200`. You just did **matrix multiplication**: every answer is "a row from the first table" combined with "a column from the second table" by multiplying matching numbers and adding (a dot product). The result is a new 2×2 table: customers × (price, calories).

The rule for when it works: the middle sizes must match. The first table is "customers × items" and the second is "items × facts", so "items" cancels and you get "customers × facts".

Order matters: Orders × Prices makes sense, but Prices × Orders would combine the wrong things. Neural networks do exactly this, millions of times per second: a table of inputs times a table of learned weights gives a table of outputs.

## Core notes
- Shapes: $(m\times n)(n\times p) = m\times p$. Inner dimensions must match; check them first.
- Entry rule: $(AB)_{ij} = \text{row}_i(A)\cdot\text{col}_j(B)$ (Vector Operations and Dot Product).
- **Not commutative**: $AB \ne BA$ in general; **associative**: $(AB)C = A(BC)$, but the FLOP count depends on the parenthesisation.
- Matrix-vector product $A\mathbf{x}$ = a linear combination of $A$'s columns weighted by $\mathbf{x}$.
- A dense layer is $Y = XW^\top + \mathbf{b}$ with $X\in\mathbb{R}^{B\times d_\text{in}}$ (Multilayer Perceptron).
- Attention is two batched matmuls: $QK^\top$ then $\text{softmax}(\cdot)V$ (Self-Attention).
- Cost: $2mnp$ FLOPs, $O(mn+np+mp)$ memory: high arithmetic intensity, so GPUs love it (GPU Architecture and CUDA).
- Outer-product view: $AB = \sum_k \mathbf{a}_k\mathbf{b}_k^\top$ makes low-rank structure visible.
- Element-wise (Hadamard) product `*` is a different operation from matmul `@`.

## The math
**Definition.** For $A\in\mathbb{R}^{m\times n}$, $B\in\mathbb{R}^{n\times p}$:
$$
C = AB \in \mathbb{R}^{m\times p}, \qquad C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}.
$$

**Properties.** $(AB)C = A(BC)$; $A(B+C) = AB + AC$; $(AB)^\top = B^\top A^\top$; $(AB)^{-1} = B^{-1}A^{-1}$ for invertible square matrices; $\operatorname{tr}(AB) = \operatorname{tr}(BA)$ (cyclic trace); $\operatorname{rank}(AB) \le \min(\operatorname{rank}A, \operatorname{rank}B)$. The identity $I$ satisfies $AI = IA = A$.

**Outer-product expansion.**
$$
AB = \sum_{k=1}^{n} \mathbf{a}_{:k}\,\mathbf{b}_{k:}^{\top}
$$
where $\mathbf{a}_{:k}$ is column $k$ of $A$ and $\mathbf{b}_{k:}$ is row $k$ of $B$.

**Gradients of a linear layer.** For $Y = XW$ with $X\in\mathbb{R}^{B\times n}$, $W\in\mathbb{R}^{n\times p}$ and upstream gradient $G = \partial\mathcal{L}/\partial Y$:
$$
\frac{\partial\mathcal{L}}{\partial W} = X^\top G, \qquad \frac{\partial\mathcal{L}}{\partial X} = G\,W^\top.
$$
So the backward pass is two more matmuls of the same size, which is why training costs about $3\times$ the forward FLOPs (Matrix Calculus, Backpropagation).

**Arithmetic intensity** of square GEMM ($n\times n$, $b$ bytes/element): $\dfrac{2n^3}{3n^2 b} = \dfrac{2n}{3b}$ FLOPs/byte, growing with $n$; element-wise ops have constant intensity $O(1)$.

**Chained products.** Optimal parenthesisation of $A_1A_2\cdots A_k$ is the classic $O(k^3)$ dynamic program; `np.linalg.multi_dot` and `einsum(optimize=True)` implement it.

## Run the code
```bash
pip install numpy torch
python matrix_multiplication.py
```

## Check yourself
<details><summary>How many FLOPs does multiplying an $m\times n$ matrix by an $n\times p$ matrix take, and how does that translate to a transformer's training cost?</summary>

Each of the $mp$ outputs needs $n$ multiplies and $n$ adds, so about $2mnp$ FLOPs. A linear layer with $N_\ell$ weights processing one token costs $2N_\ell$ forward FLOPs, and the backward pass needs two more matmuls of the same size (input gradient and weight gradient), for about $6N_\ell$ in total. Summing over layers and tokens gives the standard $\approx 6ND$ estimate for training a dense model with $N$ parameters on $D$ tokens (attention adds a $T^2$ term that matters at long context).
</details>

<details><summary>Why are matmuls compute-bound while element-wise operations are memory-bound on GPUs?</summary>

A square matmul does $2n^3$ FLOPs on $3n^2$ numbers, so each number loaded can be reused $O(n)$ times via tiling; arithmetic intensity grows with size and exceeds the hardware's FLOPs-per-byte ratio. An element-wise op like GELU does a constant number of FLOPs per element loaded, so it is limited by memory bandwidth. This is why kernel fusion (combining element-wise ops into the matmul epilogue) and FlashAttention give big speed-ups.
</details>

<details><summary>Derive the gradient of $\mathcal{L}$ with respect to $W$ for $Y = XW$.</summary>

$Y_{bj} = \sum_i X_{bi}W_{ij}$, so $\partial Y_{bj}/\partial W_{ik} = X_{bi}\,\delta_{jk}$. By the chain rule $\partial\mathcal{L}/\partial W_{ik} = \sum_{b,j} G_{bj}X_{bi}\delta_{jk} = \sum_b X_{bi}G_{bk} = (X^\top G)_{ik}$. A quick shape check confirms: $X^\top$ is $n\times B$, $G$ is $B\times p$, product is $n\times p$ like $W$. Similarly $\partial\mathcal{L}/\partial X = GW^\top$.
</details>

<details><summary>Given $A, B \in \mathbb{R}^{n\times n}$ and $\mathbf{x}\in\mathbb{R}^n$, what is the cheapest way to compute $AB\mathbf{x}$?</summary>

Compute $A(B\mathbf{x})$: two matrix-vector products at $2n^2$ FLOPs each, versus $2n^3$ for forming $AB$ first. Associativity guarantees the same result in exact arithmetic. The same principle underlies reverse-mode autodiff (vector-Jacobian products instead of full Jacobians) and Hessian-vector products in second-order methods.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.