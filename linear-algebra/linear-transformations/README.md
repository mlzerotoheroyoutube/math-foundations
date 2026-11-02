# Linear Transformations

**Level:** Beginner  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A linear transformation is a function that maps vectors to vectors while preserving addition and scaling, so straight lines stay straight and the origin stays put. Every matrix is one, every linear map between finite-dimensional spaces is a matrix, and a neural network is a stack of them separated by non-linearities.

## The intuition
Draw a grid of squares on a rubber sheet, then stretch, squash, rotate or slant the sheet, but keep the centre pinned and keep the grid lines straight and evenly spaced. Any such move is a **linear transformation**. Squares can become rectangles or slanted diamonds, but lines never bend.

A matrix is the recipe for the move. The trick: you only need to know where two arrows go, the arrow pointing one step right `[1, 0]` and the arrow pointing one step up `[0, 1]`. Everything else follows. Take the matrix with columns `[2, 0]` and `[1, 1]`. It sends "one step right" to `[2, 0]` (stretched) and "one step up" to `[1, 1]` (slanted). Where does the point `[3, 2]` go? It's 3 rights and 2 ups, so it goes to `3×[2, 0] + 2×[1, 1] = [8, 2]`.

Rotating a phone photo, resizing an image, or turning "height in cm" into "height in inches" are all linear transformations. A neural network layer starts with one: it reshapes the space of inputs so that, after a few layers, cats end up on one side and dogs on the other. Since chaining linear moves just gives another linear move, the network also needs a bending step between layers, the **activation function**.

## Core notes
- Linear means $T(a\mathbf{x}+b\mathbf{y}) = aT(\mathbf{x})+bT(\mathbf{y})$; consequence: $T(\mathbf{0}) = \mathbf{0}$.
- **Columns are images of basis vectors**: column $j$ of $A$ is $T(\mathbf{e}_j)$. To build a matrix, ask where each basis vector goes.
- Composition = multiplication; applying $B$ then $A$ is $AB$ (right-to-left).
- Affine $= $ linear $+$ shift: $\mathbf{x}\mapsto W\mathbf{x}+\mathbf{b}$; a dense layer is affine (Multilayer Perceptron).
- Stacking linear layers without non-linearity is still linear: depth buys nothing.
- Examples in 2-D: scaling $\operatorname{diag}(s_1,s_2)$, rotation $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}$, shear, reflection, projection.
- Determinant = area/volume scale; zero means the map squashes space into a lower dimension (Matrix Inverse and Determinant).
- SVD: every matrix is rotation → axis scaling → rotation (Singular Value Decomposition).
- Convolution, Embeddings lookup, positional rotations (RoPE) and attention's value mixing are all linear in the right variables.

## The math
**Linearity.** $T:\mathbb{R}^n\to\mathbb{R}^m$ is linear iff $T(a\mathbf{x}+b\mathbf{y}) = aT(\mathbf{x})+bT(\mathbf{y})$ for all $a,b\in\mathbb{R}$, $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$.

**Matrix representation.** With standard basis $\mathbf{e}_j$,
$$
T(\mathbf{x}) = T\Big(\sum_{j=1}^n x_j\mathbf{e}_j\Big) = \sum_{j=1}^n x_j\,T(\mathbf{e}_j) = A\mathbf{x}, \qquad A = \big[\,T(\mathbf{e}_1)\ \cdots\ T(\mathbf{e}_n)\,\big].
$$

**Kernel, image, rank–nullity.** $\ker A = \{\mathbf{x}: A\mathbf{x} = \mathbf{0}\}$, $\operatorname{im}A = \{A\mathbf{x}\}$, and
$$
n = \dim\ker A + \operatorname{rank}A .
$$

**Change of basis.** If $P$'s columns are a new basis of $\mathbb{R}^n$, the same map in new coordinates is $A' = P^{-1}AP$; $A$ and $A'$ share eigenvalues, $\det$, $\operatorname{tr}$.

**Collapse of deep linear networks.** $W_L W_{L-1}\cdots W_1\mathbf{x} = W_\text{eff}\,\mathbf{x}$ with $\operatorname{rank}W_\text{eff}\le\min_\ell \operatorname{rank}W_\ell$, so depth adds no expressivity (it does change optimisation dynamics).

**Affine in homogeneous coordinates.**
$$
\begin{pmatrix}\mathbf{y}\\1\end{pmatrix} = \begin{pmatrix}W & \mathbf{b}\\ \mathbf{0}^\top & 1\end{pmatrix}\begin{pmatrix}\mathbf{x}\\1\end{pmatrix}.
$$

**Lipschitz constant.** $\|A\mathbf{x}-A\mathbf{y}\|_2\le\sigma_{\max}(A)\|\mathbf{x}-\mathbf{y}\|_2$; a network's Lipschitz constant is bounded by the product of layer spectral norms times activation Lipschitz constants (Norms and Distances).

## Run the code
```bash
pip install numpy torch
python linear_transformations.py
```

## Check yourself
<details><summary>Why does a neural network need non-linear activation functions?</summary>

A composition of affine maps is affine: $W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2 = (W_2W_1)\mathbf{x} + (W_2\mathbf{b}_1+\mathbf{b}_2)$. Without non-linearities, a 100-layer network has exactly the expressive power of one linear layer and cannot represent XOR or any curved decision boundary. Non-linearities let compositions approximate arbitrary continuous functions (Universal Approximation Theorem). Depth in linear networks still changes the optimisation dynamics, which is why they are used as a theoretical model.
</details>

<details><summary>How do you construct the matrix of a linear map?</summary>

Apply the map to each standard basis vector $\mathbf{e}_j$ and write the result as column $j$. By linearity, $T(\mathbf{x}) = \sum_j x_jT(\mathbf{e}_j) = A\mathbf{x}$. For example, rotation by 90° sends $\mathbf{e}_1 = (1,0)$ to $(0,1)$ and $\mathbf{e}_2 = (0,1)$ to $(-1,0)$, giving $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$.
</details>

<details><summary>Is $f(\mathbf{x}) = W\mathbf{x}+\mathbf{b}$ linear? Why does the distinction matter?</summary>

Strictly no, unless $\mathbf{b} = \mathbf{0}$: it's affine, since $f(\mathbf{0}) = \mathbf{b}\ne\mathbf{0}$ and $f(2\mathbf{x})\ne2f(\mathbf{x})$. It becomes linear in homogeneous coordinates by appending a constant 1 to $\mathbf{x}$. The distinction matters for things like whether the zero input maps to zero, how normalisation layers interact with biases (biases before BatchNorm are redundant), and for proofs that rely on exact linearity.
</details>

<details><summary>What does the linear representation hypothesis claim and what is the evidence?</summary>

It claims that high-level concepts in a trained network correspond to directions (or low-dimensional subspaces) in activation space, so that concepts can be read out by linear probes and manipulated by adding vectors. Evidence includes word-embedding analogies, high accuracy of linear probes, activation steering, and sparse-autoencoder features that are interpretable directions. Counter-evidence includes features that are inherently multi-dimensional or circular (e.g. days of the week) and cases where probes succeed without the model using that information, so it is a useful approximation rather than an established law.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.