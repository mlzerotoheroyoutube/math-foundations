# Systems of Linear Equations

**Level:** Beginner  ·  **Topic:** Math Foundations / Linear Algebra

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A system of linear equations $A\mathbf{x} = \mathbf{b}$ asks which combination of $A$'s columns produces $\mathbf{b}$. It has zero, one or infinitely many solutions depending on rank; when there are more equations than unknowns (the usual case with data) we solve it in the least-squares sense, which is exactly linear regression.

## The intuition
You buy 2 apples and 1 banana for 5 dollars. Your friend buys 1 apple and 3 bananas for 5 dollars. How much is each fruit? That's a **system of linear equations**: two unknowns (apple price $a$, banana price $b$), two clues.

- $2a + 1b = 5$
- $1a + 3b = 5$

Solve by elimination: double the second clue to get $2a + 6b = 10$, subtract the first clue, and you're left with $5b = 5$, so a banana costs 1 dollar. Put that back in: $2a + 1 = 5$, so an apple costs 2 dollars.

Three things can happen. With good clues you get **exactly one answer**. If the clues contradict each other ("2 apples cost 4" and "2 apples cost 6"), there's **no answer**. If one clue is just the other clue doubled, it adds no information and there are **infinitely many answers**.

In machine learning you usually have *too many* clues: thousands of houses (each one an equation) but only a few unknowns (how much each bedroom, each square metre adds to the price). The clues never agree perfectly because real data is noisy, so instead you find the answer that is **least wrong overall**. That "least wrong" answer is what linear regression computes.

## Core notes
- Three views of $A\mathbf{x}=\mathbf{b}$: $m$ equations (rows), a combination of columns equal to $\mathbf{b}$, or "which input does the map $A$ send to $\mathbf{b}$?".
- Solution count: none ($\mathbf{b}\notin\operatorname{col}A$), one (full column rank and consistent), infinite (non-trivial null space).
- **Gaussian elimination** = row operations to upper-triangular form, then back-substitution; $O(n^3)$.
- **Never compute $A^{-1}$ to solve**: use `np.linalg.solve` (LU) or `cho_solve`; it is faster and more accurate (Matrix Inverse and Determinant).
- Overdetermined ($m>n$, typical data): least squares, normal equations $A^\top A\mathbf{x}=A^\top\mathbf{b}$ = Linear Regression.
- Underdetermined ($m<n$, overparameterised models): infinitely many fits; the **minimum-norm** solution $A^+\mathbf{b}$ is what GD from zero finds.
- Adding $\lambda I$ to $A^\top A$ gives ridge regression and fixes ill-conditioning.
- Condition number $\kappa$ predicts how much noise is amplified (Norms and Distances).
- Big sparse systems: iterative solvers (CG, GMRES), which need only matrix-vector products.

## The math
**Problem.** Given $A\in\mathbb{R}^{m\times n}$, $\mathbf{b}\in\mathbb{R}^m$, find $\mathbf{x}\in\mathbb{R}^n$ with $A\mathbf{x}=\mathbf{b}$.

**Solvability (Rouché–Capelli).** Solutions exist iff $\operatorname{rank}A = \operatorname{rank}[A\,|\,\mathbf{b}]$; if so, the solution set is $\{\mathbf{x}_p + \mathbf{z} : \mathbf{z}\in\ker A\}$, unique iff $\ker A = \{\mathbf{0}\}$.

**Least squares and normal equations.** Minimise $f(\mathbf{x}) = \|A\mathbf{x}-\mathbf{b}\|_2^2$. Setting $\nabla f = 2A^\top(A\mathbf{x}-\mathbf{b}) = \mathbf{0}$ gives
$$
A^\top A\,\mathbf{x}^\star = A^\top\mathbf{b}, \qquad \mathbf{x}^\star = (A^\top A)^{-1}A^\top\mathbf{b}\ \ \text{if } \operatorname{rank}A = n .
$$
The residual $\mathbf{r} = \mathbf{b}-A\mathbf{x}^\star$ is orthogonal to every column of $A$: $A^\top\mathbf{r} = \mathbf{0}$.

**Minimum-norm solution.** For any $A$, $\mathbf{x}^+ = A^+\mathbf{b}$ with $A^+ = V\Sigma^+U^\top$ (from the SVD $A = U\Sigma V^\top$, $\Sigma^+$ inverts non-zero singular values) is the least-squares solution of smallest $\|\mathbf{x}\|_2$. For wide full-row-rank $A$, $A^+ = A^\top(AA^\top)^{-1}$.

**Sensitivity.** If $A\mathbf{x}=\mathbf{b}$ and $A(\mathbf{x}+\delta\mathbf{x}) = \mathbf{b}+\delta\mathbf{b}$,
$$
\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(A)\,\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}, \qquad \kappa(A) = \|A\|\,\|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}.
$$
Since $\kappa(A^\top A) = \kappa(A)^2$, normal equations lose twice the digits of QR.

**Ridge.** $\mathbf{x}_\lambda = (A^\top A+\lambda I)^{-1}A^\top\mathbf{b}$, and $\kappa(A^\top A+\lambda I) = \dfrac{\sigma_{\max}^2+\lambda}{\sigma_{\min}^2+\lambda}$.

**Conjugate gradient** for SPD $A$: after $k$ steps, $\|\mathbf{x}_k-\mathbf{x}^\star\|_A \le 2\Big(\tfrac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\Big)^k\|\mathbf{x}_0-\mathbf{x}^\star\|_A$.

## Run the code
```bash
pip install numpy scipy
python systems_of_linear_equations.py
```

## Check yourself
<details><summary>Derive the normal equations for least squares and interpret them geometrically.</summary>

Minimise $\|A\mathbf{x}-\mathbf{b}\|^2 = \mathbf{x}^\top A^\top A\mathbf{x} - 2\mathbf{b}^\top A\mathbf{x} + \mathbf{b}^\top\mathbf{b}$. The gradient is $2A^\top A\mathbf{x} - 2A^\top\mathbf{b}$; setting it to zero gives $A^\top A\mathbf{x} = A^\top\mathbf{b}$. Geometrically, $A\mathbf{x}^\star$ is the orthogonal projection of $\mathbf{b}$ onto the column space of $A$, and the residual is perpendicular to every column ($A^\top\mathbf{r}=\mathbf{0}$). The solution is unique iff $A$ has full column rank.
</details>

<details><summary>Why shouldn't you compute $A^{-1}\mathbf{b}$ explicitly?</summary>

Computing the inverse costs about $2n^3$ FLOPs versus $\tfrac{2}{3}n^3$ for an LU factorisation plus $O(n^2)$ triangular solves, so it is roughly three times slower. It is also less accurate: the computed inverse carries rounding error that is then multiplied into $\mathbf{b}$, whereas backward-stable LU with pivoting solves a nearby problem exactly. The factorisation can be reused for multiple right-hand sides, which is all the inverse would offer anyway.
</details>

<details><summary>What happens when you have more parameters than data points in linear regression?</summary>

The system $X\mathbf{w}=\mathbf{y}$ is underdetermined: infinitely many $\mathbf{w}$ fit the data exactly, and $X^\top X$ is singular so the normal equations have no unique solution. The pseudo-inverse gives the minimum-norm interpolant, which is also what gradient descent initialised at zero converges to, because GD iterates stay in the row space of $X$. Whether this interpolant generalises depends on the spectrum of the data covariance (benign overfitting); adding ridge regularisation gives a unique, stable solution.
</details>

<details><summary>What is the condition number, and how does it show up in ML?</summary>

$\kappa(A) = \sigma_{\max}/\sigma_{\min}$ bounds how much relative input error is amplified in the solution. Ill-conditioned design matrices arise from collinear or unscaled features and produce unstable, high-variance regression coefficients. For optimisation, the condition number of the Hessian governs how slowly gradient descent converges (zig-zagging in narrow valleys). Remedies: feature standardisation, ridge regularisation, QR/SVD solvers, preconditioning, or adaptive optimisers.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.