# Functions and Limits

**Level:** Beginner  ·  **Topic:** Math Foundations / Calculus

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> A function maps each input to exactly one output; a limit describes what the output approaches as the input approaches some value. Limits define continuity, derivatives and integrals, and in ML they explain why activations saturate, why softmax becomes argmax as temperature goes to zero, and why we use tricks like log-sum-exp to keep computations finite.

## The intuition
A **function** is a machine: you put something in, exactly one thing comes out. A vending machine is a function: press B4, you always get the same snack. "Double it and add 1" is a function: put in 3, get out 7.

A **limit** asks: "what does the output get closer and closer to, as the input gets closer and closer to some value?", even if you can never actually plug that value in. Imagine walking halfway to a wall, then halfway again, then halfway again. You never quite touch it, but your distance heads towards 0. We say the limit is 0.

A concrete one: look at $1/x$ as $x$ gets bigger: $1/10 = 0.1$, $1/100 = 0.01$, $1/1000 = 0.001$. The outputs approach 0, so the limit as $x$ grows forever is 0.

A function is **continuous** if you can draw it without lifting your pen: no jumps, no holes. Temperature through the day is continuous; the price of a parcel that jumps from 5 to 8 dollars once it weighs over 1 kg is not.

In AI, functions are everywhere: a whole neural network is one giant function from a photo to the word "cat". Limits matter because computers can't store infinitely big or infinitely small numbers. If a formula heads towards infinity, the computer prints "inf" or "NaN" (not a number) and training breaks, so engineers rewrite formulas to stay in safe territory.

## Core notes
- A function gives one output per input; a model $f_\theta$ is a function chosen from a family by training.
- Domain, codomain, range: e.g. $\log$ needs inputs $>0$; sigmoid's range is $(0,1)$.
- **Limit** $\lim_{x\to a}f(x) = L$: outputs get arbitrarily close to $L$ as $x$ approaches $a$ (value at $a$ irrelevant).
- **Continuous** at $a$: $\lim_{x\to a}f(x) = f(a)$. Compositions of continuous functions are continuous, so neural nets with continuous activations are continuous.
- **Lipschitz**: $|f(x)-f(y)|\le L|x-y|$, a bound on how fast the function can change; a stronger, quantitative form of continuity.
- Saturation: $\lim_{z\to\infty}\sigma(z) = 1$ with $\sigma'(z)\to0$, so gradients die (Activation Functions).
- Temperature: $\operatorname{softmax}(\mathbf{z}/\tau)\to$ argmax as $\tau\to0$; uniform as $\tau\to\infty$ (Softmax Function).
- Key limits: $\lim_{x\to0}\frac{\sin x}{x} = 1$, $\lim_{n\to\infty}(1+\frac{1}{n})^n = e$, $\lim_{x\to0}x\log x = 0$ (why $0\log0 := 0$ in Entropy).
- Floating-point overflow/underflow are "limits gone wrong": use log-sum-exp, `log1p`, `expm1`.
- Derivatives and integrals are both defined as limits (Derivatives, Integrals).

## The math
**$\varepsilon$–$\delta$ limit.** For $f:\mathbb{R}^n\to\mathbb{R}^m$,
$$
\lim_{\mathbf{x}\to\mathbf{a}}f(\mathbf{x}) = L \iff \forall\varepsilon>0\ \exists\delta>0:\ 0<\|\mathbf{x}-\mathbf{a}\|<\delta\Rightarrow\|f(\mathbf{x})-L\|<\varepsilon .
$$

**Continuity and Lipschitz continuity.** $f$ is continuous at $\mathbf{a}$ if $\lim_{\mathbf{x}\to\mathbf{a}}f(\mathbf{x}) = f(\mathbf{a})$; $L$-Lipschitz if $\|f(\mathbf{x})-f(\mathbf{y})\|\le L\|\mathbf{x}-\mathbf{y}\|$ for all $\mathbf{x},\mathbf{y}$. For compositions, $\operatorname{Lip}(g\circ f)\le\operatorname{Lip}(g)\operatorname{Lip}(f)$.

**Softmax temperature limit.** With $p_i(\tau) = e^{z_i/\tau}/\sum_je^{z_j/\tau}$ and a unique maximiser $k = \arg\max_iz_i$:
$$
\lim_{\tau\to0^+}p_i(\tau) = \mathbb{1}[i = k],\qquad \lim_{\tau\to\infty}p_i(\tau) = \tfrac{1}{n}.
$$
Proof sketch: divide numerator and denominator by $e^{z_k/\tau}$; each ratio $e^{(z_j-z_k)/\tau}\to0$ for $j\ne k$.

**Log-sum-exp stability.** For $m = \max_iz_i$,
$$
\log\sum_ie^{z_i} = m+\log\sum_ie^{z_i-m},
$$
where every exponent is $\le0$, so no overflow, and the largest term is exactly 1, so no total underflow. Also $\max_iz_i\le\operatorname{LSE}(\mathbf{z})\le\max_iz_i+\log n$, so LSE is a smooth max.

**Sigmoid saturation.** $\sigma'(z) = \sigma(z)(1-\sigma(z))\le\tfrac14$ and $\lim_{|z|\to\infty}\sigma'(z) = 0$.

**$0\log0$ convention.** By L'Hôpital, $\lim_{x\to0^+}x\log x = \lim\frac{\log x}{1/x} = \lim\frac{1/x}{-1/x^2} = 0$.

## Run the code
```bash
pip install numpy torch
python functions_and_limits.py
```

## Check yourself
<details><summary>What happens to softmax as temperature goes to 0 and to infinity? Why does it matter?</summary>

As $\tau\to0^+$ the distribution concentrates all mass on the largest logit (argmax), because every ratio $e^{(z_j-z_{\max})/\tau}$ goes to 0. As $\tau\to\infty$ all logits are scaled towards 0 and the distribution becomes uniform. In LLM sampling, low temperature gives near-greedy, repetitive output and high temperature gives diverse but less coherent output. In distillation, a higher temperature exposes the teacher's relative probabilities over wrong classes ("dark knowledge"), and Gumbel-softmax anneals $\tau$ to approach discrete samples.
</details>

<details><summary>Why is log-sum-exp computed by subtracting the maximum?</summary>

$e^{z}$ overflows in fp32 for $z$ above about 88 (and much sooner in fp16), and underflows to 0 for very negative $z$, making $\log 0 = -\infty$. Subtracting $m = \max_iz_i$ uses the identity $\log\sum e^{z_i} = m+\log\sum e^{z_i-m}$: all shifted exponents are $\le0$ and one equals 0, so the sum is between 1 and $n$. The result is mathematically identical and numerically safe. The same trick underlies stable softmax, cross-entropy and the online softmax in FlashAttention.
</details>

<details><summary>What is Lipschitz continuity and where does it appear in ML?</summary>

$f$ is $L$-Lipschitz if $\|f(x)-f(y)\|\le L\|x-y\|$: outputs can't change faster than $L$ times the input change. It appears in robustness (a small Lipschitz constant certifies that small input perturbations can't flip a prediction), in WGAN critics (the Kantorovich–Rubinstein dual requires 1-Lipschitz functions, enforced via weight clipping, gradient penalty or spectral normalisation), and in optimisation, where Lipschitz gradients ($L$-smoothness) justify step sizes like $1/L$ and give convergence rates.
</details>

<details><summary>Is a ReLU network continuous? Differentiable?</summary>

It is continuous, because ReLU is continuous and compositions, sums and affine maps preserve continuity. It is piecewise linear and not differentiable on the measure-zero set where some pre-activation equals exactly 0. Frameworks use a subgradient there (PyTorch uses 0 for ReLU at 0), which works because hitting exactly 0 is rare and the subgradient is a valid descent direction element for convex pieces; smooth alternatives like GELU and SiLU avoid the kink.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.