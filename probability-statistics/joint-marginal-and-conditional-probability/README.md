# Joint, Marginal and Conditional Probability

**Level:** Beginner  ·  **Topic:** Math Foundations / Probability & Statistics

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> The joint distribution describes several variables at once, marginalising sums out the ones you don't care about, and conditioning restricts to a known value and renormalises. The chain rule $p(x_1,\dots,x_T)=\prod_t p(x_t\mid x_{<t})$ is literally how every GPT-style language model is defined and trained.

## The intuition
Imagine a school survey of 100 students asking two things: do you prefer cats or dogs, and are you in junior or senior year? Put the answers in a grid:

|  | Cats | Dogs | Row total |
|---|---|---|---|
| Junior | 20 | 30 | 50 |
| Senior | 25 | 25 | 50 |
| Column total | 45 | 55 | 100 |

- **Joint probability** is one cell: P(junior AND cats) = 20/100 = 0.20.
- **Marginal probability** is a total along the edge (the "margin" of the table, which is where the name comes from): P(cats) = 45/100 = 0.45. You got it by adding up a column, ignoring year.
- **Conditional probability** is zooming into one row or column: P(cats GIVEN junior) = 20/50 = 0.40. Once you know someone is a junior, only the 50 juniors count.

The **chain rule** glues these together: P(junior AND cats) = P(junior) × P(cats given junior) = 0.5 × 0.4 = 0.2. Same answer as reading the cell.

A chatbot writing a sentence uses exactly this chain rule: the chance of "the cat sat" = P("the") × P("cat" given "the") × P("sat" given "the cat"). It picks each word by looking at what came before.

## Core notes
- **Joint** $p(x,y)$: everything about $X$ and $Y$ together; a table, or a density over $\mathbb{R}^2$.
- **Sum rule (marginalisation):** $p(x)=\sum_y p(x,y)$ or $\int p(x,y)\,dy$.
- **Product rule:** $p(x,y)=p(y\mid x)p(x)=p(x\mid y)p(y)$; equating the two gives Bayes Theorem.
- **Conditional** $p(y\mid x)=p(x,y)/p(x)$: slice the joint at $x$ and renormalise.
- **Chain rule:** $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$, the definition of an autoregressive LM (GPT Series, N-gram Language Models).
- **Law of total probability:** $p(y)=\sum_x p(y\mid x)p(x)$, the forward pass of mixture and latent-variable models.
- Discriminative models learn $p(y\mid x)$; generative models learn $p(x,y)$ or $p(x)$ (Generative vs Discriminative Models).
- Marginalising latent variables is usually intractable → EM, VI, MCMC.
- Conditioning on more variables can increase *or* decrease dependence (Independence and Conditional Independence).

## The math
**Sum and product rules.**
$$
p(x) = \int p(x,y)\,dy, \qquad p(x,y) = p(y\mid x)\,p(x).
$$

**Conditional.** For $p(x)>0$:
$$
p(y\mid x) = \frac{p(x,y)}{\int p(x,y')\,dy'}.
$$

**Chain rule (autoregressive factorisation).**
$$
p(x_1,\dots,x_T) = \prod_{t=1}^{T} p(x_t\mid x_1,\dots,x_{t-1}),
\qquad
\log p(x_{1:T}) = \sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$
The LLM pretraining loss is the negative of this sum averaged over tokens; its exponential is Perplexity.

**Latent-variable marginal.** For latent $z$ with prior $p(z)$:
$$
p_\theta(x) = \int p_\theta(x\mid z)\,p(z)\,dz,
$$
generally intractable for neural $p_\theta(x\mid z)$, hence lower bounds (Variational Inference).

**Discrete case with a table.** If $P\in\mathbb{R}^{m\times k}$ holds $P_{ij}=p(x_i,y_j)$: marginals are row/column sums $P\mathbf{1}$ and $P^\top\mathbf{1}$; conditionals are $P_{ij}/\sum_{j'}P_{ij'}$.

## Run the code
```bash
pip install numpy torch
python joint_marginal_and_conditional_probability.py
```

## Check yourself
<details><summary>How is a decoder-only language model a direct application of the chain rule?</summary>

Any joint distribution over a sequence factorises as $\prod_t p(x_t\mid x_{<t})$. A decoder-only transformer outputs, at every position, a softmax over the vocabulary representing $p_\theta(x_{t+1}\mid x_{\le t})$, enforced by a causal attention mask. Training minimises the summed negative log of these conditionals, which is exact maximum likelihood of the joint. Sampling generates tokens one at a time from the conditionals.
</details>

<details><summary>Why is computing p(x) in a latent-variable model hard, and what do we do about it?</summary>

$p(x)=\int p(x\mid z)p(z)\,dz$ requires integrating over all latent configurations, which has no closed form when $p(x\mid z)$ is a neural network and is high-dimensional. Options: EM when the posterior is tractable (GMMs), variational inference that optimises a lower bound (ELBO, used in VAEs), importance sampling for evaluation (IWAE), or MCMC. Diffusion models sidestep this by fixing the inference process and training on a bound.
</details>

<details><summary>Given P(rain)=0.3, P(umbrella|rain)=0.9, P(umbrella|no rain)=0.2, what is P(umbrella) and P(rain|umbrella)?</summary>

By total probability, P(umbrella) = 0.9·0.3 + 0.2·0.7 = 0.27 + 0.14 = 0.41. By Bayes, P(rain | umbrella) = 0.27 / 0.41 ≈ 0.659. The joint P(rain, umbrella) = 0.27 is the shared numerator. This is the template for most conditioning problems: compute joint cells, marginalise, renormalise.
</details>

<details><summary>What is the difference between a discriminative and a generative classifier in terms of these distributions?</summary>

A discriminative classifier models $p(y\mid x)$ directly (logistic regression, most neural classifiers). A generative classifier models $p(x\mid y)$ and $p(y)$, then applies Bayes to get $p(y\mid x)$ (Naive Bayes, LDA). Generative models can handle missing inputs and generate data and often win with little data; discriminative models usually win asymptotically because they don't spend capacity on modelling $p(x)$.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.