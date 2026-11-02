# Probability Basics

**Level:** Beginner  ·  **Topic:** Math Foundations / Probability & Statistics

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Probability is a consistent way to assign numbers between 0 and 1 to uncertain events, built on three axioms. Every ML model that outputs a "confidence", every loss derived from likelihood, and every sampling-based generator rests on this foundation.

## The intuition
Think of a weather app that says "70% chance of rain". It is not saying it will rain 70% of the day. It is saying: on days that look like today, it rained on about 7 out of every 10. Probability is just a careful way of saying "how often, out of all the ways things could go".

Three ideas do almost all the work:

1. The **sample space** is the list of everything that could happen. For one die: {1, 2, 3, 4, 5, 6}.
2. An **event** is any group of those outcomes you care about, like "roll an even number" = {2, 4, 6}.
3. The **probability** of an event is a number from 0 (impossible) to 1 (certain). For a fair die, each face gets 1/6, so "even" gets 3/6 = 0.5.

Two simple rules follow. The chance something *doesn't* happen is 1 minus the chance it does: P(not six) = 1 − 1/6 = 5/6. And if two events can't both happen (rolling a 1 or a 2), you can add their chances: 1/6 + 1/6 = 1/3.

When a phone's photo app says "92% sure this is a cat", it is producing exactly this kind of number. Learning how to make those numbers honest is a big part of machine learning.

## Core notes
- **Axioms:** $P(A)\ge 0$, $P(\Omega)=1$, $P(\bigcup_i A_i)=\sum_i P(A_i)$ for disjoint $A_i$.
- **Complement:** $P(A^c) = 1 - P(A)$. Often the easiest route ("at least one" problems).
- **Inclusion–exclusion:** $P(A\cup B) = P(A)+P(B)-P(A\cap B)$; the union bound $P(\bigcup A_i)\le\sum P(A_i)$ drives much of PAC Learning.
- **Conditional probability** $P(A\mid B)=P(A\cap B)/P(B)$ is the bridge to Bayes Theorem and Joint, Marginal and Conditional Probability.
- **Equally likely outcomes:** $P(A)=|A|/|\Omega|$, which turns probability into counting (combinatorics).
- **Discrete vs continuous:** probabilities come from a mass function (sum) or a density (integral); see Random Variables.
- A classifier's softmax output is a distribution over classes; whether it is *honest* is a separate question (Model Calibration).
- Work in **log-probabilities** in code: products of many small probabilities underflow float32 quickly.
- Frequentist vs Bayesian is a disagreement about *what* gets a probability, not about the rules.

## The math
**Kolmogorov axioms.** For a probability space $(\Omega,\mathcal{F},P)$:
$$
\begin{aligned}
&P(A) \ge 0 \quad \forall A\in\mathcal{F}, \\
&P(\Omega) = 1, \\
&P\Big(\bigcup_{i=1}^{\infty} A_i\Big) = \sum_{i=1}^{\infty} P(A_i) \quad \text{for pairwise disjoint } A_i .
\end{aligned}
$$

**Consequences.** $P(\varnothing)=0$; $A\subseteq B \Rightarrow P(A)\le P(B)$; $P(A^c)=1-P(A)$; and inclusion–exclusion
$$
P(A\cup B) = P(A) + P(B) - P(A\cap B).
$$

**Union bound (Boole's inequality).** For any events $A_1,\dots,A_n$:
$$
P\Big(\bigcup_{i=1}^{n} A_i\Big) \le \sum_{i=1}^{n} P(A_i).
$$
In learning theory, $A_i$ = "hypothesis $h_i$ looks good on the sample but is bad"; the bound gives $|\mathcal{H}|\,e^{-2n\epsilon^2}$-style guarantees.

**Conditional probability and product rule.** For $P(B)>0$:
$$
P(A\mid B) = \frac{P(A\cap B)}{P(B)}, \qquad P(A\cap B) = P(A\mid B)\,P(B).
$$

**Law of total probability.** If $\{B_k\}$ partitions $\Omega$: $P(A)=\sum_k P(A\mid B_k)P(B_k)$.

## Run the code
```bash
pip install numpy
python probability_basics.py
```

## Check yourself
<details><summary>State the Kolmogorov axioms and derive P(A ∪ B) = P(A) + P(B) − P(A ∩ B).</summary>

The axioms are non-negativity, unit total mass $P(\Omega)=1$, and countable additivity for disjoint events. Write $A\cup B$ as the disjoint union $A \cup (B\setminus A)$, so $P(A\cup B)=P(A)+P(B\setminus A)$. Also $B$ is the disjoint union of $A\cap B$ and $B\setminus A$, so $P(B\setminus A) = P(B) - P(A\cap B)$. Substituting gives the result. The correction term removes the double-counted overlap.
</details>

<details><summary>What is the probability of getting at least one six in four rolls of a fair die?</summary>

Use the complement: P(no six in one roll) = 5/6, and rolls are independent, so P(no six in four) = $(5/6)^4 \approx 0.482$. Therefore P(at least one six) = $1 - 0.482 \approx 0.518$. This is de Méré's classic problem, and the complement trick is the standard approach for "at least one" questions.
</details>

<details><summary>If a classifier outputs 0.9 for class "cat", is the probability that the image is a cat 0.9?</summary>

Only if the model is calibrated. The softmax output is a valid distribution (non-negative, sums to 1), but modern deep networks trained with cross-entropy are often overconfident, especially after long training or under distribution shift. You check with a reliability diagram or expected calibration error and fix with temperature scaling or similar. See Model Calibration.
</details>

<details><summary>Why do ML implementations work with log-probabilities?</summary>

Products of many probabilities (e.g. a sequence likelihood over thousands of tokens) underflow to zero in floating point. Logs turn products into sums, which are numerically stable and easier to differentiate. Normalisation is done with the log-sum-exp trick, subtracting the max before exponentiating. Losses like negative log-likelihood and cross-entropy are written in log space for exactly this reason.
</details>

<details><summary>Explain the frequentist vs Bayesian interpretation in one minute.</summary>

Frequentists define probability as long-run frequency, so fixed but unknown parameters do not have probabilities; inference uses estimators, confidence intervals, and p-values. Bayesians treat probability as degree of belief, so parameters get priors and posteriors via Bayes' theorem. Both obey the same axioms. In ML, MLE and hypothesis tests are frequentist tools, while priors, MAP, and Bayesian neural nets are Bayesian; regularisation sits between the two.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.