"""Joint, Marginal and Conditional Probability: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

import numpy as np

# Survey table: rows = year (junior, senior), cols = pet (cats, dogs)
counts = np.array([[20, 30],
                   [25, 25]])
joint = counts / counts.sum()
p_year = joint.sum(axis=1)                  # marginal over pets
p_pet = joint.sum(axis=0)                   # marginal over years
p_pet_given_year = joint / p_year[:, None]  # condition on year
p_year_given_pet = joint / p_pet[None, :]

print("P(junior, cats) =", joint[0, 0])
print("P(cats)         =", p_pet[0])
print("P(cats | junior)=", p_pet_given_year[0, 0])
print("P(junior | cats)=", round(p_year_given_pet[0, 0], 4))
print("chain rule check:", np.allclose(joint, p_year[:, None] * p_pet_given_year))

# --------------------------------------------------------------------------

import torch
import torch.nn.functional as F

# The chain rule as an LLM loss: log p(sequence) = sum of next-token log-probs
torch.manual_seed(0)
vocab, T = 50, 6
tokens = torch.randint(0, vocab, (1, T))
logits = torch.randn(1, T, vocab)               # stand-in for model(tokens) outputs
logp = F.log_softmax(logits[:, :-1], dim=-1)    # predict token t+1 from prefix up to t
seq_logp = logp.gather(-1, tokens[:, 1:, None]).sum()
ce = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), tokens[:, 1:].reshape(-1), reduction="sum")
print(seq_logp.item(), -ce.item())              # identical
