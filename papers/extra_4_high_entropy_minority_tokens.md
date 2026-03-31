# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

**Section:** Reasoning and RL Context (additional)

**Source:** Awesome-RL-for-LRMs list

---

## Why This Paper Is Critical

One of the three most important additional papers. Directly connects to the forking tokens finding (7.2.3) and **strengthens the case that entropy-guided feedback should improve prompt optimization** — the core thesis of Distributional GEPA.

---

## Key Findings

1. **High-entropy minority tokens are disproportionately important for RL training:** Not all tokens contribute equally to RL-based reasoning improvement. The ~20% of tokens with high entropy drive most of the learning signal.

2. **Validates the 80/20 pattern:** Consistent with the forking tokens paper (7.2.3) finding that >50% of CoT tokens have near-zero entropy and ~20% are high-entropy forking tokens.

3. **Entropy-identified tokens improve RL training efficiency:** Focusing RL training on these tokens yields better results than training on all tokens uniformly.

---

## How We Use It in Distributional GEPA

1. **Strengthens core thesis:** If entropy-identified high-entropy tokens are the most important for RL weight optimization, they should also be the most important for prompt optimization. This directly supports our DAL's design of focusing annotations on forking tokens.

2. **Parallels between RL and prompt optimization:** The insight transfers: just as RL benefits from focusing on high-entropy tokens, GEPA's reflector should benefit from seeing which tokens were high-entropy and what the model was considering at those positions.

3. **Citation for the DAL's selective focus:** When justifying why the DAL annotates only forking tokens (not all tokens), this paper provides additional evidence that selective focus on high-entropy positions is more effective than uniform attention.

---

## Connection to Other Papers

- **Forking Tokens (7.2.3):** Directly validates and extends the forking token finding
- **Entropy Minimization (7.2.2):** Both papers show entropy contains optimization-relevant signal
- **Entropy Trajectory Shape (7.2.1):** High-entropy tokens are the positions that make trajectories non-monotone
- **RLCR (extra_1):** Both address RL training for reasoning; this paper focuses on which tokens matter

---

## Key Implication

The convergence of evidence — forking tokens paper, this paper, and the entropy trajectory shape paper — establishes a strong empirical basis for entropy-guided annotation. The DAL's focus on high-entropy positions is not arbitrary; it targets the positions that drive learning in both RL and (by extension) prompt optimization.
