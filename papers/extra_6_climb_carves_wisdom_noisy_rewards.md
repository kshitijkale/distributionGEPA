# The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason

**Section:** Reasoning and RL Context (additional)

**Paper:** [arXiv:2505.22653](https://arxiv.org/abs/2505.22653)

**Venue:** arXiv, May 2025

---

## Why This Paper Is Relevant

Demonstrates that LLMs are **surprisingly robust to noisy rewards** and that rewarding reasoning patterns (not just correctness) achieves comparable performance. This supports the thesis that **richer feedback signals (distributional diagnostics) could be more valuable than precise scalar rewards**.

---

## Key Findings

1. **Robustness to noisy rewards:** LLMs trained with RL maintain performance even when reward signals are noisy — suggesting the learning process is driven more by reasoning patterns than precise reward values.

2. **Rewarding reasoning patterns works:** Training that rewards good reasoning structure (not just final answer correctness) achieves comparable performance. The process matters, not just the outcome.

3. **Implications for feedback design:** If models learn from the pattern of feedback rather than precise reward magnitudes, richer qualitative feedback (like distributional diagnostics) may be more valuable than more precise numerical rewards.

---

## How We Use It in Distributional GEPA

1. **Supports the GEPA thesis:** GEPA's core claim is that rich natural language feedback outperforms scalar rewards. This paper provides evidence from the RL side: even in RL, the richness of the learning signal matters more than its precision.

2. **Justifies qualitative over quantitative DAL annotations:** If precise numbers don't help RL, they probably don't help the reflector either. Qualitative descriptions ("the model was uncertain at step 3") may be more valuable than precise entropy values ("H=2.31 bits") — supporting H3.

3. **Framing for distributional feedback:** Our distributional annotations are a form of "richer feedback about reasoning patterns" — they describe how the model reasoned (entropy trajectories, forking tokens) rather than just whether it got the right answer.

---

## Connection to Other Papers

- **Entropy Minimization (7.2.2):** Both show that distributional/pattern-level signals carry optimization-relevant information
- **RLVR Implicit Reasoning (extra_5):** Both investigate what RL actually learns from reasoning tasks
- **RLCR (extra_1):** RLCR adds calibration reward; this paper suggests the precise reward formulation matters less than the feedback richness
