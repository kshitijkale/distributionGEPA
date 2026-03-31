# RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs

**Section:** Reasoning and RL Context (additional)

**Authors:** Wen et al.

**Paper:** [arXiv:2506.14245](https://arxiv.org/abs/2506.14245)

**Venue:** arXiv, June 2025

---

## Why This Paper Is Relevant

Introduces the **CoT-Pass@K metric** that accounts for both final answer correctness AND intermediate reasoning step quality. Provides methodology for evaluating reasoning chain quality — relevant to our secondary metrics.

---

## Key Findings

1. **RLVR implicitly improves reasoning quality:** Even though RLVR only rewards final answer correctness, the intermediate reasoning steps also improve — the model learns better reasoning patterns, not just better answer guessing.

2. **CoT-Pass@K metric:** A new metric that evaluates reasoning chains holistically, considering both the final answer and the quality of intermediate steps. Pass@K is computed considering whether the chain-of-thought is correct, not just the final answer.

---

## How We Use It in Distributional GEPA

1. **Secondary metric methodology:** CoT-Pass@K or a similar metric could supplement our primary accuracy metric (Section 5.1) by measuring whether distributional feedback produces prompts that improve reasoning quality, not just final answer accuracy.

2. **Evaluation of DAL impact on reasoning:** If distributional GEPA produces prompts where the reasoning chains are better (not just the final answers), CoT-Pass@K would capture this — strengthening the paper's contribution.

3. **Reasoning chain evaluation:** The methodology for evaluating intermediate reasoning steps is useful for our Phase 4 analysis (emergent mutation strategies, case studies).

---

## Connection to Other Papers

- **Entropy Trajectory Shape (7.2.1):** Monotone entropy trajectories correlate with correct reasoning; CoT-Pass@K evaluates reasoning quality directly
- **High-Entropy Minority Tokens (extra_4):** Both papers demonstrate that reasoning process quality matters, not just final answers
- **The Climb Carves Wisdom (extra_6):** Both address what RL actually learns in reasoning tasks
