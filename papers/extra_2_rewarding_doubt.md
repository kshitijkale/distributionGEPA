# Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models

**Section:** Calibration-Aware Training (additional)

**Authors:** Bani-Harouni et al.

**Paper:** [arXiv:2503.02623](https://arxiv.org/abs/2503.02623)

**Venue:** arXiv, March 2025 (updated February 2026)

---

## Why This Paper Is Relevant

Similar to RLCR but uses a **logarithmic scoring rule** instead of Brier score. Demonstrates generalization to unseen tasks without re-training — suggesting emergence of **general confidence awareness**. Achieves ECE of 0.0226 on TriviaQA.

---

## Key Findings

1. **Logarithmic scoring rule for calibration:** Models confidence expression as a "betting game" where the model is rewarded for honest probability estimates.

2. **Generalization without re-training:** Once trained with the confidence reward, models maintain calibration on unseen tasks — suggesting a generalizable metacognitive skill.

3. **ECE of 0.0226 on TriviaQA:** Very well-calibrated confidence expression.

---

## How We Use It in Distributional GEPA

1. **Calibration measurement context:** When measuring ECE across our conditions, this paper provides a reference point for what "well-calibrated" looks like.

2. **Generalization finding is relevant:** If calibration generalizes without re-training in the RL setting, might distributional feedback similarly produce prompts with generalizable uncertainty-awareness?

3. **Related work completeness:** Cite alongside RLCR for the calibration-aware training landscape.

---

## Connection to Other Papers

- **RLCR (extra_1):** Same goal, different scoring rule (Brier vs. logarithmic)
- **DCPO (extra_3):** Critiques both RLCR and related calibration-training approaches
- **Closing the Confidence-Faithfulness Gap (7.5.4):** Addresses the same confidence-correctness alignment problem
