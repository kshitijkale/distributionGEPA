# CCGSPG: Calibration-aware GRPO

**Section:** Calibration and Confidence (additional)

**Authors:** Liu et al. (2025)

**Source:** Cited in DCPO paper

**Venue:** 2025

---

## Why This Paper Is Relevant

Another calibration-aware RL variant that modifies the GRPO objective according to token-based confidence. Important for **related work completeness** when discussing how calibration interacts with RL-based optimization.

---

## Key Concept

Modifies GRPO's objective to incorporate token-based confidence, creating a calibration-aware variant of the RL baseline that GEPA competes against.

---

## How We Use It in Distributional GEPA

1. **Related work completeness:** When discussing GRPO baselines and calibration, cite CCGSPG alongside RLCR and DCPO to show the landscape of calibration-aware RL approaches.

2. **Broader context:** CCGSPG, RLCR, Rewarding Doubt, and DCPO collectively show that calibration is an active concern in RL-based LLM training — validating our decision to measure it as a secondary metric.

3. **Differentiation:** All these papers modify model weights for calibration. Our approach explores whether calibration can be influenced through prompt-level changes alone.

---

## Connection to Other Papers

- **RLCR (extra_1):** Same goal, different approach
- **DCPO (extra_3):** DCPO cites CCGSPG and identifies similar calibration collapse risks
- **Rewarding Doubt (extra_2):** Another calibration-aware RL variant
