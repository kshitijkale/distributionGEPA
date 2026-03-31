# DCPO: Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards

**Section:** Calibration-Aware Training (additional)

**Paper:** [arXiv:2603.09117](https://arxiv.org/abs/2603.09117)

**Venue:** arXiv, March 2026

---

## Why This Paper Is Critical

One of the three most important additional papers. DCPO **critiques RLCR directly** and identifies a fundamental problem: coupling accuracy and calibration rewards causes gradient conflicts that collapse confidence estimates toward extreme values. This is a **cautionary finding** for our DAL design.

---

## Key Findings

1. **RLCR collapses confidence estimates:** When accuracy and calibration are coupled in the same reward signal, the model learns to push confidence toward extreme values (0 or 1) with poor granularity — defeating the purpose of calibration training.

2. **Gradient conflict between accuracy and calibration:** The gradients from "be more accurate" and "be better calibrated" can point in opposite directions, causing optimization instability.

3. **Decoupled optimization proposed:** Separate the accuracy objective from the calibration objective, optimizing them independently to avoid gradient conflicts.

---

## How We Use It in Distributional GEPA

1. **Cautionary finding for DAL design:** Even calibration-trained models may produce **distributional artifacts** — extreme confidence values, bimodal distributions — that our annotation layer must handle gracefully. The DAL should not naively trust absolute probability values.

2. **Supports relative/semantic over absolute/numerical:** DCPO's finding that calibration training can produce extreme values reinforces our design choice to use **relative comparisons** and **semantic descriptions** rather than raw probability values in the DAL.

3. **Pareto dimension caution:** If we add calibration quality as a Pareto dimension (Condition F), DCPO warns that optimizing for calibration might conflict with optimizing for accuracy. The `"hybrid"` frontier type may mitigate this by allowing trade-offs, but we should monitor for signs of the same conflict.

4. **Related work positioning:** DCPO provides important context for why calibration is hard, even with explicit optimization. If distributional GEPA improves calibration as a side effect (without explicit calibration optimization), that's especially noteworthy given DCPO's findings.

---

## Connection to Other Papers

- **RLCR (extra_1):** DCPO directly critiques RLCR's approach
- **Rewarding Doubt (extra_2):** Also subject to the gradient conflict critique
- **Language Models Know What They Know (7.5.1):** The metacognitive abilities DCPO tries to preserve
- **Beyond Next Token Probabilities (7.6.4):** DCPO's finding about extreme confidence aligns with the distribution shape insight — absolute values are misleading; distribution shape matters

---

## Key Design Implication

The DAL should be robust to models with distorted calibration. Use:
- **Competing token ratios** (relative), not absolute probabilities
- **Trajectory shapes** (qualitative), not entropy magnitudes
- **Semantic descriptions** ("torn between X and Y"), not probability values
