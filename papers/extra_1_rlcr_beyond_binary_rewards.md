# RLCR: Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty

**Section:** Calibration-Aware Training (additional)

**Authors:** Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas (MIT)

**Paper:** [arXiv:2507.16806](https://arxiv.org/abs/2507.16806)

**Venue:** arXiv, July 2025

---

## Why This Paper Is Critical

One of the three most important additional papers. RLCR directly validates the calibration measurement axis of the Distributional GEPA project and establishes a key finding: **standard RL degrades calibration** — a problem that distributional prompt optimization may address without retraining.

---

## Key Findings

1. **Brier score calibration reward alongside correctness:** RLCR trains reasoning models using a Brier score calibration reward combined with a correctness reward, encouraging models to be both accurate and well-calibrated.

2. **Bounded proper scoring rules yield accurate and well-calibrated models:** The theoretical insight is that Brier score (a proper scoring rule) incentivizes truthful confidence expression.

3. **Reduces calibration error by up to 90%:** Dramatic improvement in calibration compared to standard RL training.

4. **Standard RL degrades calibration:** A critical finding — regular GRPO/RLVR training improves accuracy but worsens calibration. Models become more confidently wrong.

---

## How We Use It in Distributional GEPA

1. **Validates calibration measurement:** RLCR confirms that calibration is a meaningful axis to measure across our experimental conditions (Section 5.2 of PROJECT_SEED). Even though we don't claim calibration improvement, RLCR provides the theoretical foundation for why calibration matters.

2. **RL's calibration degradation is an opportunity:** If standard RL degrades calibration, and GEPA's prompt-only optimization doesn't, that's a significant practical advantage. The DAL's distributional annotations could help the reflector produce prompts that maintain calibration — a side-benefit we can measure.

3. **Brier score as a metric:** RLCR validates Brier score as a calibration metric, supporting its inclusion in our secondary metrics (Section 5.2).

4. **Context for GRPO comparison:** When we compare distributional GEPA to GRPO baselines, RLCR provides context for why GRPO might produce well-scoring but poorly-calibrated prompts.

---

## Connection to Other Papers

- **DCPO (extra_3):** Critiques RLCR directly — finds calibration collapse issues
- **Rewarding Doubt (extra_2):** Similar approach with logarithmic scoring rule
- **Language Models Know What They Know (7.5.1):** Foundational metacognition work that RLCR extends
- **Just Ask for Calibration (7.5.2):** Text-based calibration; RLCR uses reward-based calibration

---

## Key Implication

If distributional GEPA happens to produce better-calibrated prompts than vanilla GEPA (because the reflector sees confidence information and naturally produces prompts that improve confidence-correctness alignment), this is a publishable finding that connects to RLCR's work — achieving calibration benefits through prompt optimization rather than weight updates.
