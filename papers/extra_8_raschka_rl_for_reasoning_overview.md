# The State of Reinforcement Learning for LLM Reasoning

**Section:** Survey and Overview (additional)

**Authors:** Sebastian Raschka

**Venue:** Overview article, April 2025

---

## Why This Paper Is Relevant

Comprehensive overview covering RLHF, GRPO, RLVR, and recent reasoning RL developments. Useful for **positioning our work relative to the RL landscape** that GEPA competes with.

---

## Key Content

Covers the full landscape of RL methods for LLM reasoning:
- **RLHF:** Foundational approach, human preference-based
- **GRPO:** Group Relative Policy Optimization — GEPA's primary RL comparison
- **RLVR:** Reinforcement Learning from Verifiable Rewards — reward from automated verification
- **Recent developments:** Post-DeepSeek-R1 reasoning RL approaches

---

## How We Use It in Distributional GEPA

1. **Positioning:** When framing GEPA as an alternative to RL, this overview provides the landscape context for what RL methods exist and their trade-offs.

2. **Background section writing:** Useful reference for the paper's background section on RL-based optimization.

3. **GRPO context:** Understanding GRPO's strengths and weaknesses helps frame why GEPA's prompt-only approach with distributional feedback is a compelling alternative.

---

## Connection to Other Papers

- **RLCR (extra_1), DCPO (extra_3):** Specific RL advances covered in this overview
- **GEPA paper (not in scope):** GEPA's comparison to GRPO
- **Academic survey (extra_9):** More formal academic survey of the same landscape
