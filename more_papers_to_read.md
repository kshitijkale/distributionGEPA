Based on everything that surfaced during the RLCR deep-dive, here are the papers you should add to your reading list, beyond what's already in the seed document:

**Calibration-Aware Training (new subsection for the seed doc)**

**RLCR: Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty** — Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas (MIT). arXiv:2507.16806, July 2025. Trains reasoning models with Brier score calibration reward alongside correctness. Proves that bounded proper scoring rules yield accurate and well-calibrated models. Reduces calibration error by up to 90%. Directly validates your calibration measurement axis and establishes that standard RL degrades calibration — a problem your distributional prompt optimization may address without retraining.

**Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models** — Bani-Harouni et al. arXiv:2503.02623, March 2025 (updated February 2026). Similar to RLCR but uses logarithmic scoring rule instead of Brier score. Models confidence as a betting game. Demonstrates generalization to unseen tasks without re-training, suggesting emergence of general confidence awareness. Achieves ECE of 0.0226 on TriviaQA.

**DCPO: Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards** — arXiv:2603.09117, March 2026. Critiques RLCR directly — finds RLCR collapses confidence estimates toward extreme values with poor granularity. Identifies gradient conflict between accuracy and calibration when coupled in the same reward. Proposes decoupled optimization. Important cautionary finding: even calibration-trained models may produce distributional artifacts your annotation layer must handle.

**Reasoning and RL Context**

**Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning** — (from the Awesome-RL-for-LRMs list). Directly connects to the forking tokens finding already in your seed doc. Validates that entropy-identified minority tokens are disproportionately important for RL training, strengthening your case that entropy-guided feedback should also improve prompt optimization.

**RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs** — Wen et al. arXiv:2506.14245, June 2025. Introduces CoT-Pass@K metric that accounts for both final answer and intermediate reasoning steps. Relevant because it provides methodology for evaluating reasoning chain quality, not just final answer accuracy — one of your secondary metrics.

**The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason** — arXiv:2505.22653, May 2025. Shows LLMs are surprisingly robust to noisy rewards, and that rewarding reasoning patterns (not just correctness) achieves comparable performance. Supports your thesis that richer feedback signals (distributional diagnostics) could be more valuable than precise scalar rewards.

**Calibration and Confidence (additional)**

**CCGSPG: Calibration-aware GRPO** — Liu et al., 2025 (cited in DCPO paper). Modifies GRPO objective according to token-based confidence. Another calibration-aware RL variant to be aware of for related work completeness.

**Survey and Overview**

**The State of Reinforcement Learning for LLM Reasoning** — Sebastian Raschka, April 2025. Comprehensive overview article covering RLHF, GRPO, RLVR, and recent reasoning RL developments. Good for positioning your work relative to the RL landscape that GEPA competes with.

**A Survey of Reinforcement Learning for Large Reasoning Models** — arXiv:2509.08827, October 2025. Academic survey covering RL for LRMs post-DeepSeek-R1. Useful for understanding the broader context and citing appropriately.

That gives you nine new papers. The three most important to read carefully are RLCR, DCPO, and the "High-Entropy Minority Tokens" paper — they each directly touch a core mechanism of your proposed system. The rest are important for positioning and related work completeness but won't change your architecture.