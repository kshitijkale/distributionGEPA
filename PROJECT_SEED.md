# Distributional Feedback for Reflective Prompt Evolution

## Project Seed Document

**Working Title:** *What Does the Reflector Need to See? Distributional Diagnostics as Feedback for Prompt Evolution*

**Target Venue:** EMNLP 2026 (Main Conference)

**Core Research Question:** Can structured natural-language summaries of an LLM's output distribution — entropy trajectories, forking-token identities, answer confidence — provide diagnostic information that text-only execution traces miss, and if so, does GEPA's reflective mutation produce better prompts when given this information?

---

## 1. The Central Idea

### 1.1 Motivation

GEPA (Genetic-Pareto, Agrawal et al., 2025) is the current state-of-the-art prompt optimizer. It works by sampling execution traces from an LLM system, reflecting on those traces in natural language to diagnose problems, and proposing targeted prompt mutations — all while maintaining a Pareto frontier of diverse candidate prompts. GEPA outperforms RL-based methods (GRPO) by 6pp on average while using 35x fewer rollouts, and outperforms the previous best prompt optimizer (MIPROv2) by 10+pp.

However, GEPA's feedback loop is entirely text-based. The reflection LLM sees what the model *said* — the chain-of-thought tokens, tool calls, and final answers — but has no access to what the model *felt* while saying it. The output distribution at each token position encodes rich information about the model's confidence, uncertainty, and internal deliberation that is invisible in the sampled text. A model that confidently produces a wrong answer (low entropy, peaked distribution) has a fundamentally different failure mode from a model that hesitantly guesses correctly (high entropy, diffuse distribution), yet both look the same to a text-only reflector.

### 1.2 The Proposed Approach

We propose a **Distributional Annotation Layer** (DAL) that sits between GEPA's execution engine and its reflection module. After each candidate prompt runs on a minibatch, this layer computes lightweight distributional diagnostics from token-level logprobs and converts them into structured natural language annotations that are appended to the execution trace before reflection. Crucially, we do **not** impose a fixed taxonomy of failure modes or prescribe mutation strategies — we enrich the trace and let the reflector LLM determine what the distributional information means in context and how to act on it.

The key design principle is: **translate distributional signals into the natural language medium that GEPA already handles well, rather than creating a parallel numerical optimization pathway.** This respects GEPA's core thesis that language is a richer learning medium than scalar rewards, while addressing its blind spot of not having access to sub-textual confidence information.

### 1.3 Falsifiable Hypotheses

This project tests four specific hypotheses. If H1 fails, the project has a clear negative result. If H1 holds but H2 fails, the contribution shifts to a representation-design study. H3 and H4 are secondary but independently publishable.

**H1 (Information Gap):** There exist prompt-induced failure modes that are diagnosable from distributional signals but not from text traces alone. Specifically: on at least 20% of incorrectly-answered examples across HotpotQA and HoVer, distributional diagnostics (entropy trajectory, forking tokens, answer confidence) reveal information about *where* and *why* reasoning failed that is absent from the sampled chain-of-thought text.

**H2 (Actionable Feedback):** When the reflector LLM receives distributional annotations alongside text traces, it proposes prompt mutations that yield higher accuracy than mutations proposed from text traces alone — specifically, distributional GEPA outperforms both vanilla GEPA and text-enriched GEPA (see Section 4.1) by at least 2pp average accuracy across the task suite at equal rollout budget.

**H3 (Representation Matters):** The format in which distributional information is presented to the reflector significantly affects its utility. Qualitative descriptions ("the model was very uncertain here, torn between 'second' and 'third'") and semantic annotations (competing token identities) outperform raw numerical reporting ("H=2.31 bits") — because LLMs reason poorly about numbers but well about semantic relationships.

**H4 (Emergent Differentiation):** The reflector spontaneously produces qualitatively different mutation strategies for different distributional signatures (e.g., surgical edits for isolated high-entropy decision points vs. structural rewrites for non-monotone entropy trajectories), without being instructed to do so.

### 1.4 The Contribution

The naive framing — "add more information to the reflection prompt" — invites the critique that any improvement is expected and unsurprising. The actual contributions are:

1. **A systematic study of what feedback information the reflector can and cannot use.** We compare vanilla traces, text-enriched traces (verbalized confidence, self-consistency as text), and distributional annotations across multiple representation formats. This ablation reveals what LLMs can reason about when given self-referential information — a finding about LLM metacognition that is independently valuable regardless of accuracy gains.

2. **Representation design for distributional-to-textual translation.** Raw logprobs are massive, noisy, and mostly irrelevant (>50% of tokens have near-zero entropy). Identifying which distributional features carry semantic diagnostic value, at what granularity to report them, and in what format the reflector LLM can effectively consume them is a non-trivial design problem with transferable lessons.

3. **Empirical evidence for or against the distributional information gap.** Does the output distribution carry actionable diagnostic information beyond what text traces express? If yes, we quantify the gap. If no, that is an equally valuable finding — it would validate GEPA's text-only design and suggest that distributional methods (entropy minimization, uncertainty-guided decoding) operate through different mechanisms than what a reflector can exploit.

### 1.5 What This Is NOT

- This is NOT a new prompt optimizer. GEPA is the optimizer; we enrich its feedback signal.
- This is NOT a classification system. We do not bucket failures into types. We provide richer traces and observe what emerges.
- This is NOT an entropy-based decoding method. We do not modify the generation process. We use distributional information post-hoc for optimization feedback.
- This is NOT limited to entropy. We explore multiple distributional features: entropy, competing token identities, trajectory shape, anchor token confidence, and distributional shape beyond point estimates.

---

## 2. Critical Assumptions and Honest Assessment

This section exists because intellectual honesty about assumptions and risks upfront is both good science and good strategy for EMNLP review. Each assumption identifies a concrete falsification condition and what we do if it fails.

### 2.1 The Information Gap Assumption

**Assumption:** The output distribution at each token position encodes diagnostic information about prompt-induced failures that is not recoverable from the sampled text alone.

**Why it might be wrong:** A model that hedges in its chain-of-thought ("I think the answer might be Paris, but it could also be London") is already expressing uncertainty in text. Sophisticated reflectors might extract uncertainty signals from hedging language, error patterns, and reasoning structure without needing logprobs. The information gap might be smaller than we expect — or zero.

**How we test it:** Phase 1 Experiment 1.2 directly measures this by having annotators compare text-only vs. distributional diagnostics on 50 failure cases. The text-enriched GEPA baseline (Section 4.1) controls for the possibility that *any* additional information helps, not specifically distributional information.

**If it fails:** The project pivots to a negative-result paper: "Text Traces Are Sufficient — Why Distributional Signals Don't Help Reflective Prompt Optimization." This is publishable at EMNLP if the methodology is rigorous.

### 2.2 The Numerical Reasoning Assumption

**Assumption:** The reflector LLM can effectively reason about distributional information when presented in its feedback.

**Why it might be wrong:** LLMs are notoriously poor at numerical reasoning. Reporting "H=2.31 bits" or "P(Paris)=0.41, P(London)=0.33" may not help a reflector that cannot meaningfully compare or interpret these numbers. The reflector might ignore numerical information entirely, or worse, be confused by it.

**How we address it:** H3 directly tests this by comparing numerical vs. qualitative vs. semantic representations. We design the DAL to produce primarily qualitative and semantic annotations ("the model was torn between 'second' and 'third' document references") rather than raw numbers. Numbers are included only when they convey relative magnitude ("barely above chance" vs. "highly confident"). The format ablation (Section 4.4) is one of our most important experiments.

**Implication for design:** The DAL should be built to produce *semantic* annotations from the start. Competing token *identities* ("Paris" vs. "London") are more useful than competing token *probabilities* (0.41 vs. 0.33). Trajectory *shape* ("confidence dropped at step 3 then recovered") is more useful than trajectory *values* ("entropy went from 0.4 to 2.3 to 0.8").

### 2.3 The Model Access Problem

**Assumption:** We can obtain the logprobs needed to compute distributional diagnostics.

**Reality:** This fundamentally constrains which models we can optimize prompts for.
- **Open-weight models via vLLM:** Full vocabulary logprobs available. This is our primary experimental setting. Requesting full vocabulary (~152K tokens for Qwen 2.5) incurs ~10x overhead vs. top-10; we only need top-5 + entropy estimate.
- **OpenAI API:** Top-5 logprobs only. Sufficient for competing token identities and crude entropy estimates, but trajectory shape analysis is degraded.
- **Anthropic Claude:** Zero logprobs. The DAL cannot function at all. This is the most popular frontier model.
- **Google Gemini:** Logprobs via Vertex AI. Workable.
- **Other providers (Mistral, Together, Fireworks, xAI):** Top 8-20 logprobs. Sufficient.

**Why this matters:** GEPA's greatest practical advantage is model-agnostic universality. The `GEPAAdapter` protocol requires only text-in/text-out via `evaluate()` and `make_reflective_dataset()`. Requiring logprobs breaks this universality. The distributional variant cannot optimize prompts for Claude — a severe practical limitation.

**How we address it:** We are honest about this constraint. Primary experiments use open-weight models where we have full access. We report API model results where feasible (GPT-4o with top-5 logprobs). We explicitly do not claim this approach works for all models. We investigate whether the self-consistency feature (no logprobs needed, just multiple samples) can partially substitute for logprob-based features on API-only models — this is a key ablation. Note: the optimizer/reflector model can still be any strong model (GPT-4o, Claude); only the *student* model being optimized needs logprob access.

### 2.4 The Cost-Adjusted Comparison Problem

**Assumption (implicit in original design):** Comparing accuracy at equal *rollout count* is fair.

**Why it's not:** Extracting logprobs adds overhead. Requesting top-10 logprobs from vLLM adds negligible cost. But the diagnostic report adds ~200-500 tokens to each reflection prompt, increasing reflector LLM cost. Self-consistency (if used) costs 2-3x per example. The fair comparison is accuracy-vs-total-compute (FLOPs or API dollars), not accuracy-vs-rollouts.

**How we address it:** We report both accuracy-vs-rollouts AND accuracy-vs-total-compute curves. We track total LLM tokens consumed (student + reflector) across all conditions. If distributional GEPA is better at equal rollouts but worse at equal compute, that is an important finding that we report honestly. See Section 5.4.

### 2.5 The Calibration Hypothesis

**Original claim:** "Distributional feedback should improve calibration quality, generalization robustness, and sample efficiency — dimensions that current prompt optimizers do not measure or optimize for."

**Problem with this claim:** There is no mechanism by which seeing entropy values in the feedback would cause the reflector to produce prompts whose outputs are better-calibrated. Calibration improvement requires either (a) an explicit calibration loss in the optimization objective, or (b) a causal pathway from "reflector sees uncertainty" to "reflector proposes prompt changes that happen to improve calibration." Pathway (b) is plausible but speculative — the reflector might propose prompts that make the model more confident (lower entropy) regardless of correctness, which would *worsen* calibration.

**Revised position:** We *measure* calibration (ECE, Brier score) across all conditions as a secondary metric, but we do not *claim* it will improve. If it does, we report it as an empirical finding and analyze why. If it doesn't, we do not treat it as a failure. Calibration improvement becomes a genuine finding if it emerges, not a foregone conclusion.

**Optional extension:** If we want to *deliberately* optimize for calibration, we can add calibration quality as a Pareto dimension using GEPA's existing `objective_scores` in `EvaluationBatch` and the `"objective"` or `"hybrid"` frontier type. This would be a separate ablation condition (Section 4.3 condition E).

---

## 3. Technical Architecture

### 3.1 The Distributional Annotation Layer

This is the core new component. It sits between GEPA's execution engine and its reflection module.

```
┌─────────────────────────────────────────────────────────┐
│                     GEPA Loop                           │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Candidate │───>│   Execute    │───>│  Execution   │  │
│  │  Prompt   │    │  on Batch    │    │   Traces     │  │
│  └──────────┘    └──────────────┘    └──────┬───────┘  │
│                                             │          │
│                                    ┌────────▼────────┐ │
│                                    │  DISTRIBUTIONAL  │ │
│                                    │  ANNOTATION      │ │
│                                    │  LAYER (NEW)     │ │
│                                    └────────┬────────┘ │
│                                             │          │
│                                    ┌────────▼────────┐ │
│  ┌──────────┐    ┌──────────────┐  │   Annotated    │ │
│  │  Child   │<───│   Reflect    │<──│    Traces     │ │
│  │  Prompt  │    │  & Mutate    │   └───────────────┘ │
│  └────┬─────┘    └──────────────┘                     │
│       │                                                │
│       ▼                                                │
│  [Evaluate → Update Pareto Frontier]                   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Distributional Annotation Layer — Internal Design

**Input:** Execution trace (reasoning tokens, tool calls, outputs) + raw logprobs from the execution

**Processing Pipeline:**

1. **Entropy Computation:** For each token position, compute Shannon entropy from available logprobs:
   - Full vocabulary entropy (open-weight models): $H_t = -\sum_{v} p(v | \text{ctx}) \log p(v | \text{ctx})$
   - Top-k approximation (API models): $\hat{H}_t = -\sum_{i=1}^{k} p(v_i | \text{ctx}) \log p(v_i | \text{ctx})$ with residual mass correction

2. **Forking Token Identification:** Flag tokens where entropy exceeds a threshold (calibrate per-model, but ~1.5 bits is a reasonable starting point based on Wang et al.'s finding that ~20% of tokens are high-entropy). For each forking token, record:
   - The entropy value
   - The identities and probabilities of the top-3 competing tokens
   - The position within the reasoning chain (which step)
   - The local context (preceding and following tokens)

3. **Entropy Trajectory Computation:** Segment the reasoning chain into logical steps (using newlines, step markers, or sentence boundaries). For each step, compute aggregate entropy (mean or max of token-level entropy within the step). Classify the trajectory:
   - Is it monotonically decreasing? (strongest predictor of correct reasoning per the trajectory shape paper)
   - Where are the entropy spikes? (which steps show uncertainty)
   - What is the overall trajectory shape? (decreasing, increasing, U-shaped, oscillating)

4. **Anchor Token Analysis:** For the final answer or key decision tokens:
   - Extract probability of the selected answer token
   - Compute normalized confidence: $P(\text{selected}) / (P(\text{selected}) + P(\text{runner-up}))$
   - Flag if confidence is below threshold (e.g., <0.6)

5. **Self-Consistency Check (optional, budget-dependent):** If rollout budget allows, sample 2-3 additional responses with the same prompt and input:
   - Compute agreement rate on the final answer
   - Note where reasoning paths diverge (which step)
   - This is expensive (2-3x more LLM calls per example) so may be reserved for high-value diagnostics or used only in the text-enriched baseline (Section 4.1) where it doesn't require logprobs

6. **Natural Language Report Generation:** Convert all of the above into a structured, primarily semantic annotation. This is the critical design step — per Section 2.2, we prioritize semantic and qualitative descriptions over raw numbers.

**Preferred format (semantic-first):**
```
=== DISTRIBUTIONAL DIAGNOSTIC ===
Reasoning trajectory: NON-MONOTONE — the model's confidence dropped at step 3
(identifying the relevant document) and then partially recovered.

Decision points where the model was uncertain:
- Step 3: The model was nearly equally split between referencing the "second",
  "third", or "first" document. This suggests the prompt's retrieval instructions
  are ambiguous about document ordering.
- Step 5 (final answer): The model was torn between "Paris" and "London",
  with only a slight lean toward "Paris". The reasoning in steps 3-4 did not
  decisively resolve which city was correct.

Answer confidence: LOW — the model's selected answer "Paris" was barely
preferred over the runner-up "London".

Self-consistency (3 samples): 2/3 agreed on "Paris". The dissenting path
diverged at step 3, choosing a different source document.
=== END DIAGNOSTIC ===
```

**Comparison format (numerical, for ablation):**
```
=== DISTRIBUTIONAL DIAGNOSTIC ===
Overall: avg entropy 0.82 bits. Trajectory: NON-MONOTONE.

High-entropy positions:
- Step 3: H=2.31 bits. Top tokens: "second" (0.28), "third" (0.25), "first" (0.22).
- Step 5: H=1.44 bits. Top tokens: "Paris" (0.41), "London" (0.33).

Answer confidence: P(Paris)=0.41, normalized=0.55.
Self-consistency: 2/3. Divergence: step 3.
=== END DIAGNOSTIC ===
```

### 3.3 Integration with GEPA's Codebase

The DAL integrates with GEPA at specific, well-defined points in the existing architecture. No changes to core protocols are required.

**1. Primary integration: `GEPAAdapter.make_reflective_dataset()`** (`src/gepa/core/adapter.py:146`)

This method takes execution trajectories and returns `Mapping[str, Sequence[Mapping[str, Any]]]` — a dict of component name to list of records with keys like `"Inputs"`, `"Generated Outputs"`, `"Feedback"`. The DAL enriches the `"Feedback"` field (or adds a `"Distributional_Diagnostic"` key) in these records before they reach the reflector.

Concretely, this means we implement a wrapper adapter or a post-processing step in `make_reflective_dataset()`:

```python
# Conceptual integration — the DAL wraps the existing adapter
class DistributionalAdapter(GEPAAdapter[DataInst, Trajectory, RolloutOutput]):
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        # Get the base reflective dataset from the wrapped adapter
        base_dataset = self.inner_adapter.make_reflective_dataset(
            candidate, eval_batch, components_to_update
        )
        # Enrich each record with distributional annotations
        return self.dal.annotate(base_dataset, eval_batch.trajectories)
```

The `evaluate()` method also needs modification: when `capture_traces=True`, it must request logprobs from the student model and store them in the trajectory objects. The `Trajectory` type is opaque to GEPA (the engine never inspects it), so we can include logprobs without changing any core types.

**2. Instruction proposal template** (`src/gepa/strategies/instruction_proposal.py:13`)

The `InstructionProposalSignature` has a `<side_info>` placeholder where reflective dataset records are rendered as markdown (`# Example N` / `## Key` hierarchy). Distributional annotations flow through here automatically if they are part of the reflective dataset records — no template changes required beyond optionally adding a one-line instruction about distributional diagnostics.

The default meta-prompt already says:
> "Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information..."

The "feedback" here includes whatever our `make_reflective_dataset()` puts in the `"Feedback"` field. So the DAL annotations reach the reflector through the existing pipeline.

**3. Pareto frontier for calibration dimensions** (`src/gepa/core/state.py`)

If we add calibration quality or entropy monotonicity as Pareto objectives, they go into `objective_scores: list[dict[str, float]]` in `EvaluationBatch`. GEPA already supports four frontier types — `instance`, `objective`, `hybrid`, `cartesian` — and `objective_scores` are already tracked per-candidate for domination checks. No new Pareto machinery is needed; we use `frontier_type="objective"` or `"hybrid"` in the `optimize()` call.

**4. Evaluation cache** (`src/gepa/core/state.py`)

`EvaluationCache` uses SHA-256 hashing of candidate text for cache lookups. Distributional annotations are per-execution (tied to the specific logprob output), not per-candidate-text, so they do not interact with or invalidate the cache. However, the logprob-enriched `evaluate()` call (with `capture_traces=True`) is never cached anyway — GEPA always runs fresh evaluations when it needs traces for reflection (see `reflective_mutation.py:210`).

**5. Callbacks** (`src/gepa/core/callbacks.py`)

GEPA has 16+ typed callback events (`TypedDict` classes). Adding a `DistributionalAnnotationEvent` would let experiment tracking (MLflow/W&B) log annotation quality metrics without touching the core loop.

### 3.4 Distributional Pareto Dimensions (Optional Extension)

Beyond augmenting the reflection, we can optionally add distributional quality metrics as objectives in the Pareto frontier:

- **Entropy trajectory monotonicity rate:** Fraction of evaluation examples where the reasoning chain shows monotonically decreasing entropy. Higher = more reliable reasoning.
- **Calibration quality (ECE):** Expected calibration error of the prompt's outputs. Lower = better calibrated.

These create a richer Pareto frontier where a slightly less accurate but more reliable prompt can survive selection pressure. We test this as a separate ablation to isolate the effect of richer feedback (annotation layer) from richer selection (Pareto dimensions).

### 3.5 Implementation Constraints

**Logprob storage:**
- Full vocabulary logprobs: ~1.2 MB per token position — impractical to store at scale
- Top-20 logprobs: ~160 bytes per position — manageable
- Our diagnostic pipeline only needs top-5 logprobs + total entropy estimate per position
- Pre-compute diagnostics immediately after execution; store only the diagnostic reports, not raw logprobs

**Computational overhead:**
- Entropy computation from top-k logprobs: negligible
- Diagnostic report generation: ~200-500 additional tokens per example in the reflection prompt
- Self-consistency sampling (if used): 2-3x increase in evaluation cost per example
- Total overhead estimate: 10-30% increase in optimization cost (excluding self-consistency), potentially offset by faster convergence
- This overhead must be accounted for in cost-adjusted comparisons (Section 5.4)

---

## 4. Experimental Design

### 4.1 The Critical Baselines

The experimental design hinges on one question: **does the improvement come from distributional information specifically, or from any richer feedback?** Without controlling for this, the paper has a fatal reviewer objection.

**Condition A — Unoptimized Baseline:** The seed prompt, no optimization.

**Condition B — MIPROv2:** The standard comparison optimizer from the prompt optimization literature.

**Condition C — Vanilla GEPA:** GEPA with standard text-only traces. This is the primary baseline.

**Condition D — Text-Enriched GEPA (CRITICAL CONTROL):** GEPA with enriched text traces that do NOT use logprobs. The reflective dataset includes:
- Verbalized confidence: Ask the student model "How confident are you in this answer? Rate 1-10 and explain." after each response.
- Self-consistency results as text: Sample 2-3 additional responses and report agreement in natural language ("2 out of 3 runs agreed on 'Paris'; the third said 'London' and diverged at the document selection step").
- Explicit uncertainty hedges: Extract hedging language from the chain-of-thought ("I think...", "probably...", "not sure...") and highlight it.

This baseline is essential because it enriches the reflective dataset with uncertainty information without requiring logprobs. If Condition D matches or beats Condition E, then distributional signals per se don't help — text-based uncertainty expression is sufficient. If Condition E significantly outperforms D, we have evidence for the distributional information gap.

**Condition E — Distributional GEPA:** GEPA with the DAL providing distributional annotations in the semantic-first format (Section 3.2).

**Condition F — Distributional GEPA + Pareto Dimensions (optional):** Adds calibration and/or entropy monotonicity as Pareto objectives. Tests whether richer selection pressure (not just richer feedback) improves outcomes.

### 4.2 Phase 1: Validation (Go/No-Go)

Before building the full system, validate the core assumptions. Budget: 2 weeks. Clear go/no-go criteria.

**Experiment 1.1 — Replication:** On HotpotQA with Llama 3 70B via vLLM, run 200 examples with CoT prompting. Compute entropy trajectories. Test whether entropy trajectory monotonicity predicts correctness on this task with this model. If the published finding (68.8% vs. 46.8%) doesn't replicate with at least a 10pp gap, reconsider the approach.

**Experiment 1.2 — Information Gap Assessment:** For 50 incorrectly answered examples, have two annotators independently examine: (a) the text trace only, (b) the text trace + distributional diagnostic. For each, annotators answer: "Can you identify the specific reasoning step where this went wrong, and why?" Compare diagnostic quality. If distributional information reveals additional diagnostic value in fewer than 10/50 examples (20%), the information gap may be too small to justify the approach.

**Experiment 1.3 — Reflector Pilot:** Give the reflector LLM (GPT-4o or Claude) 20 example traces in three conditions: text-only, text + numerical diagnostics, text + semantic diagnostics. Compare proposed mutations on relevance (rated by annotators on 1-5 scale). This pilots H3 (qualitative > numerical) and checks whether the reflector uses distributional information at all.

**Go/No-Go Criteria:**
- Experiment 1.1 replicates with ≥10pp monotone vs. non-monotone gap → GO
- Experiment 1.2 shows distributional info adds value in ≥10/50 cases → GO
- Experiment 1.3 shows reflector uses distributional info (mean relevance score higher than text-only by ≥0.5 on 5-point scale) → GO
- If 1.1 fails: STOP. Entropy trajectory prediction doesn't hold for our setup.
- If 1.1 passes but 1.2 fails: PIVOT to studying what text-based enrichment can do (Condition D becomes primary).
- If 1.1 and 1.2 pass but 1.3 fails: PIVOT to representation design study (how to present distributional info to LLMs).

### 4.3 Phase 2: Core Experiments

Run all six conditions (A-F) on 4 tasks (see Section 6). Report all primary metrics from Section 5.

**Configuration:**
- Student model: Llama 3 70B via vLLM (primary), Qwen 2.5 72B (secondary, for generalization)
- Reflector/optimizer model: GPT-4o (to match GEPA's published setup)
- Rollout budget: 400, 800, 1200 (matching GEPA's published budget range)
- Minibatch size: 25 (GEPA default)
- Candidate selector: `ParetoCandidateSelector` (default)
- Component selector: `RoundRobinReflectionComponentSelector` (default)
- Frontier type: `instance` (default), with `hybrid` for Condition F

**For each condition, record:**
- All metrics from Section 5
- Full reflective dataset records and proposed mutations (for the analysis in Section 4.5)
- Total tokens consumed (student + reflector) for cost-adjusted curves

### 4.4 Phase 3: Ablation Studies

These ablations directly test our hypotheses.

**Ablation 1 — Representation Format (tests H3):**
Remove each diagnostic feature independently from the DAL and test combinations:
- Semantic-first format (full DAL as described) — the default
- Numerical-only format (entropy values, probabilities, no semantic interpretation)
- Competing token identities only (no entropy values, just "the model was torn between X and Y")
- Trajectory shape only (monotone/non-monotone classification, no per-token details)
- Anchor confidence only (answer-level confidence statement)

Key question: Do competing token identities (semantic information) matter more than entropy magnitudes (numerical information)?

**Ablation 2 — Granularity:**
- Per-token entropy reporting vs. per-step summaries vs. per-chain summary
- Key question: What level of detail helps the reflector without overwhelming it?

**Ablation 3 — Self-Consistency as Substitute:**
- Self-consistency (text-only, no logprobs) vs. logprob-based features vs. both
- Key question: Can self-consistency substitute for logprobs on API-only models?

**Ablation 4 — Budget Sensitivity:**
- Accuracy-vs-rollouts and accuracy-vs-compute curves at {50, 100, 200, 400, 800, 1200} rollouts
- Key question: Does distributional feedback help more at low budgets (each reflection is more informative) or high budgets (more iterations to exploit richer feedback)?

### 4.5 Phase 4: Analysis

**Emergent Mutation Strategy Analysis (tests H4):**
Collect all (distributional_profile, proposed_mutation) pairs across optimization runs. Cluster diagnostic profiles by signature type. For each cluster, analyze mutation patterns. Present findings as:
- If distinct strategies emerge: report taxonomy as empirical finding about LLM metacognition
- If continuous refinement instead: report the nature of improvement (more targeted edits, better-justified changes, etc.)
- Present 5-6 detailed case studies with full traces and reflections side-by-side

**Error Analysis — When Does It NOT Help:**
- Identify tasks/failure modes where distributional signals are noisy or misleading
- Expected failure cases: confidently wrong (low entropy, wrong answer — distributional info confirms confidence but doesn't help diagnose knowledge gaps), short reasoning chains, high aleatoric uncertainty tasks
- Report these honestly

**Reflector Behavior Comparison:**
- Side-by-side comparison of vanilla GEPA reflections vs. distributional GEPA reflections on the same failing examples
- Look for: misdiagnoses prevented, new diagnoses enabled, cases where distributional info confused the reflector

---

## 5. Metrics

### 5.1 Primary Metrics

**Task Accuracy:** Exact match / F1 (task-dependent) on held-out test sets across all conditions. Table stakes.

**Sample Efficiency:** Accuracy-vs-rollouts curves at {50, 100, 200, 400, 800, 1200} rollouts. GEPA's existing claim: 100-500 rollouts sufficient vs. 24K+ for RL. Does distributional feedback push this lower?

**Cost-Adjusted Efficiency (Section 5.4):** Accuracy-vs-total-tokens curves. See below.

### 5.2 Secondary Metrics

**Calibration Quality:** ECE, Brier score, reliability diagrams. Measured as observation, NOT claimed as expected improvement (per Section 2.5). Computed from: token-level logprob of selected answer, self-consistency frequency, anchor token probability.

**Generalization Gap:** |accuracy_optimization_set - accuracy_test_set|. Does distributional feedback favor robust prompts over lucky ones?

**Mutation Precision:** Edit distance between parent and child prompts. Ratio of accuracy gain to edit distance ("mutation efficiency"). Does distributional feedback produce more surgical edits?

**Convergence Stability:** Pareto frontier quality trajectory over iterations. Reduced oscillation, earlier plateau.

### 5.3 Core Analyses

These are the findings that make or break the paper at EMNLP. Not metrics to optimize but phenomena to discover and report.

1. **Emergent mutation strategies** (described in Section 4.5)
2. **Which distributional features matter** (from ablation results)
3. **Reflector behavior analysis** (side-by-side case studies)
4. **When distributional feedback does NOT help** (honest error analysis)

### 5.4 Cost-Adjusted Metrics

Because logprob extraction and longer reflection prompts add overhead, raw rollout counts are not a fair comparison. We track:

- **Total student tokens:** prompt + completion tokens across all student model calls, including self-consistency samples
- **Total reflector tokens:** prompt + completion tokens for all reflection/proposal calls (longer prompts with diagnostic annotations)
- **Total cost estimate:** (student tokens x student price) + (reflector tokens x reflector price)
- **Accuracy-vs-total-tokens curves:** the honest comparison. Plotted alongside accuracy-vs-rollouts.

If distributional GEPA reaches 80% accuracy at 400 rollouts but vanilla GEPA reaches 80% at 500 rollouts, yet the total token cost is 2x higher for distributional GEPA, the win is illusory. We report this transparently.

---

## 6. Task Selection

We select 4 tasks (not 6+) to maintain experimental rigor within the timeline. Each task is chosen for a specific reason related to our hypotheses.

### 6.1 Primary Tasks

**HotpotQA** — Multi-hop question answering. Core NLP task. Multi-step reasoning with meaningful intermediate steps for entropy trajectory analysis. GEPA baseline available. EMNLP-friendly. *Why this task:* Multi-hop structure creates natural entropy variation across steps; forking tokens at document selection points are semantically meaningful.

**HoVer** — Multi-hop fact verification. Classification task with clear right/wrong answers. Calibration measurement is natural (binary SUPPORTS/REFUTES). EMNLP-friendly. GEPA baseline available. *Why this task:* Binary classification makes calibration analysis straightforward; verification requires evidence selection steps where distributional signals should be informative.

**FEVER** — Fact extraction and verification. EMNLP-native task. Classification with clear calibration semantics. Well-established baselines. *Why this task:* Different from HoVer in requiring single-hop verification, testing whether distributional signals help even with simpler reasoning chains. Also adds an EMNLP-recognized benchmark.

**IFBench** — Instruction following benchmark. Tests multi-constraint instruction compliance. GEPA baseline available. *Why this task:* Failure modes are diverse (missed constraints, misinterpreted constraints, partial compliance) — distributional signals might differentiate these. Also connects to practical enterprise use cases.

### 6.2 Optional Extension Task

**AIME-2025** — Competition-level mathematics. GEPA baseline available. Long reasoning chains with rich entropy trajectories. If compute budget allows, this demonstrates generality beyond NLP tasks. Lower priority for EMNLP framing.

### 6.3 Selection Criteria

All chosen tasks satisfy:
1. Chain-of-thought reasoning is beneficial (meaningful intermediate steps)
2. Answer space is well-defined enough to measure calibration
3. Multiple distinct failure modes exist
4. GEPA baseline performance is available or reproducible
5. Multi-step reasoning creates meaningful entropy trajectories

---

## 7. Relevant Papers — Detailed Reading List

### 7.1 GEPA and Prompt Optimization Foundations

#### GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
- **Authors:** Lakshya A. Agrawal et al. (Databricks, UC Berkeley)
- **Venue:** ICLR 2026 (Oral)
- **Paper:** arXiv:2507.19457
- **Code:** https://github.com/gepa-ai/gepa
- **Why essential:** This is the system we are extending. Must understand every component: the Pareto frontier selection (four frontier types: instance, objective, hybrid, cartesian), the reflection prompt structure (`InstructionProposalSignature` with `<curr_param>` and `<side_info>` placeholders), the mutation operator (reflective mutation via `ReflectiveMutationProposer`), the merge operation (`MergeProposer` with common ancestor detection), the evaluation protocol (`EvaluationBatch` with scores, trajectories, objective_scores), and the benchmark tasks. The adapter protocol (`GEPAAdapter`) with its three methods (`evaluate`, `make_reflective_dataset`, optional `propose_new_texts`) defines where our DAL integrates.
- **Key numbers:** +6pp over GRPO on average, up to 19pp. 35x fewer rollouts. +12pp over MIPROv2 on AIME-2025. 400-1200 rollouts for major benchmarks. Minibatch acceptance criterion: `sum(new_scores) > sum(old_scores)`.

#### MIPROv2: Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs
- **Authors:** Krista Opsahl-Ong et al. (Stanford)
- **Venue:** NeurIPS 2024
- **Paper:** arXiv:2406.11695
- **Why relevant:** The primary comparison optimizer. Uses Bayesian optimization (Optuna TPE) to search over (instruction, few-shot demo) combinations. Generates all candidates upfront then searches, rather than iteratively evolving. Does NOT use execution traces or reflection. Understanding what MIPROv2 can and cannot do highlights what GEPA's reflection adds.
- **Framework:** DSPy (dspy.ai)

#### SIMBA: Stochastic Introspective Mini-Batch Ascent
- **Authors:** DSPy team (Stanford)
- **Why relevant:** Another DSPy optimizer that uses mini-batch sampling and introspective analysis. A weaker form of what we're proposing (reflection on failures, but without distributional information).

#### EvoPrompt: Language Models Are Human-Level Prompt Engineers — Connecting LLMs with Evolutionary Algorithms
- **Authors:** Guo et al.
- **Venue:** ICLR 2024
- **Paper:** arXiv:2309.08532
- **Why relevant:** Earlier evolutionary prompt optimizer using genetic algorithms. Key limitation: uses only task accuracy as fitness — no reflection, no trace analysis. Establishes that evolutionary approaches work for prompt optimization.

#### PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution
- **Authors:** Fernando et al. (DeepMind)
- **Venue:** ICML 2024
- **Paper:** arXiv:2309.16797
- **Why relevant:** Evolves both task prompts and mutation prompts. Self-referential improvement is conceptually related to using the model's own distributional information to guide its own optimization. But does not use logprobs.

#### APO/ProTeGi: Automatic Prompt Optimization with "Gradient Descent" and Beam Search
- **Authors:** Pryzant et al. (Microsoft)
- **Venue:** EMNLP 2023
- **Paper:** arXiv:2305.03495
- **Why relevant:** Published at our target venue. Uses natural language "gradients" to optimize prompts. Conceptually similar to GEPA's reflection but without the Pareto frontier. Demonstrates EMNLP accepts prompt optimization papers when framed around language understanding.

#### DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
- **Authors:** Khattab et al. (Stanford)
- **Venue:** ICLR 2024
- **Paper:** arXiv:2310.03714
- **Why relevant:** The framework within which GEPA can operate. Understanding DSPy's abstraction (signatures, modules, optimizers) is useful for implementation context.

#### Building State-of-the-Art Enterprise Agents 90x Cheaper with Automated Prompt Optimization
- **Authors:** Databricks
- **Source:** Databricks Blog, 2025
- **Why relevant:** Validates GEPA in enterprise settings. Shows GEPA-optimized open-source models outperforming Claude Opus 4.1 baseline. Reports ~3x runtime overhead vs. MIPROv2/SIMBA — distributional GEPA should aim not to worsen this.

### 7.2 Uncertainty in Chain-of-Thought Reasoning

#### Entropy Trajectory Shape Predicts LLM Reasoning Reliability
- **Authors:** (2025)
- **Paper:** arXiv:2603.18940
- **Why critical:** DIRECTLY supports our core mechanism. Key finding: whether per-step entropy monotonically decreases predicts accuracy (68.8% for monotone vs. 46.8% for non-monotone, p=0.0005). Crucially demonstrates a *shape-over-magnitude dissociation* — the pattern of entropy change matters, not the absolute values. Costs ~1/8 the compute of self-consistency.
- **How we use it:** Entropy trajectory monotonicity as a Pareto dimension; trajectory annotations in the diagnostic report; evidence that distributional signals carry diagnostic information beyond text traces. Also supports H3 — shape matters more than magnitude.

#### The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning
- **Authors:** (2025)
- **Paper:** arXiv:2505.15134
- **Why critical:** Shows that minimizing output entropy — without labeled data — elicits reasoning priors. EM-INF optimizes logits at inference time with no parameter updates. Demonstrates distributional signals contain sufficient information for optimization even without ground truth.
- **How we use it:** Theoretical motivation — if entropy minimization alone improves reasoning, entropy-informed prompt optimization should be more powerful because it can make structural prompt changes.

#### Token-Level Entropy Patterns in LLM Reasoning (Forking Tokens)
- **Authors:** Wang et al. (2025)
- **Source:** Emergent Mind topic synthesis + underlying papers
- **Why critical:** >50% of CoT tokens have near-zero entropy; ~20% are high-entropy "forking tokens" at logical branching points. Training RLVR only on forking tokens yields +7.7 on AIME'24 over full-token training.
- **How we use it:** Our annotation layer focuses on forking tokens, not deterministic tokens. Directly informs DAL design.

#### Entropy-Guided Loop: Achieving Reasoning through Uncertainty-Aware Generation
- **Authors:** (2025)
- **Paper:** arXiv:2509.00079
- **Why critical:** The closest existing analog to our DAL. Computes token-level entropy, combines three signals (perplexity, max-entropy, low-confidence token counts), and passes a compact structured uncertainty report back to the model for self-refinement. Directly demonstrates the bridge between distributional and text-based feedback.
- **How we use it:** Design template for our annotation format.

#### ERGO: Entropy-guided Resetting for Generation Optimization
- **Authors:** (2025)
- **Paper:** arXiv:2510.14077
- **Why relevant:** Monitors Shannon entropy during multi-turn conversations and triggers adaptive prompt consolidation when entropy spikes, yielding 56.6% average performance gain.
- **How we use it:** Validates entropy monitoring for prompt-level decisions at inference time; we extend to optimization time.

#### Reinforcement Inference: Leveraging Uncertainty for Self-Correcting Language Model Reasoning
- **Authors:** (2025)
- **Paper:** arXiv:2602.08520
- **Why relevant:** Uses entropy and maximum softmax probability as triggers for self-correction.
- **How we use it:** Supports uncertainty-guided minibatch selection.

### 7.3 Semantic Entropy and Meaning-Level Uncertainty

#### Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
- **Authors:** Kuhn, Gal, Farquhar (Oxford OATML)
- **Venue:** ICLR 2023
- **Paper:** arXiv:2302.09664
- **Why essential:** Introduces semantic entropy — clustering sampled outputs by meaning via bidirectional NLI and computing entropy over semantic clusters. Solves the problem that token-level entropy conflates meaning uncertainty with phrasing uncertainty.
- **How we use it:** Conceptual foundation. Our approach occupies a middle ground — we use token-level distributional features but annotate them with semantic context (what competing tokens *mean*, not just their probabilities).

#### Detecting Hallucinations in Large Language Models Using Semantic Entropy
- **Authors:** Farquhar, Kossen, Kuhn, Gal (Oxford)
- **Venue:** Nature, 2024
- **Paper:** nature.com/articles/s41586-024-07421-0
- **Why essential:** Extends semantic entropy to hallucination detection. Published in Nature — highest-profile validation of distributional uncertainty methods for LLMs.

#### Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs
- **Authors:** Kossen et al. (Oxford)
- **Venue:** NeurIPS 2024
- **Paper:** arXiv:2406.15927
- **Code:** github.com/OATML/semantic-entropy-probes
- **Why relevant:** Linear probes on hidden states predict semantic entropy from a single forward pass. Shows distributional information is encoded in hidden states, not just output logits.

#### Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities
- **Authors:** (NeurIPS 2024 / ICLR 2025)
- **Paper:** arXiv:2405.20003
- **Why relevant:** Replaces hard semantic clustering with von Neumann entropy over kernel matrices. State-of-the-art in semantic uncertainty.

### 7.4 Self-Consistency and Sampling-Based Confidence

#### Self-Consistency Improves Chain-of-Thought Reasoning in Language Models
- **Authors:** Wang et al.
- **Venue:** ICLR 2023
- **Paper:** arXiv:2203.11171
- **Why essential:** Sample diverse CoT paths, select most frequent answer via majority voting. +17.9% absolute on GSM8K. Majority frequency serves as confidence score. Self-consistency is both a diagnostic feature in our DAL and a key component of the text-enriched baseline (Condition D).

#### Confidence Improves Self-Consistency in LLMs (CISC)
- **Authors:** (ACL 2025)
- **Paper:** aclanthology.org/2025.findings-acl.1030.pdf
- **Why relevant:** P(True)-weighted voting achieves same accuracy with 46% fewer samples. Motivates combining logprob and sampling signals.

### 7.5 Calibration and Verbalized Confidence

#### Language Models (Mostly) Know What They Know
- **Authors:** Kadavath et al. (Anthropic)
- **Venue:** arXiv, 2022
- **Paper:** arXiv:2207.05221
- **Why essential:** Foundational work on LLM self-knowledge. P(True) provides useful confidence estimates. Temperature adjustment (T=2.5) largely fixes RLHF-induced miscalibration. Establishes metacognitive ability.

#### Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models
- **Authors:** Tian et al. (Stanford)
- **Venue:** EMNLP 2023
- **Paper:** aclanthology.org/2023.emnlp-main.330
- **Why essential:** Published at our target venue. Verbalized confidence is often better-calibrated than conditional token probabilities for RLHF models, reducing ECE by ~50%. Key insight: RLHF distorts logprob calibration while preserving metacognitive text ability.
- **Critical implication for our work:** This paper suggests that verbalized confidence (text-based, no logprobs) might be a *better* uncertainty signal than raw logprobs for RLHF'd models. This strengthens the text-enriched baseline (Condition D) and makes it a serious threat to our distributional hypothesis. We must address this head-on.

#### Calibrating Verbalized Confidence with Self-Generated Distractors
- **Authors:** Wang et al. (2025)
- **Paper:** arXiv:2509.25532
- **Why relevant:** Improves verbalized confidence calibration by generating plausible alternatives.

#### Closing the Confidence-Faithfulness Gap in Large Language Models
- **Authors:** (2026)
- **Paper:** arXiv:2603.25052
- **Why relevant:** Addresses confidence-factuality gap. Directly relevant to calibration analysis.

#### Know When You're Wrong: Aligning Confidence with Correctness for LLM Error Detection
- **Authors:** (2026)
- **Paper:** arXiv:2603.06604
- **Why relevant:** Trains models to align confidence with correctness.

### 7.6 Contrastive and Distribution-Based Methods

#### Contrastive Decoding: Open-ended Text Generation as Optimization
- **Authors:** Li et al.
- **Venue:** ACL 2023
- **Paper:** aclanthology.org/2023.acl-long.687
- **Why relevant:** Establishes that *contrasts* between distributions carry richer information than individual distributions. Conceptual motivation for our work.

#### DoLa: Decoding by Contrasting Layers Improves Factuality and Reasoning
- **Authors:** Chuang et al.
- **Venue:** ICLR 2024
- **Paper:** arXiv:2309.03883
- **Why relevant:** Layer-wise distributional contrasts detect surface-pattern reasoning. For open-weight models, an additional diagnostic feature.

#### Discovering Latent Knowledge in Language Models Without Supervision (CCS)
- **Authors:** Burns et al.
- **Venue:** ICLR 2023
- **Paper:** arXiv:2212.03827
- **Why relevant:** Hidden states encode truth even when outputs are misleading. Motivates distributional diagnostics for detecting cases where the model "knows" the right answer but the prompt elicits the wrong one.

#### Beyond Next Token Probabilities: LLM Output Signatures
- **Authors:** (2025)
- **Paper:** arXiv:2503.14043
- **Why relevant:** A token probability of 0.1 means fundamentally different things depending on whether the full distribution is peaked or diffused. Motivates shape features (kurtosis, number of modes) beyond entropy.

### 7.7 Uncertainty Decomposition

#### Decomposing Uncertainty for LLMs through Input Clarification Ensembling
- **Authors:** Hou et al.
- **Venue:** ICML 2024
- **Paper:** arXiv:2311.08718
- **Why relevant:** Aleatoric uncertainty (ambiguous input) vs. epistemic (model doesn't know) demand different prompt mutations. Could inform reflector reasoning.

#### Fine-Grained Uncertainty Decomposition: A Spectral Approach
- **Authors:** (2025)
- **Paper:** arXiv:2509.22272
- **Why relevant:** Theoretical frontier of uncertainty decomposition. Likely too expensive for GEPA's budget.

#### The Anatomy of Uncertainty in LLMs (Three-Way Decomposition)
- **Authors:** (2026)
- **Why relevant:** Input ambiguity, knowledge gaps, decoding randomness as three distinct sources. Framing for why different distributional signatures might demand different mutations.

### 7.8 Uncertainty-Calibrated Prompt Optimization (Most Directly Related)

#### How Confident Is the First Token? (UCPOF)
- **Authors:** Xie et al. (2026)
- **Paper:** arXiv:2603.18009
- **Why critical:** Most directly related existing work. Uses first-token logprobs with calibration to select few-shot exemplars and trigger RAG. +6.03% accuracy while cutting retrieval by 50.66%.
- **How we differentiate:** UCPOF uses distributional signals for prompt *construction* decisions (which examples, whether to retrieve). We use them for prompt *mutation guidance* (what to change in instructions). They are complementary, not competing.
- **Must cite and differentiate carefully.**

### 7.9 Practical Tools and Infrastructure

#### LM-Polygraph: Uncertainty Estimation for Language Models
- **Authors:** Fadeeva et al.
- **Venue:** EMNLP 2023 (Demo), TACL 2025 (Benchmark)
- **Code:** github.com/IINemo/lm-polygraph
- **Why relevant:** Unified API for 15+ uncertainty methods. Primary candidate for uncertainty computation backend.

#### LogitScope: A Framework for Analyzing LLM Uncertainty Through Information Metrics
- **Authors:** (2026)
- **Paper:** arXiv:2603.24929
- **Why relevant:** Lightweight token-level entropy and varentropy computation. Could serve as entropy computation engine.

#### OpenLogProbs
- **Authors:** Chiu et al.
- **Code:** github.com/justinchiu/openlogprobs
- **Why relevant:** Extracts full next-token distributions from APIs exposing only top-k, using bisection with logit bias. Solves API constraint for non-open-weight models.

### 7.10 Logprobs API Availability

| Provider | Access Level | Notes |
|----------|-------------|-------|
| OpenAI | Top-5 logprobs | Via Chat Completions API |
| Anthropic Claude | None | No logprobs exposed — DAL cannot function |
| Google Gemini | Logprobs | Via Vertex AI native API |
| Mistral, Together, Fireworks, xAI | Top 8-20 | Sufficient for DAL |
| Groq | Defined but unsupported | Parameter exists but not yet functional |
| vLLM (open-weight) | Full vocabulary | ~10x overhead for full vocab (~152K for Qwen 2.5) vs. top-10 |

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Information gap is too small** — distributional signals don't reveal much beyond text traces | Medium | Fatal | Phase 1 validation before full investment; pivot to negative-result paper |
| **Text-enriched baseline matches distributional** — verbalized confidence + self-consistency (no logprobs) works just as well | Medium | High | Design the text-enriched baseline seriously (Condition D); if it wins, that's a useful finding about text-based metacognition being sufficient |
| **Reflector can't reason about distributional info** — LLMs are poor at numerical reasoning | Medium | High | Semantic-first format design; test qualitative vs. numerical early in Phase 1; H3 addresses this directly |
| **Improvement too small to publish** — <2pp accuracy gain | Medium | High | Lead with analysis/metacognition findings, not accuracy. The ablation across representation types and the emergent strategy analysis are publishable even with modest accuracy gains |
| **Cost-adjusted comparison is unfavorable** — distributional GEPA wins on rollouts but loses on total compute | Medium | Medium | Report honestly; if distributional GEPA converges faster (fewer iterations), the cost difference may be small |
| **Model access constraint is too limiting** — only works on open-weight models | Low (accepted) | Medium | Frame as a property of the approach, not a limitation. Investigate self-consistency-only variant for API models |
| **Entropy is noisy on some tasks** | Medium | Medium | Task-specific analysis; report honestly where it fails |
| **Compute requirements prohibitive** | Low | Medium | Top-k approximation; pre-compute diagnostics; efficient vLLM serving |
| **Scooped before submission** | Medium | High | Move quickly; focus on unique aspects (systematic representation study, emergent strategies, text-enriched baseline comparison) |
| **EMNLP reviewers see it as a systems paper** | Medium | High | Frame around linguistic/metacognition findings, lead with analysis, choose EMNLP-native tasks |
| **Calibration doesn't improve without explicit objective** | High | Low | We don't claim it will (Section 2.5). We measure and report. If it improves, that's a bonus finding |
| **Scope creep** | High | High | Strict 4-task limit. Cut AIME-2025 if behind schedule. Phase 1 go/no-go prevents wasted effort |

---

## 9. Timeline and Milestones

**Assuming EMNLP 2026 submission deadline (~June 2026):**

- **Weeks 1-2:** Literature deep-dive (all papers in Section 7). Set up vLLM serving for Llama 3 70B. Implement basic entropy computation pipeline. Reproduce GEPA baselines on HotpotQA.
- **Weeks 3-4:** Phase 1 validation experiments (1.1, 1.2, 1.3). **Go/no-go decision by end of week 4.** If no-go, pivot (see Section 4.2).
- **Weeks 5-7:** Implement DAL. Implement text-enriched baseline (Condition D — verbalized confidence, self-consistency as text). Integrate both with GEPA's adapter protocol. Run first end-to-end experiments on HotpotQA (Conditions C, D, E).
- **Weeks 8-10:** Full Phase 2 experiments on all 4 tasks. All 6 conditions. Collect all metrics.
- **Weeks 11-12:** Phase 3 ablations (representation format, granularity, self-consistency substitution, budget sensitivity).
- **Weeks 13-14:** Phase 4 analysis. Emergent mutation strategy clustering. Error analysis. Case studies.
- **Weeks 15-16:** Paper writing. Iterate on framing based on which findings are strongest.
- **Week 17:** Internal review, final edits, submission.

**Buffer notes:**
- Weeks 8-10 are the crunch. If HotpotQA results from weeks 5-7 show clear trends, we can parallelize remaining tasks.
- If behind schedule at week 10, cut AIME-2025 and reduce ablation scope.
- Paper writing should start in parallel with Phase 4 analysis (week 13).

---

## 10. Open Questions to Resolve

1. **What is the right entropy threshold for "forking token" identification?** The literature suggests ~20% of tokens are high-entropy, but this is model- and task-dependent. We likely need adaptive thresholding per model — start with percentile-based (top 20% of entropy values) rather than fixed-bit threshold.

2. **Should self-consistency be part of the DAL or only the text-enriched baseline?** It provides strong signal but costs 2-3x more LLM calls per example. If the self-consistency ablation (Section 4.4) shows it's the dominant feature, the DAL might not need logprobs at all — which would be an important finding.

3. **What optimizer/reflector model do we use?** GEPA typically uses a strong model (GPT-4o, Claude) as the reflector. The reflector doesn't need logprobs — only the student model does. Use GPT-4o as reflector to match GEPA's published setup.

4. **How do we segment reasoning chains into "steps" for trajectory analysis?** Per-sentence? Per-reasoning-step (marked by "Step N:" or numbered lists)? Per-paragraph? This varies by task and prompting style. Start with sentence-level segmentation and tune if needed.

5. **What is the minimum set of distributional features worth the complexity?** If the ablation shows that competing token identities alone capture 80% of the improvement, the full DAL pipeline is over-engineered. Design the implementation to be modular so features can be independently toggled.

6. **How do we handle the Pareto frontier with additional dimensions?** Adding calibration and monotonicity as objectives increases frontier size and potentially dilutes selection pressure on accuracy. GEPA's existing `hybrid` frontier type (combines instance-level and objective-level domination) may handle this naturally, but needs empirical verification.

7. **Is the text-enriched baseline (Condition D) actually a fair control?** Verbalized confidence requires an extra LLM call per example ("How confident are you?"), and self-consistency requires 2-3 extra calls. The cost of Condition D might be comparable to or higher than Condition E. We need to track total tokens carefully.

---

*Last updated: March 31, 2026*
*Status: Seed document — pre-experimentation, incorporating critical analysis from codebase review*
