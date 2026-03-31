# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

**Paper**: arXiv:2507.19457v1 [cs.CL], July 25, 2025

**Authors**: Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, Omar Khattab

**Affiliations**: UC Berkeley, Stanford University, BespokeLabs.ai, Notre Dame, Databricks, MIT

**Open-source implementation**: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa) --- a production Python framework (MIT license, Python 3.10+) that generalizes the paper's algorithm into a universal text-optimization API. The framework supports optimizing any text artifact (prompts, code, configs, SVGs, agent architectures) against any evaluator.

---

## 1. Overview and Motivation

Large language models (LLMs) are increasingly adapted to downstream tasks via reinforcement learning (RL) methods like Group Relative Policy Optimization (GRPO). However, these RL methods require **tens of thousands of rollouts** (i.e., system executions) to learn new tasks, making them computationally expensive and sample-inefficient.

GEPA (**G**enetic-**P**areto, standing for **Genetic Evolutionary Prompt Adaptation**) is a **prompt optimizer** that takes a fundamentally different approach. Instead of using scalar rewards and policy gradients (as in RL), GEPA leverages the **interpretable nature of language** itself. It reads the full natural-language execution traces produced by an AI system, **reflects** on them to diagnose problems, and proposes targeted prompt updates --- all without modifying model weights.

### Key Claim

Algorithms that **learn deliberately in natural language by reflecting on execution trajectories** can make far more effective use of the strong language priors that LLMs already possess, compared to standard RL approaches that collapse rich traces into scalar rewards.

### Headline Results

- GEPA outperforms GRPO by **10% on average** and by up to **20%** on individual tasks.
- GEPA uses up to **35x fewer rollouts** than GRPO (24,000 rollouts with LoRA).
- GEPA outperforms **MIPROv2** (the leading prompt optimizer) by over **10%** across two LLMs.
- Promising results as an **inference-time search strategy** for code optimization (NPUEval, KernelBench).

---

## 2. Problem Statement

### Compound AI Systems

GEPA targets **compound AI systems** --- modular systems composed of one or more LLM invocations, potentially interleaved with external tool calls, orchestrated through arbitrary control flow. This definition covers:

- Agents (e.g., ReAct)
- Multi-agent systems (e.g., Archon)
- General-purpose scaffolding techniques

Formally, a compound AI system is defined as:

```
Phi = (M, C, X, Y)
```

Where:
- **M** = a sequence of language modules M_1, ..., M_|M|
- Each module **M_i = (pi_i, theta_i, X_i, Y_i)** where:
  - `pi_i` = system prompt (instructions + few-shot demonstrations)
  - `theta_i` = underlying model weights
  - `X_i, Y_i` = input/output schemas
- **C** = control flow logic (sequencing, conditional invocation, tool APIs)
- **X, Y** = global input/output schemas

### Optimization Objective

Given a dataset D_train = {(x, m)_i} of task instances, the goal is to find optimal parameters (prompts and/or weights) that maximize a metric mu over a task distribution, subject to a **rollout budget B**:

```
<Pi*, Theta*> = arg max E_{(x,m)~T} [mu(Phi(x; <Pi, Theta>), m)]
    subject to: #rollouts <= B
```

The core challenge: **How can we extract maximal learning signal from every expensive rollout to enable effective adaptation in low-data or budget-constrained settings?**

### How This Maps to the Codebase

In the `gepa` framework, these formal concepts have concrete representations:

- **Candidate** = `dict[str, str]` --- a mapping from named component to its text. For example, `{"query_writer": "Given a question, write a search query...", "summarizer": "Summarize the passages..."}`. Each key is a named "predictor" (module) in the system; each value is its current prompt/instruction text.
- **System execution** = the user implements a `GEPAAdapter` protocol (`src/gepa/core/adapter.py`) with three methods:
  - `evaluate(batch, candidate, capture_traces)` --- runs the system and returns an `EvaluationBatch` containing per-example `outputs`, `scores` (floats, higher is better), optional `trajectories`, and optional `objective_scores` (for multi-objective optimization).
  - `make_reflective_dataset(candidate, eval_batch, components_to_update)` --- extracts a JSON-serializable dataset from execution traces, keyed by component name. The recommended schema per record is `{"Inputs": ..., "Generated Outputs": ..., "Feedback": ...}`.
  - `propose_new_texts` (optional) --- override GEPA's default LLM-based instruction proposal with custom logic.
- **Metric mu** = the `scores` list in `EvaluationBatch`. GEPA uses `sum(scores)` for minibatch acceptance and `mean(scores)` over the validation set for Pareto tracking.
- **Budget B** = the `max_metric_calls` parameter, which counts individual evaluation calls (not iterations).

The framework supports three optimization modes via the `optimize_anything()` API:
1. **Single-Task Search** (no dataset): solve one hard problem; the candidate *is* the solution (e.g., circle packing).
2. **Multi-Task Search** (dataset, no valset): solve a batch of related problems with cross-task transfer (e.g., CUDA kernels for multiple operations).
3. **Generalization** (dataset + valset): build a skill that transfers to unseen problems (e.g., prompt optimization for AIME math).

---

## 3. GEPA: The Algorithm

GEPA is built on three core principles:

1. **Genetic prompt evolution** (Section 3.1)
2. **Reflection using natural language feedback** (Section 3.2)
3. **Pareto-based candidate selection** (Section 3.3)

### 3.1 Genetic Optimization Loop

GEPA maintains a **candidate pool** P, where each candidate is a concrete instantiation of the learnable parameters (prompts) of the compound AI system. The optimization proceeds as follows:

1. **Initialize**: The candidate pool starts with just the base system's parameters.
2. **Iterate**: In each iteration, GEPA:
   - **Selects** a promising candidate from the pool (via Pareto-based selection).
   - **Selects a target module** within the system to improve (via round-robin).
   - **Generates rollouts** on a minibatch of tasks sampled from D_feedback.
   - **Gathers feedback**: Collects execution traces, scores, and textual feedback.
   - **Reflects and mutates**: Uses an LLM to reflectively examine traces and propose a new prompt.
   - **Evaluates**: Tests the new candidate on the minibatch.
   - **Accepts or rejects**: If the new candidate improves performance, it's added to the pool.
3. **Return**: After budget exhaustion, return the candidate with the best aggregate performance on D_pareto.

Each new candidate **inherits learning signals** from its parents along the ancestry tree, plus signals from the current rollout. This enables GEPA to accumulate lessons across the entire optimization trajectory.

**Concrete acceptance criterion in the codebase**: After the reflection LM proposes a new prompt for a component, GEPA evaluates the new candidate on the same minibatch. Acceptance is determined by strict improvement: `sum(new_subsample_scores) > sum(old_subsample_scores)`. Only then does GEPA spend budget evaluating the new candidate on the full validation set and adding it to the candidate pool (`GEPAState`). This is implemented in `GEPAEngine.run()` at `src/gepa/core/engine.py`.

**State persistence**: After every iteration, `GEPAState` is saved to `run_dir` via pickle (or cloudpickle for dynamically-generated DSPy signatures). If the process is interrupted, GEPA resumes from the last saved state when re-run with the same `run_dir`. Graceful stopping is supported by placing a `gepa.stop` file in the run directory --- a `FileStopper` checks for this before each iteration.

### Candidate Proposal Strategies

GEPA proposes new candidates through two strategies:

- **Mutation**: Modifying an existing candidate's prompts based on reflective feedback (primary strategy).
- **Crossover (Merge)**: Combining the best modules from two different candidates that have evolved along different lineages (see GEPA+Merge variant).

### 3.2 Reflective Prompt Mutation

This is the heart of GEPA. The key insight is that natural language execution traces generated during system execution offer rich **visibility** and **diagnostic value**:

- They capture the behavior of each module.
- They record intermediate inferences and reasoning steps.
- They can be paired with final outcomes (success/failure).

**How Reflection Works:**

1. Given a selected candidate to mutate, GEPA runs it on a minibatch of tasks.
2. It examines the **execution traces** of the target module --- inputs, outputs, and reasoning.
3. An LLM **reflectively examines** this information, performing implicit **credit assignment**:
   - Attributing successes or failures to specific elements of the module's prompt.
   - Identifying what worked and what didn't.
4. The LLM proposes **new instructions** for the target module.
5. A new candidate is created with the updated prompt.

**Evaluation Trace as Diagnostic Signal:**

Beyond execution traces, GEPA also leverages **evaluation traces** --- the intermediate steps produced by the evaluation metric mu (e.g., compiler error messages, profiling results). GEPA introduces a **feedback function** mu_f that:
- Identifies relevant textual traces from evaluation.
- Returns them alongside the final score as `feedback_text`.
- Can provide **module-level feedback** (e.g., in multi-hop systems, feedback after each hop).

**Meta-Prompt for Reflection:**

GEPA uses a meta-prompt that instructs the LLM to:
1. Read the current instruction and input/output examples with feedback.
2. Identify domain-specific factual information and generalizable strategies.
3. Write a new, improved instruction.

The actual default meta-prompt template (from `InstructionProposalSignature` in `src/gepa/strategies/instruction_proposal.py`) is:

```
I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant
along with the assistant's response for each of them, and some feedback on how
the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task
description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all
niche and domain specific factual information about the task and include it in
the instruction, as a lot of it may not be available to the assistant in the
future. The assistant may have utilized a generalizable strategy to solve the
task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.
```

The `<curr_param>` placeholder is replaced with the current component text, and `<side_info>` is replaced with a markdown-formatted rendering of the reflective dataset (the inputs, outputs, and feedback from the minibatch). Users can override this template per-component by passing `reflection_prompt_template` as a `dict[str, str]` mapping component names to custom templates. Each custom template must include both `<curr_param>` and `<side_info>` placeholders.

The output extractor (`InstructionProposalSignature.output_extractor`) parses the LLM response by extracting text between the first and last ``` blocks. This is the new proposed instruction.

**Multimodal support**: The reflective dataset renderer handles `Image` objects. When images are present, the meta-prompt is sent as an OpenAI-compatible multimodal messages list with `[IMAGE-N]` placeholders in the text and image content parts appended, allowing reflection LMs to analyze visual execution traces (e.g., rendered outputs, screenshots).

### 3.3 Pareto-Based Candidate Selection

A naive strategy of always selecting the best-performing candidate leads to **local optima** --- the optimizer gets stuck trying to improve one dominant strategy.

GEPA instead employs a **Pareto-based "illumination" strategy** (inspired by MAP-Elites):

1. **Build instance-wise Pareto sets**: For each training instance i, identify the highest score achieved by any candidate. Find all candidates that achieve this best score on at least one instance.
2. **Prune dominated candidates**: If Candidate A's best scores are a strict subset of Candidate B's best scores, remove A.
3. **Stochastically sample**: From the remaining non-dominated candidates, sample one with probability proportional to the number of instances where it achieves the best score.

**Why this matters**: This strategy ensures:
- **Diversity**: Different "winning" strategies are preserved.
- **Exploration**: The optimizer doesn't fixate on a single dominant candidate.
- **Balanced search**: Resources are allocated to candidates that have demonstrated value on different subsets of the data.

**Implementation detail**: The sampling is implemented in `select_program_candidate_from_pareto_front()` in `src/gepa/gepa_utils.py`. After domination pruning, it builds a sampling list where each candidate appears once per validation instance where it is Pareto-best, then draws uniformly from this list. This naturally gives higher probability to candidates that win on more instances. The domination check (`is_dominated()`) iteratively removes candidates whose front-coverage is a strict subset of another's, starting from the lowest-scoring candidates.

**Four frontier tracking modes**: The codebase extends the paper's instance-based Pareto frontier with three additional modes, configured via the `frontier_type` parameter:
- `"instance"` (default, paper's approach) --- tracks best score per validation example.
- `"objective"` --- tracks best score per named objective metric (requires evaluator to return `objective_scores`).
- `"hybrid"` --- combines both instance-level and objective-level fronts.
- `"cartesian"` --- tracks per (example, objective) pair, the finest granularity.

These are maintained in `GEPAState` (`src/gepa/core/state.py`) as separate dictionaries: `program_at_pareto_front_valset`, `program_at_pareto_front_objectives`, and `program_at_pareto_front_cartesian`.

---

## 4. System Aware Merge (Crossover)

GEPA+Merge extends GEPA with a **system-aware crossover** strategy:

1. Sample two candidates from different lineages (not ancestors of each other).
2. Find a common ancestor.
3. For each module in the system, strategically select the prompt from one of the two candidates:
   - If a module has evolved (changed) in one candidate but not the other, pick the evolved version.
   - If both have evolved, pick the one from the higher-scoring candidate.
   - If neither has evolved, keep the ancestor's version.

This creates a new candidate that combines the best independently-evolved modules from different optimization lineages.

### Merge Implementation Details (from `src/gepa/proposer/merge.py`)

The merge process in code is more nuanced than the paper summary:

1. **Candidate pair selection**: `find_common_ancestor_pair()` samples two candidates from the Pareto front's dominator set (up to `max_attempts=10` tries). It verifies they are not ancestors of each other by traversing the `parent_program_for_candidate` ancestry tree stored in `GEPAState`.

2. **Common ancestor filtering** (`filter_ancestors()`): A common ancestor is only considered if:
   - The (i, j, ancestor) triplet hasn't been tried before (tracked in `merges_performed`).
   - The ancestor's aggregate score is <= both descendants' scores (the merge should combine improvements, not regress).
   - The triplet has "desirable predictors" --- at least one component where one descendant kept the ancestor's text and the other evolved it (checked by `does_triplet_have_desirable_predictors()`).

3. **Module-level merging**: For each component (predictor):
   - If one descendant kept the ancestor's text and the other changed it: use the changed version (the evolved one).
   - If both descendants changed the text differently: pick the one from the higher-scoring descendant (with random tiebreaking).
   - If both descendants have the same text (whether changed or not): use either (defaults to id1).

4. **Acceptance**: The merged candidate is evaluated on a strategically sampled subset of validation instances (`select_eval_subsample_for_merged_program()`) drawn from indices where the two parents disagree in performance. The merge is accepted if `sum(merged_scores) >= max(parent1_sum, parent2_sum)`.

5. **Budget control**: Merges are limited to `max_merge_invocations` (default 5), and a merge is only attempted after the previous iteration produced a new accepted candidate (`last_iter_found_new_program` flag). When a merge is attempted (whether accepted or rejected), the reflective mutation step is skipped for that iteration.

### Additional Candidate Selection Strategies (Beyond the Paper)

The codebase implements four candidate selection strategies, not just the two discussed in the paper:

| Strategy | Class | Behavior |
|----------|-------|----------|
| `"pareto"` (default) | `ParetoCandidateSelector` | Paper's Pareto-based illumination: sample from non-dominated candidates weighted by front frequency |
| `"current_best"` | `CurrentBestCandidateSelector` | Paper's greedy ablation: always pick the candidate with highest aggregate validation score |
| `"epsilon_greedy"` | `EpsilonGreedyCandidateSelector` | With probability epsilon (default 0.1), pick a random candidate; otherwise pick the best |
| `"top_k_pareto"` | `TopKParetoCandidateSelector` | Pareto sampling restricted to the top K candidates (default K=5) by aggregate score |

### Component (Module) Selection Strategies

- `"round_robin"` (default, `RoundRobinReflectionComponentSelector`): Cycles through components in order. Each candidate tracks its own round-robin pointer (`named_predictor_id_to_update_next_for_program_candidate`), so different candidates may be evolving different modules at any given iteration.
- `"all"` (`AllReflectionComponentSelector`): Selects all components for modification every iteration. The reflection LM produces new text for each component independently.

---

## 5. Evaluation Setup

### 5.1 Benchmarks

GEPA is evaluated on four diverse tasks:

| Benchmark | Task Type | System Structure | Feedback Signal |
|-----------|-----------|-----------------|-----------------|
| **HotpotQA** | Multi-hop QA | Multi-hop retrieval + summarization + answer generation | Set of relevant docs remaining to be retrieved |
| **IFBench** | Instruction following | 2-stage: answer then rewrite following constraints | Descriptions of satisfied/failed constraints |
| **HoVer** | Claim verification | 3-hop retrieval with query writers + summarizers | Set of correct/remaining documents |
| **PUPA** | Privacy-preserving delegation | Query rewriter + response rewriter with trusted/untrusted models | Response quality score + PII leakage score |

### 5.2 Models

- **Qwen3 8B** (open-source): temperature 0.6, top-p 0.95, top-k 20
- **GPT-4.1 Mini** (commercial): temperature 1.0, accessed via OpenAI API

### 5.3 Baselines

| Optimizer | Type | Description |
|-----------|------|-------------|
| **Baseline** | None | Base program without optimization |
| **MIPROv2** | Prompt optimizer | Joint instruction + few-shot optimization via Bayesian optimization (TPE). 2,270--6,926 rollouts. |
| **GRPO** | RL (weight-based) | Group Relative Policy Optimization with LoRA. Fixed 24,000 rollouts (500 training steps). |

---

## 6. Results and Analysis

### 6.1 Main Results (Table 1)

#### Qwen3 8B

| Model | HotpotQA | IFBench | HoVer | PUPA | Aggregate | Improvement |
|-------|----------|---------|-------|------|-----------|-------------|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 48.85 | -- |
| MIPROv2 | 55.33 | 36.22 | 47.33 | 81.55 | 55.11 | +6.26 |
| GRPO | 43.33 | 35.88 | 38.67 | 86.66 | 51.14 | +2.29 |
| **GEPA** | 62.33 | **38.61** | **52.33** | **91.85** | **61.28** | **+12.44** |
| GEPA+Merge | **64.33** | 28.23 | 51.67 | 86.26 | 57.62 | +8.78 |

#### GPT-4.1 Mini

| Model | HotpotQA | IFBench | HoVer | PUPA | Aggregate | Improvement |
|-------|----------|---------|-------|------|-----------|-------------|
| Baseline | 38.00 | 47.79 | 46.33 | 78.57 | 52.67 | -- |
| MIPROv2 | 58.00 | 49.15 | 48.33 | 83.37 | 59.71 | +7.04 |
| **GEPA** | **69.00** | 52.72 | 51.67 | 94.47 | 66.97 | +14.29 |
| **GEPA+Merge** | 65.67 | **55.95** | **56.67** | **96.46** | **68.69** | **+16.02** |

### 6.2 Key Observations

#### Observation 1: GEPA is Highly Sample-Efficient and Outperforms RL

- GEPA outperforms GRPO (24,000 rollouts with LoRA) by up to **19%** while using up to **35x fewer rollouts**.
- GEPA achieves optimal test-set performance on HotpotQA, IFBench, HoVer, and PUPA with only **6,438**, **6,858**, **678**, and **2,157** rollouts respectively.
- GEPA matches GRPO's best validation scores after only **402**, **330**, **1,179**, and **306** rollouts --- up to **78x greater sample efficiency**.
- If counting only **training rollouts** (not validation), GEPA needs just **737**, **79**, **558**, and **269** rollouts.

#### Observation 2: Instruction-Optimization Alone Outperforms Joint Instruction + Few-Shot Optimization

- GEPA consistently outperforms MIPROv2 by up to **11.1%** for GPT-4.1 Mini and **10.3%** for Qwen3 8B.
- GEPA and GEPA+Merge more than double the aggregate gains over baseline compared to MIPROv2 (+14.29% and +16.02% vs +7.04% for GPT-4.1 Mini).
- GEPA's prompts contain detailed **declarative instructions** rather than relying on few-shot demonstrations.
- GEPA achieves a **lower generalization gap** (difference between validation and test performance) compared to prior methods.

#### Observation 3: Pareto-Based Sampling Provides a Distinct Advantage

- GEPA with Pareto-based sampling outperforms the SelectBestCandidate ablation by up to **8.17%** and maintains an aggregate margin of **+6.4%** across all benchmarks.
- SelectBestCandidate tends to stall after finding one good strategy, spending all remaining budget trying to improve it.
- Pareto-based sampling explores the search space more effectively, maintaining a balanced search tree.

#### Observation 4: Instruction-Optimized Prompts are Cheaper and More Efficient

- GEPA's prompts are up to **9.2x shorter** than MIPROv2's prompts.
- On average, GEPA's prompts are around **33% of the size** of MIPROv2's.
- This is because MIPROv2 relies heavily on few-shot demonstrations, which can be very long. GEPA evolves compact, instruction-only prompts.
- Shorter prompts reduce runtime cost (API input tokens) and latency.

#### Observation 5: System-Aware Merge Can Provide Large Additional Gains

- GEPA+Merge can outperform GEPA by up to **5%**, providing an aggregate **2% additional improvement**.
- Merge works by combining complementary strategies from different optimization lineages.
- Works especially well with Qwen3 8B; for GPT-4.1 Mini, results are mixed --- suggesting hyperparameters for when to invoke crossover need further tuning.

---

## 7. GEPA for Inference-Time Search

Beyond optimization, GEPA can serve as an **inference-time search strategy**. By passing the full task set as both D_train and D_pareto, GEPA can intentionally "overfit" to a specific set of tasks, iteratively proposing better solutions. Lessons from one task can also transfer to others.

### NPU Kernels (AMD XDNA2 Architecture)

- Task: Generate kernels for AMD NPUs using NPUEval benchmark.
- **Sequential10** (10 sequential refinements with GPT-4o): 4.25% mean vector utilization.
- **Sequential10 + RAG** (adding retrieval-augmented generation): 16.33%.
- **Sequential10 + MIPROv2**: 19.03%.
- **Sequential10 + GEPA** (without RAG): **30.52%** mean vector utilization, with some kernels reaching 70%.
- A single GEPA-generated prompt (no runtime RAG) enables Sequential10 to reach **26.85%**.

### CUDA Kernels (KernelBench)

- Task: Generate CUDA code for NVIDIA V100 GPUs.
- GEPA with GPT-4o boosts the `fast_1` score (fraction of kernels faster than Pytorch-eager) from near 0% to **over 20%** across 35 representative tasks.

---

## 8. How GEPA Compares to Related Work

### vs. Reinforcement Learning (GRPO)

| Aspect | GRPO | GEPA |
|--------|------|------|
| Learning signal | Scalar reward | Natural language reflection on traces |
| What it modifies | Model weights (LoRA) | System prompts only |
| Sample efficiency | ~24,000 rollouts | ~700--6,500 rollouts |
| Hardware requirements | GPU for training (1x H100/A100) | LLM inference only |
| Applicability | Open-weight models only | Any LLM (open or proprietary) |

### vs. MIPROv2

| Aspect | MIPROv2 | GEPA |
|--------|---------|------|
| Optimization target | Instructions + few-shot demos | Instructions only |
| Search method | Bayesian optimization (TPE) | Evolutionary with Pareto selection |
| Feedback used | Scores only | Scores + natural language traces |
| Prompt length | Long (many demos) | Short (instructions only, up to 9.2x shorter) |
| Performance | Baseline | +6--7% improvement over MIPROv2 |

### vs. EvoPrompt / AlphaEvolve

| Aspect | EvoPrompt | AlphaEvolve | GEPA |
|--------|-----------|-------------|------|
| Mutation type | Random | Code-level | Reflective (uses feedback) |
| Scope | Single prompt | Single problem | Compound AI systems |
| Feedback | None (population-based) | Code correctness | Rich NL traces + evaluation feedback |
| Selection | Population-based | Solution-based | Pareto-front based |

---

## 9. Algorithm Details

### Algorithm 1: GEPA Core Loop

```
Input: System Phi, dataset D_train, eval metric mu, feedback function mu_f, budget B

1. Split D_train into D_feedback and D_pareto (|D_pareto| = n_pareto)
2. Initialize candidate pool P = {Phi}, ancestry A = [None]
3. Evaluate Phi on D_pareto, record scores S_Phi
4. While budget B not exhausted:
   a. k = SelectCandidate(P, S)          # Pareto-based selection
   b. j = SelectModule(Phi_k)             # Round-robin module selection
   c. M = minibatch of size b from D_feedback
   d. Gather feedback, scores, traces for Phi_k[j] on M using mu_f
   e. pi'_j = UpdatePrompt(pi_j, feedbacks, traces[j])  # Reflective mutation
   f. Phi' = Copy of Phi_k with module j updated by pi'_j
   g. sigma, sigma' = avg score on M (before, after)
   h. If sigma' improved:
      - Add Phi' to P, record ancestry
      - Evaluate Phi' on full D_pareto
5. Return Phi* maximizing average score on D_pareto
```

**Mapping to codebase** (`GEPAEngine.run()` at `src/gepa/core/engine.py`):
- Steps 1-3: `initialize_gepa_state()` creates a `GEPAState` with the seed candidate and its full validation evaluation.
- Step 4a: `ReflectiveMutationProposer` calls `self.candidate_selector.select_candidate_idx(state)`.
- Step 4b: `self.module_selector(state, trajectories, scores, candidate_idx, candidate)` returns component names.
- Step 4c: `self.batch_sampler.sample(trainset)` yields the next minibatch.
- Step 4d: `adapter.evaluate(batch, candidate, capture_traces=True)` then `adapter.make_reflective_dataset(...)`.
- Step 4e: `InstructionProposalSignature.run_with_metadata(lm, ...)` calls the reflection LM.
- Step 4f-g: `adapter.evaluate(batch, new_candidate, capture_traces=False)` --- acceptance uses `sum(new_scores) > sum(old_scores)`.
- Step 4h: `_run_full_eval_and_add()` evaluates on the full valset and calls `state.update_state_with_new_program()`.
- The engine also checks for merge opportunities at the top of each iteration (before reflective mutation): if `merge_proposer.merges_due > 0` and the last iteration found a new candidate, it attempts a merge first. If a merge is attempted (accepted or rejected), the reflective step is skipped for that iteration.

### Algorithm 2: Pareto-Based Candidate Selection

```
Input: Candidate pool P, score matrix S

1. For each task instance i:
   a. s*[i] = max score across all candidates on instance i
   b. P*[i] = {candidates achieving s*[i] on instance i}
2. C = union of all P*[i] (unique candidates)
3. Remove dominated candidates:
   - Phi is dominated if there exists another candidate that achieves
     the best score on all instances where Phi does, plus more
4. For remaining candidates, count f[Phi] = number of instances
   where Phi is in the Pareto-best set
5. Sample Phi_k with probability proportional to f[Phi_k]
6. Return index k
```

---

## 10. Qualitative Analysis: How Prompts Evolve

The paper provides a detailed case study of prompt evolution on the PUPA task (privacy-preserving delegation):

| Node | Score | Key Changes |
|------|-------|-------------|
| **0** (Base) | 82.26 | Simple 2-line instruction: "create a privacy-preserving request" |
| **2** | 90.99 | Added detailed task description, key points on privacy preservation, query understanding, quality maximization |
| **4** | 94.44 | Added domain-specific details, common reformulation strategies, explanation requirements, input/output format specification |
| **5** | 94.67 | Restructured with task overview, output format, best practices, common pitfalls |
| **8** | 96.02 | Enforced transparent privacy reasoning, detailed abstraction techniques, fictional character handling |
| **11** (Best) | 97.60 | Strict protocol: bans partial redaction, requires full abstraction, zero leakage tolerance, always justify approach |

This trajectory shows how **each iteration adds targeted, task-specific nuances** informed by feedback from actual failures, progressively building a comprehensive, high-performing prompt.

---

## 11. Limitations and Future Work

1. **Boundary between prompt-based and weight-based optimization** is not well understood. Weight updates may outperform prompting when rollouts are cheap and data is abundant.

2. **LoRA vs. full-parameter finetuning**: The paper uses LoRA for GRPO baselines. Full-parameter GRPO may perform differently but typically requires 100K--512K rollouts.

3. **No few-shot/exemplar optimization**: GEPA currently optimizes instructions only. Incorporating in-context examples could further improve performance.

4. **Validation overhead**: Most of GEPA's rollout budget goes to candidate validation, not learning. Future work could explore smaller Pareto validation sets or dynamic subsampling.

5. **Feedback engineering**: Identifying which execution or evaluation traces provide the most valuable learning signal is an underexplored direction.

6. **Combining with RL**: Using GEPA's language-based lessons to perform RL rollouts could yield additive gains and help unify prompt- and weight-based optimization.

---

## 12. Conclusion

GEPA demonstrates that **language-based reflection can offer a scalable strategy for optimizing complex real-world AI workflows**, especially in resource-constrained settings. By:

- Reflecting on natural language execution traces instead of collapsing them to scalar rewards,
- Maintaining a diverse Pareto front of candidates instead of greedy best-first search,
- Evolving compact, declarative instructions instead of relying on long few-shot demonstrations,

GEPA achieves superior performance with dramatically fewer rollouts than both RL-based (GRPO) and prior prompt optimization (MIPROv2) approaches. It also shows promise as an inference-time search strategy for challenging code generation tasks.

The work suggests a broader insight: **modern LLMs' strong language understanding and instruction-following capabilities make natural language itself a powerful medium for learning** --- one that can, in many practical settings, outperform gradient-based adaptation.

---

## 13. The GEPA Software Framework: Beyond the Paper

The open-source `gepa` Python package (`src/gepa/`) generalizes the paper's algorithm into a production framework with capabilities not discussed in the paper.

### 13.1 Two Public APIs

**`gepa.optimize()`** (`src/gepa/api.py`, ~415 LOC) is the lower-level API matching the paper's algorithm closely. It takes a `seed_candidate` dict, separate `trainset`/`valset`, a `GEPAAdapter` implementation, a `reflection_lm` for proposal, and returns a `GEPAResult`. This API requires implementing the full adapter protocol.

**`gepa.optimize_anything()`** (`src/gepa/optimize_anything.py`, ~1555 LOC) is the higher-level universal API. It takes a seed candidate (string or dict), an evaluator function, and a natural language `objective` description. The framework handles adapter construction, prompt rendering, and LLM reflection internally. This is the primary recommended API.

Key `optimize_anything` concepts:
- **ASI (Actionable Side Information)**: The text-optimization analogue of a gradient. Where gradients tell a numerical optimizer which direction to move, ASI tells the LLM proposer *why* a candidate failed and *how* to fix it. ASI is captured via `oa.log()` calls within the evaluator or as the second element of an evaluator return tuple `(score, side_info)`.
- **Seedless mode**: When `seed_candidate=None`, the reflection LM bootstraps the first candidate from the `objective` and optional `background` strings, enabling exploratory optimization without a starting artifact.

### 13.2 Evaluation Caching

When `cache_evaluation=True`, GEPA maintains an `EvaluationCache` (`src/gepa/core/state.py`) that stores `(score, output, objective_scores)` for each `(candidate, example)` pair. Candidates are hashed via SHA-256 of their JSON-serialized sorted key-value pairs (`_candidate_hash()`). Before evaluating a batch, `evaluate_with_cache_full()` splits it into cached hits and uncached misses, only calling the adapter for misses. This avoids redundant evaluations when the same candidate is re-evaluated on overlapping validation subsets.

### 13.3 Callback System

The engine emits 16+ typed events through a `GEPACallback` protocol (`src/gepa/core/callbacks.py`), enabling observability without modifying engine code:

| Event | When Fired |
|-------|------------|
| `OptimizationStartEvent` | Before first iteration, with seed candidate and dataset sizes |
| `IterationStartEvent` | At the beginning of each iteration, with full `GEPAState` access |
| `CandidateSelectedEvent` | After Pareto selection chooses a candidate to evolve |
| `MinibatchSampledEvent` | After sampling training examples for reflection |
| `EvaluationStartEvent` / `EvaluationEndEvent` | Before/after adapter evaluation calls |
| `ReflectiveDatasetBuiltEvent` | After constructing the reflective dataset from traces |
| `ProposalStartEvent` / `ProposalEndEvent` | Before/after LLM reflection proposes new text |
| `CandidateAcceptedEvent` / `CandidateRejectedEvent` | After minibatch acceptance test |
| `MergeAttemptedEvent` / `MergeAcceptedEvent` / `MergeRejectedEvent` | During crossover |
| `ValsetEvaluatedEvent` | After evaluating an accepted candidate on the full validation set |
| `ParetoFrontUpdatedEvent` | After the Pareto front is updated with displaced/new candidates |
| `BudgetUpdatedEvent` | Real-time budget consumption updates via hooks on `GEPAState.increment_evals()` |
| `StateSavedEvent` | After state checkpoint is written to disk |
| `OptimizationEndEvent` | At termination, with final state and best candidate index |

Callbacks are synchronous and observational --- they cannot modify state.

### 13.4 Experiment Tracking

GEPA integrates with **MLflow** and **Weights & Biases** via an `ExperimentTracker` (`src/gepa/logging/experiment_tracker.py`). Both can be used simultaneously. The tracker logs:
- Per-iteration metrics: validation scores, Pareto front aggregates, budget consumption.
- Candidate table: iteration, candidate index, parent IDs, per-component texts, validation score.
- Candidate tree visualization.
- Configuration snapshot.

An `attach_existing` mode allows GEPA to log into an already-active MLflow/W&B run when embedded in a larger training loop.

### 13.5 Stopping Conditions

Multiple stoppers can be combined via `CompositeStopper` (any-of semantics):

| Stopper | Trigger |
|---------|---------|
| `MaxMetricCallsStopper` | Total evaluation calls exceed `max_metric_calls` |
| `FileStopper` | A `gepa.stop` file appears in `run_dir` |
| `TimeoutStopCondition` | Wall-clock time limit reached |
| `SignalStopper` | OS signal received (e.g., SIGTERM) |
| `NoImprovementStopper` | No improvement for N iterations |

The budget is tracked at the individual evaluation level (each `(example, candidate)` evaluation counts as 1), not at the iteration level. This is important because a single iteration involves multiple evaluations: the minibatch evaluation with traces, the minibatch re-evaluation with the new candidate, and (if accepted) the full validation set evaluation.

### 13.6 Pre-built Adapters

The framework ships with 8 adapter implementations for common integration patterns:

| Adapter | Location | Purpose |
|---------|----------|---------|
| `DefaultAdapter` | `adapters/default_adapter/` | Single-turn LLM prompt optimization. Uses `litellm` for model calls. Default evaluator: `ContainsAnswerEvaluator` (checks if expected answer is a substring of the response). |
| `DSPyAdapter` | `adapters/dspy_adapter/` | Integrates with DSPy programs, optimizing module prompts. |
| `DSPyFullProgramAdapter` | `adapters/dspy_full_program_adapter/` | Evolves entire DSPy program structures, not just prompts. |
| `GenericRAGAdapter` | `adapters/generic_rag_adapter/` | Vector-store-agnostic RAG pipeline optimization. Supports ChromaDB, Weaviate, Qdrant, Pinecone. |
| `MCPAdapter` | `adapters/mcp_adapter/` | Optimizes systems using Model Context Protocol tools. |
| `TerminalBenchAdapter` | `adapters/terminal_bench_adapter/` | Terminus terminal-use agent optimization. |
| `AnyMathsAdapter` | `adapters/anymaths_adapter/` | Math problem-solving systems. |
| `OptimizeAnythingAdapter` | `adapters/optimize_anything_adapter/` | Internal adapter powering the `optimize_anything()` API. |

### 13.7 Upstream Integrations

GEPA has been integrated into several external frameworks:
- **DSPy** (`dspy.GEPA`) --- tutorial-supported optimizer.
- **MLflow** --- `mlflow.genai.optimize_prompts()` uses GEPA under the hood.
- **Comet ML Opik** --- GEPA Optimizer for prompt engineering.
- **Pydantic AI** --- prompt optimization for Pydantic AI agents.
- **OpenAI Cookbook** --- self-evolving agents recipe.
- **HuggingFace Cookbook** --- prompt optimization guide.
- **Google ADK** --- Agent Development Kit integration.

### 13.8 GEPAResult: What You Get Back

`GEPAResult` (`src/gepa/core/result.py`) is the immutable output of optimization:

```python
result = gepa.optimize(...)

result.best_candidate      # dict[str, str] --- the optimized component texts
result.best_idx             # int --- index of the highest-scoring candidate
result.candidates           # list[dict[str, str]] --- all explored candidates
result.parents              # list[list[int | None]] --- ancestry tree
result.val_aggregate_scores # list[float] --- per-candidate mean validation score
result.val_subscores        # list[dict[DataId, float]] --- per-candidate, per-example scores
result.per_val_instance_best_candidates  # dict[DataId, set[int]] --- Pareto front
result.total_metric_calls   # int --- total evaluations consumed
result.discovery_eval_counts # list[int] --- evaluation count at time each candidate was discovered
```

Results support JSON serialization/deserialization via `to_dict()` / `from_dict()` for storage and analysis.

### 13.9 Batch Sampling

The default batch sampler is `EpochShuffledBatchSampler` (`src/gepa/strategies/batch_sampler.py`): it shuffles the training set at the start of each epoch, then yields non-overlapping minibatches of size `reflection_minibatch_size` (default 3). This ensures all training examples are seen before any is repeated, providing uniform coverage of the training distribution. Users can implement custom `BatchSampler` subclasses for alternative strategies (e.g., curriculum learning, difficulty-based sampling).
