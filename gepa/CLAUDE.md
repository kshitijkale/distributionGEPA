# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is GEPA

GEPA (Genetic-Pareto) is a Python framework for optimizing text components (AI prompts, code, instructions) using LLM-based reflection and Pareto-efficient evolutionary search. It works on any system with textual parameters, requiring no weight updates.

## Build & Development Commands

All commands must be run through `uv` (Rust-based Python package manager):

```bash
# Setup
uv sync --extra dev --python 3.11

# Tests
uv run pytest                          # All tests
uv run pytest tests/test_foo.py        # Single test file
uv run pytest tests/test_foo.py -k "test_name"  # Single test

# Linting & formatting
uv run ruff check src/                 # Lint check
uv run ruff check --fix-only src/      # Auto-fix
uv run ruff format src/                # Format

# Type checking
uv run pyright src/                    # Full type check
uv run pyright src/gepa/strategies/    # Check specific module

# Pre-commit
uv run pre-commit run --all-files
```

## Code Style

- **Linter/formatter**: ruff (line length 120, double quotes, 4-space indent)
- **Type checker**: pyright
- **Python target**: 3.10+ (tested on 3.10–3.14)
- **No relative imports** (enforced by ruff `ban-relative-imports = "all"`)
- Follows Google Python Style Guide
- ruff config is in `pyproject.toml` under `[tool.ruff]`

## Architecture

### Two Main Entry Points

1. **`gepa.optimize()`** (`src/gepa/api.py`) — Primary API for optimizing prompts in compound AI systems. Takes a seed candidate (dict of component name → text), training/validation data, an adapter, and configuration. Returns `GEPAResult`.

2. **`gepa.optimize_anything()`** (`src/gepa/optimize_anything.py`) — Universal API for optimizing any text artifact (code, configs, SVGs). Takes a seed candidate, evaluator function, and objective description.

### Core Engine (`src/gepa/core/`)

- **`engine.py`** — `GEPAEngine`: orchestrates the optimization loop (select candidate → select module → rollout → reflect → mutate → evaluate → accept/reject).
- **`state.py`** — `GEPAState`: manages candidate pool, ancestry tree, Pareto frontier tracking, evaluation cache. Supports multiple frontier types: `instance`, `objective`, `hybrid`, `cartesian`.
- **`adapter.py`** — `GEPAAdapter` protocol: the integration interface users implement. Three methods: `evaluate()`, `make_reflective_dataset()`, `propose_new_texts()`. Generic over `DataInst`, `Trajectory`, `RolloutOutput`.
- **`result.py`** — `GEPAResult`: optimization output container.
- **`callbacks.py`** — Event callbacks for the optimization lifecycle.

### Proposers (`src/gepa/proposer/`)

- **`reflective_mutation/`** — Core innovation: LLM reads execution traces, performs credit assignment, proposes improved prompts via a meta-prompt.
- **`merge.py`** — System-aware crossover: combines best modules from two candidates with different lineages.

### Strategies (`src/gepa/strategies/`)

- **`candidate_selector.py`** — How to pick which candidate to evolve next: `ParetoCandidateSelector` (default), `CurrentBestCandidateSelector`, `EpsilonGreedyCandidateSelector`, `TopKParetoCandidateSelector`.
- **`component_selector.py`** — How to pick which module to mutate: `RoundRobinReflectionComponentSelector`, `AllReflectionComponentSelector`.
- **`batch_sampler.py`** — How to sample training batches.
- **`eval_policy.py`** — When/how to evaluate new candidates.

### Adapters (`src/gepa/adapters/`)

Pre-built integrations: `default_adapter` (single-turn LLM), `dspy_adapter`, `dspy_full_program_adapter`, `generic_rag_adapter`, `mcp_adapter`, `terminal_bench_adapter`, `anymaths_adapter`, `optimize_anything_adapter`. Each adapter implements `GEPAAdapter`.

### Data Flow

`Candidate` is `dict[str, str]` mapping component names to their text. The engine loop: select candidate from pool (Pareto) → pick module (round-robin) → run on minibatch → gather traces + feedback → LLM reflects and proposes new text → evaluate → if improved, add to pool and evaluate on full validation set.

## Dependencies

Core has zero required dependencies. Optional extras: `full` (litellm, datasets, mlflow, wandb, tqdm, cloudpickle), `dspy`, `test`, `dev`, `build`, `gskill`.
