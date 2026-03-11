# Prompt Optimization Demo

A working example of automated prompt optimization for a multi-agent LLM system.

This repository demonstrates how an optimizer agent can analyze evaluation results and rewrite a system prompt to fix failure patterns in an agent system.

The demo system is a shopping concierge that routes user requests across 6 specialized agents. The optimizer is general — the same pipeline applies to any system prompt you can evaluate.

---

# High-level learnings

This project showed that prompt optimization can work well for real agent systems — but only when the surrounding evaluation and dataset setup is strong.

The biggest gains did **not** come from blindly searching prompt variants with an auto-optimizer. They came from:

- high-quality evals
- a representative labeled dataset
- an optimizer that analyzes real successes and failures
- explicit filtering against overfitting

Prompt optimization worked especially well here because the target behavior was **structured routing**, not open-ended generation.

Routing, tool selection, sequencing, and instruction-following are much more stable optimization targets than subjective generation quality.

Key takeaways:

- **Evaluation first.** If the metric rewards the wrong thing, the optimizer will improve the wrong behavior.
- **Dataset quality matters more than dataset size.**
- **Deterministic checks remove noise** and make results interpretable.
- **Evaluators should check task requirements, not specific trajectories.**
- **Reflection steps prevent overfitting** to training examples.

---

# Results

One automated prompt rewrite improved agent performance by roughly **~50% on average**.

The gains held:

- across train and validation sets
- across 6 repeated runs
- and after introducing adversarial edge cases

---

## Overall: Before vs After

*(validation set, averaged across 6 runs)*

### Task Completeness

    Before  ██████████░░░░░░░░░░░  49%
    After   █████████████░░░░░░░░  62%   +28%

### Critical Agents Called

    Before  ████████████░░░░░░░░░  58%
    After   ███████████████████░░  93%   +60%

### Sequence Respected

    Before  ████████████░░░░░░░░░  59%
    After   ███████████████████░░  91%   +55%

Verified across **6 repeated runs** — the gap between baseline and optimized is **10–14× larger than the noise** on each metric.

---

# Train vs Val: Did the optimizer overfit?

A common failure mode in prompt optimization is **overfitting** — prompts improve on training examples but degrade on unseen ones.

That did **not** happen here.

### Task Completeness

    Train — Before  ███████████░░░░░░░░░░  54%
    Train — After   █████████████░░░░░░░░  63%   +17%

    Val   — Before  ██████████░░░░░░░░░░░  49%
    Val   — After   █████████████░░░░░░░░  62%   +28%

### Critical Agents Called

    Train — Before  ██████████████░░░░░░░  68%
    Train — After   ██████████████████░░░  91%   +34%

    Val   — Before  ████████████░░░░░░░░░  58%
    Val   — After   ███████████████████░░  93%   +60%

### Sequence Respected

    Train — Before  ███████████████░░░░░░  76%
    Train — After   ███████████████████░░  91%   +20%

    Val   — Before  ████████████░░░░░░░░░  59%
    Val   — After   ███████████████████░░  91%   +55%

Validation gains **match or exceed training gains on every metric**, suggesting the improvements generalize.

---

# Repeated Runs: Stability

LLMs are stochastic, so the same prompt can produce slightly different scores.

We ran each version **6 times** to measure variance.

### Critical Agents Called

    Baseline  •  •  •  •  •  •      range: 0.53–0.62
              avg: 0.578

    Optimized      •••••••          range: 0.88–0.94
                   avg: 0.926

### Sequence Respected

    Baseline  •••••  •              range: 0.54–0.67
              avg: 0.585

    Optimized      ••••••           range: 0.90–0.93
                   avg: 0.911

### Task Completeness

    Baseline  ••••• •               range: 0.41–0.53
              avg: 0.485

    Optimized      •• •• •          range: 0.56–0.71
                   avg: 0.623

Across all metrics, **baseline and optimized ranges do not overlap**.

Even the worst optimized run beats the best baseline run on routing metrics.

---

# How it works

The core of this repo is a **LangGraph optimizer agent** that rewrites system prompts based on evaluation results.

<img width="753" height="754" alt="image" src="https://github.com/user-attachments/assets/322b0be4-6219-46f8-95be-21d68c965790" />

### Optimizer pipeline

1. **pull_results** — fetches all runs from a LangSmith baseline eval experiment with full context: inputs, reference outputs, actual outputs, agents called, per-metric scores, and LLM judge reasoning

2. **analyze** *(Sonnet)* — identifies failure patterns per metric across task types

3. **reflect** *(Opus)* — filters analysis down to changes that generalize vs. changes that would overfit to training examples

4. **generate** *(Opus)* — rewrites the prompt incorporating only the validated improvements

5. **review** *(Opus)* — strips anything that hardcodes training-specific details

6. **save** — writes `prompts/optimized.md`

The **reflection node** is the key mechanism. Without it, the optimizer tends to patch individual examples rather than fix the underlying prompt logic.

---

# Repo structure

    src/
      optimizer/
        run_optimizer.py     # the LangGraph optimizer agent
      agents/
        shopping_assistant.py  # the demo multi-agent system
        subagents.py
      eval/
        evaluator.py         # 3 evaluators (1 LLM judge + 2 code)
        dataset.py           # core dataset builder
        dataset_edge.py      # 15 adversarial edge cases
        dataset_with_edge.py # combined train/val with edge cases
        run_eval.py          # runs an eval experiment on LangSmith
        run_function.py      # wraps the agent for evaluate()
      tools/                 # SQL, web, search, action tools

    prompts/
      baseline.md            # starting system prompt
      optimized.md           # output of the optimizer

    data/eval/
      train.jsonl            # 54 examples
      val.jsonl              # 17 examples
      test.jsonl             # 23 examples
      edge.jsonl             # 15 adversarial edge cases

    langgraph.json           # open optimizer in LangGraph Studio

---

# Dataset

109 examples across 7 task types:

- action_with_prerequisite
- return_eligibility
- product_comparison
- price_assessment
- discovery_with_validation
- sizing_with_context
- warranty_lookup

Stratified train/val/test split so each split has proportional task type coverage.

Separate edge case file for progressive stress testing.

Each example contains:

- **inputs**
  - user query
  - user email

- **outputs**
  - task type
  - required info
  - expected agents
  - critical agents (`cannot_complete_without`)
  - sequencing requirements

---

# Evaluators

Three metrics measure routing correctness and answer quality.

### task_completeness

LLM-as-judge metric grounded in each example's `required_info`.

Scores:

    0.0   missing required info
    0.5   partially correct
    1.0   fully complete

### critical_agents_called

Deterministic check: were all agents in `cannot_complete_without` called?

### sequence_respected

Deterministic ordering check.

Example:

    order lookup → eligibility check → return action

Required ordering contributes **80% of the score**.  
Optional agents contribute **20% bonus**.

---

# Setup

    # Install dependencies
    uv sync

    # Copy and fill in your keys
    cp .env.example .env

    # Upload datasets to LangSmith
    uv run python src/eval/dataset.py --suffix="-v1"

    # Run baseline eval
    uv run python src/eval/run_eval.py --prompt baseline --split train --suffix="-v1"

    # Run optimizer
    uv run python src/optimizer/run_optimizer.py

    # Evaluate optimized prompt
    uv run python src/eval/run_eval.py --prompt optimized --split val --suffix="-v1"

---

# LangGraph Studio

    uv run langgraph dev

Opens the optimizer graph in LangGraph Studio.

Pass:

    {"experiment_name": "your-baseline-train-experiment-id"}

as the initial state, or leave it empty to auto-detect the latest baseline run.

---

# Environment variables

    ANTHROPIC_API_KEY=
    LANGSMITH_API_KEY=
    LANGSMITH_TRACING=true
    LANGSMITH_PROJECT=your-project-name
    OPENAI_API_KEY=
    TAVILY_API_KEY=
