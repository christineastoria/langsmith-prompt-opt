# Prompt Optimization Demo

A working example of automated prompt optimization for a multi-agent LLM system. The core of this repo is a **LangGraph optimizer agent** that takes a baseline system prompt, pulls evaluation results from LangSmith, and rewrites the prompt to fix the failure patterns it finds.

The demo system is a shopping concierge that routes user requests across 6 specialized agents. But the optimizer is general — the same pipeline applies to any system prompt you can evaluate.

## How it works

```
baseline prompt
      +
eval results (from LangSmith)
      |
      v
pull_failures → analyze → reflect → generate → review → save
                                                          |
                                                          v
                                                  optimized.md
```

1. **pull_failures** — fetches all runs from a LangSmith baseline eval experiment with full context: inputs, reference outputs, actual agent outputs, agents called, per-metric scores, and LLM judge reasoning
2. **analyze** *(Sonnet)* — identifies failure patterns per metric across task types
3. **reflect** *(Opus)* — filters analysis down to changes that generalize vs. changes that would overfit to training examples
4. **generate** *(Opus)* — rewrites the prompt incorporating only the validated improvements
5. **review** *(Opus)* — strips anything that hardcodes training-specific details
6. **save** — writes `prompts/optimized.md`

The reflection node is the key mechanism. Without it, the optimizer tends to patch individual examples rather than fix the underlying prompt logic.

## Results

One optimizer pass on the shopping concierge system prompt:

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| Task completeness *(LLM judge)* | 49% | 62% | +28% |
| Critical agents called *(code)* | 58% | 93% | +60% |
| Sequence respected *(code)* | 59% | 91% | +55% |

Verified across 6 repeated runs — the gap between baseline and optimized is 10–14× larger than the noise on each metric.

## Repo structure

```
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
```

## Dataset

109 examples across 7 task types: `action_with_prerequisite`, `return_eligibility`, `product_comparison`, `price_assessment`, `discovery_with_validation`, `sizing_with_context`, `warranty_lookup`. Stratified train/val/test split so each split has proportional task type coverage. Separate edge case file for progressive stress testing.

Each example has:
- `inputs` — user query + user email
- `outputs` — task type, required info, expected agents, critical agents (`cannot_complete_without`), sequencing requirements

## Evaluators

Three metrics:

- **`task_completeness`** — LLM-as-judge grounded in each example's `required_info` field. Scores 0 / 0.5 / 1.0.
- **`critical_agents_called`** — deterministic. Were all agents in `cannot_complete_without` called?
- **`sequence_respected`** — deterministic. For tasks with ordering requirements, did prerequisite agents fire before action agents?

## Setup

```bash
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

# Evaluate the optimized prompt
uv run python src/eval/run_eval.py --prompt optimized --split val --suffix="-v1"
```

## LangGraph Studio

```bash
uv run langgraph dev
```

Opens the optimizer graph in LangGraph Studio for step-through debugging. Pass `{"experiment_name": "your-baseline-train-experiment-id"}` as the initial state, or leave it empty to auto-detect the latest baseline-train run.

## Environment variables

```
ANTHROPIC_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your-project-name
OPENAI_API_KEY=        # used for web search tools in the demo agent
TAVILY_API_KEY=        # optional, for web_research_agent
```
