# Prompt Optimization in LLM Systems: A Practical Survey

**Status:** Draft  
**Author:** LangChain  
**Goal:** Survey approaches to prompt optimization and outline the workflow we recommend for improving prompts in real-world agent systems.

---

# Overview

Prompt optimization has become a core technique for improving LLM system performance without changing model weights. Instead of training, developers modify instructions, examples, constraints, and tool interfaces to improve outcomes on specific tasks.

Recent work spans search-based approaches, reflection loops, evolutionary methods, and programmatic frameworks (e.g., DSPy) as well as gradient-inspired methods (e.g., GEPA). At the same time, production systems—especially agentic systems—introduce ambiguity, tool interactions, and large prompt surface area that makes applying these techniques meaningfully harder than typical benchmark settings.

This document has three parts:

1. **What is prompt optimization + why it’s hard**
2. **Common techniques (survey)**
3. **What we recommend now (LangChain workflow)**

It also includes TODOs for experiments and for turning this into a simple repo + customer-facing post.

---

# 1) What is Prompt Optimization + Why It’s Hard

## What is prompt optimization?

Prompt optimization is the systematic process of modifying prompts to improve task performance under an evaluation metric measured on representative data.

Prompts can include:

- **system prompts**
- **task instructions**
- **tool descriptions**
- **few-shot examples / demonstrations**
- **formatting constraints**
- **reasoning instructions / policies**

Prompt optimization therefore operates over a large and heterogeneous space of text “parameters.”

## Why it’s hard

### Large prompt surface area
Modern agent systems contain many prompt components (system prompt, tool descriptions, step prompts, output schemas, instructions embedded in harness code). These interact, producing a large search space and making attribution difficult: a win may come from *where* the model saw a constraint, not only *what* the constraint was.

### System prompt as a strong lever (and a risky one)
Empirically, the **system prompt is often the strongest single lever** because it sets global policy and behavior across tasks (tool discipline, output format discipline, safety posture, refusal style, etc.). Small system prompt changes can produce large shifts—and large regressions—so system prompt work benefits from tight evaluation and incremental changes.

### Non-linear interactions in agent systems
Even when optimizing “just a prompt,” behavior interacts with:

- retrieval context and doc selection
- tool availability and tool outputs
- chain/agent orchestration (planning, retries, stopping)
- memory and scratchpad conventions

A prompt that improves isolated behavior can behave differently once integrated into the full system.

### Evaluation sensitivity (and reward hacking)
Prompt optimization is only as good as its evaluator(s). Metrics can be:

- noisy
- incomplete
- easy to game (models optimize for the judge, not the task)

This is why evaluation design becomes the central object in most practical optimization workflows.

### Research-to-practice gap: optimizers vs ambiguous development settings
Many prompt optimization methods are developed and tested on **narrow, well-specified tasks** (classification, short transformations, single-step QA) with clear ground truth. Real-world agent development is often ambiguous:

- goals are multi-objective (correctness, safety, UX, latency, cost)
- supervision is incomplete
- tasks require multi-step tool use
- “correct” is contextual

This mismatch often explains why methods that look strong on benchmark tasks can be less plug-and-play in agentic systems.

---

# 2) Common Techniques (Survey)

Most approaches can be expressed as variations on:
propose candidate prompt
evaluate candidate
retain best candidates
repeat


Below is a practical taxonomy that matches what we see in research and in production engineering.

## Manual iterative refinement (baseline)
Still the most common and often the most reliable in practice:

1. collect failures (via evals + traces)
2. categorize failure modes (tool misuse, schema errors, missing constraints, hallucinations, refusal misfires)
3. propose targeted prompt edits (one hypothesis at a time)
4. re-evaluate and regress

This is also the workflow automated systems often attempt to mechanize.

## Generate-and-rank / prompt search
Automatically propose prompt candidates and score them on a dataset:

- generate N variants (rewrite instructions, reorder constraints, add examples)
- evaluate on a validation set
- keep best (or best K) and iterate

This is black-box optimization over text parameters and often works best when:
- evaluation is stable
- the prompt surface is narrow
- the task is well-defined

## Reflection-based optimization (critique → revise)
Use a model to critique failures and propose revisions:


Conceptually: use critique text as a “directional signal” toward better prompts.

This can be implemented as:
- “prompt doctor” meta-prompts
- self-critique + revision loops
- optimizer agents (see below)

## Evolutionary methods (population-based search)
Treat prompts (or programs) as a population and optimize via mutation + selection.

A concrete reference implementation is **Imbue’s Darwinian Evolver**, which maintains a population of candidates, mutates them, evaluates fitness, and selects strong candidates for subsequent generations:
- https://github.com/imbue-ai/darwinian_evolver/

Evolutionary methods can be appealing when:
- the evaluator is noisy (selection can smooth variance)
- diversity matters (avoid local minima)
- you want robust exploration strategies

## Programmatic prompt optimization (DSPy)
DSPy frames prompt optimization as **compiling** a program-like LLM pipeline against a dataset + metric. Rather than optimizing a single prompt, you optimize prompts/demonstrations across a pipeline of modules (e.g., reasoning, retrieval, extraction). This reframes prompt work as *program optimization* with a metric in the loop.

(Reference you can cite: DSPy paper: https://arxiv.org/abs/2310.03714)

Key idea to borrow for customers: *treat prompts and demonstrations as parameters, and treat your evaluation function as the objective.*

## Gradient-inspired methods (GEPA)
Gradient-inspired prompt optimization methods attempt to create “update directions” for prompts analogous to gradient descent in ML. GEPA is a commonly cited representative: evaluate prompt → generate feedback signal → update prompt. In practice, these methods still depend heavily on evaluation stability and problem definition, and their behavior under ambiguous agent tasks is an active area.

(TODO: add the canonical GEPA reference link/citation you want to use.)

---

# 3) What We Recommend Now (LangChain workflow)

This section is intentionally concrete: the goal is a workflow customers can follow and a small repo that demonstrates it.

## 3.1 A decision framework: when prompt optimization is likely to work

Use this as a go/no-go checklist.

### Prerequisites (strongly recommended)
- **A reliable eval metric and rubric**  
  If your metric doesn’t correlate with real success, optimization will optimize the wrong thing.
- **Stable task definition**  
  Classification-ish tasks (routing, tool selection, schema correctness) tend to be easier than open-ended generation.
- **Narrow prompt surface**  
  If the behavior is dominated by retrieval or tool reliability, prompt optimization may have limited effect.
- **Strong labeled dataset**  
  Labels/rubrics must be consistent and representative of production.
- **Enough labeled examples to generate signal**  
  Heuristics:
  - 30–50 for exploratory iteration
  - 100–300 for stable signal and reduced stochastic variance
- **Stable distribution (or a refresh plan)**  
  If inputs drift weekly, the “best prompt” won’t stay best.

If these prerequisites are missing, the highest-ROI step is usually: **improve evals + dataset first**.

## 3.2 Problem decomposition: isolate what you’re optimizing

To keep attribution clean, isolate one “surface” per experiment:

- **System prompt** (global policies, tool discipline, output discipline)
- **Tool descriptions** (tool selection + argument formatting + usage policies)
- **Skill prompts** (capability-specific prompts like extraction, routing, critique)
- **Chain step prompts** (step-level instructions)
- **Few-shot examples** (demonstration sets, counterexamples)

### One problem at a time
A practical experimental pattern:

> fix one issue → validate on val → run regressions → move to the next issue

Avoid changing skills + system prompt + tool descriptions in the same run. You’ll win sometimes, but you won’t know why, and you’ll struggle to reproduce/maintain the improvement.

## 3.3 Start with evaluation (best practice #1)

If you do nothing else, do this:

- define a rubric (what is “good”)
- create train/val/test splits
- implement at least one deterministic check where possible (JSON parse, schema validity, tool args validity)
- run a baseline and save results

This is the foundation for any method—manual, search-based, evolutionary, programmatic, or gradient-inspired.

## 3.4 Use an optimizer agent (not just a blind loop)

We recommend treating prompt optimization as an **agentic analysis task**:

### Optimizer agent responsibilities
#EXAMPLE CLAUDE CODE can be fine here
1. **Read LangSmith evaluation outputs and traces**
   - collect failing examples
   - cluster failures (by rubric tag, tool error type, schema failure type)
2. **Form hypotheses**
   - “constraint is missing or buried”
   - “tool description ambiguous”
   - “system policy conflicts with step prompt”
3. **Propose targeted prompt edits**
   - minimal diffs, one hypothesis per candidate
   - optionally propose 2–5 variants to preserve diversity
4. **Run controlled experiments**
   - evaluate candidates on val under budget constraints
   - track regressions (don’t just maximize one score)
5. **Summarize outcomes**
   - what improved, what regressed, and what failure cluster remains

This “agent interprets eval results” approach makes optimization more robust in ambiguous settings, because it anchors changes to observed failure modes rather than purely optimizing a single scalar objective.

## 3.5 Cost control (smart, but not overfit to one tool)

Cost control is not just “cap iterations.” In practice it’s a combination of:

- **budgets** (max candidates per iteration, max iterations, max spend)
- **staged evaluation** (cheap filters first, full eval later)
- **adaptive sampling** (focus on failure clusters; don’t re-evaluate easy cases every time)
- **tracking and reproducibility** (log prompt diffs, datasets, evaluator versions)
- **stopping rules** (stop when improvements plateau or regressions rise)

The exact mechanism can vary, but cost visibility and staged evaluation are the two ideas customers adopt fastest.

## 3.6 Practical expectations (what this is good for)
What we generally see work well:

- **0→1 prompt creation** in a scoped domain (turning informal requirements into a robust prompt)
- **formalizing repeated errors** into constraints and tests
- improving **classification-ish** behaviors (routing, tool selection, schema correctness)

Where it often struggles:

- squeezing large gains from an already well-crafted prompt (diminishing returns)
- tasks with weak supervision and ambiguous success criteria
- settings where the real bottleneck is retrieval/tooling rather than instruction-following

EXPERIMENT SECTION: 
Our recommendation is based on two practical principles:

1. **evaluation-driven iteration**
2. **optimize one prompt surface at a time**

---

# Experiment 1 — System Prompt Optimization (Subagent Routing)

## Goal

Measure how system prompt changes affect **which subagent the system calls**.

Example:

User question → system must choose:

- retrieval agent
- coding agent
- browsing agent

Incorrect routing is a common real-world failure.

---

## Dataset

Construct a dataset of routing examples.

Each example:


{
"query": "...",
"expected_agent": "retrieval_agent"
}


Target size:

- train: ~50
- validation: ~50
- test: ~100

---

## Evaluation metric

Primary metric:

**routing accuracy**


predicted_agent == expected_agent


Secondary metrics:

- abstention correctness
- fallback usage

---

## Hypothesis

System prompt clarity significantly affects routing accuracy.

---

## Experiment steps

1. Baseline system prompt
2. Modify routing instructions
3. Evaluate on validation
4. Measure regressions

---

## Expected outputs


baseline accuracy
optimized accuracy
confusion matrix across agents


---

# Experiment 2 — Tool Description Optimization

## Goal

Improve correctness of tool usage for a specific failing tool.

Example:

A tool frequently fails due to:

- incorrect arguments
- incomplete parameters
- misuse of tool purpose

---

## Dataset

Examples requiring the tool.


{
"query": "...",
"expected_tool": "search_tool",
"expected_args": {...}
}


---

## Evaluation

Metrics:

1. correct tool selected
2. arguments valid
3. tool called at correct step

---

## Hypothesis

Clarifying tool descriptions and adding examples improves tool call correctness.

---

## Experiment steps

1. Baseline tool description
2. Add argument examples
3. Add usage constraints
4. Evaluate changes

---

# Optimizer Agent Workflow

Prompt optimization can be implemented as an **optimizer agent**.

Responsibilities:

1. Read LangSmith evaluation results
2. Identify failure clusters
3. Propose targeted prompt edits
4. Run experiments
5. Produce prompt diffs and metric changes

---

# Best Practices

## Start with strong evaluation

Define:

- rubric
- dataset
- metric
- regression tests

---

## Optimize one surface at a time

Examples:

- system prompt
- tool description
- chain step prompt

Avoid modifying multiple surfaces simultaneously.

---

## Inspect failures before optimizing

Most improvements come from addressing **specific failure clusters**.

---

## Manage experiment cost

Use:

- candidate caps
- iteration budgets
- staged evaluation

---

# Repository Structure


prompt-optimization-survey/

data/
routing_train.jsonl
routing_val.jsonl
routing_test.jsonl

tool_train.jsonl
tool_val.jsonl
tool_test.jsonl

prompts/
system_prompt.md
tool_descriptions.md

src/
run_eval.py
run_routing_eval.py
run_tool_eval.py
optimizer_agent.py
run_optimize.py

results/
routing_baseline.json
routing_optimized.json
tool_baseline.json
tool_optimized.json


---

# Coding TODOs

## Dataset creation

- [ ] Create routing dataset
- [ ] Create tool usage dataset
- [ ] Ensure examples represent real production failures

---

## Evaluation

- [ ] Implement routing accuracy evaluator
- [ ] Implement tool usage evaluator
- [ ] Add confusion matrix for routing
- [ ] Add argument validation for tool calls

---

## Experiment runner

- [ ] Baseline evaluation script
- [ ] Prompt version tracking
- [ ] Regression detection

---

## Optimizer agent

- [ ] Read LangSmith eval results
- [ ] Cluster failure cases
- [ ] Generate prompt edits
- [ ] Re-run evaluation automatically

---

# TODO (Blog / Writeup)

- [ ] Add routing experiment results
- [ ] Add tool description experiment results
- [ ] Add prompt diff examples
- [ ] Add graphs showing performance improvement

---

# References

LangChain Prompt Optimization Survey  
https://blog.langchain.com/exploring-prompt-optimization/

LangChain Promptim  
https://blog.langchain.com/promptim/

Darwinian Evolver  
https://github.com/imbue-ai/darwinian_evolver/

DSPy  
https://arxiv.org/abs/2310.03714