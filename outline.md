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

Show that system prompt improvements measurably improve **which subagents a multi-agent concierge calls, in what order, and whether it hallucinates**.

The demo system is a personal shopping concierge built with [Deep Agents](https://github.com/langchain-ai/deepagents) with six specialized subagents:

| Subagent | Data it owns |
|---|---|
| `product_catalog_agent` | order history, delivery status, prices, SKUs, internal reviews |
| `policy_and_sizing_agent` | return windows, brand-specific rules, sizing guides, non-returnable items |
| `cart_and_orders_agent` | executes actions — add to cart, initiate return, wishlist (needs real SKU/order_id) |
| `product_comparison_agent` | structures comparisons once catalog data is available |
| `product_discovery_agent` | searches catalog by category or criteria |
| `web_research_agent` | external prices, reviews, competitor data |

Baseline system prompt: `"You are a personal shopping concierge. Help customers find products, answer questions, and manage their orders."` — one sentence, no routing guidance.

---

## Dataset

90 handcrafted examples across 7 task types, split 54 train / 18 val / 18 test.

**Task types:**

| Type | Count | What it tests |
|---|---|---|
| `price_assessment` | 10 | catalog price history + competitor comparison |
| `return_eligibility` | 15 | order date from catalog + policy rules |
| `warranty_lookup` | 8 | purchase date + brand warranty terms |
| `product_comparison` | 12 | catalog specs → comparison agent structures analysis |
| `discovery_with_validation` | 10 | catalog candidates + external review validation |
| `sizing_with_context` | 12 | policy-specific sizing guidance (not general knowledge) |
| `action_with_prerequisite` | 23 | catalog lookup → (eligibility check) → cart action |

**Three dataset splits:**
- **Train** — what the optimizer sees. Used to identify failure patterns and generate a better prompt.
- **Val** — development benchmark. Used after each optimization to measure improvement. Never shown to the optimizer.
- **Test** — locked until the end. Evaluated exactly once to report the final honest result.

**Edge cases** (`shopping-concierge-routing-edge`, 10 examples) live in a separate dataset, added progressively as optimization advances. These test harder reasoning: policy exceptions, cross-user references, ambiguous product references, price match edge cases. They are never part of train/val/test splits.

### Key dataset design decisions

**`cannot_complete_without`** is tight — only agents that hold ground truth data without which a correct answer requires hallucination. If an agent is helpful but the LLM could answer correctly from general knowledge, it is not listed here.

**`requires_sequencing: True`** only when the cart action literally needs the output of the catalog call (real order_id or SKU). The cart agent cannot hallucinate these. Policy can always be called in parallel with catalog — it is never in `expected_sequence`.

**`expected_agents`** = exactly `cannot_complete_without` ∪ agents in `expected_sequence`. No others. If an agent is not required and not sequenced, it is not expected.

**No cart agent when the answer is "no"**: if return eligibility fails (non-returnable, out of window, not yet delivered, already returned), `cart_and_orders_agent` is not in expected_agents or sequence. The correct behavior is to stop at the eligibility check and inform the customer.

**`required_info` contains only verifiable facts** — specific prices, delivery dates, SKUs, policy rules. For discovery examples that require web research, the web result is deliberately excluded from `required_info` (it is inherently variable at runtime). The web agent call is enforced by `critical_agents_called` instead.

---

## Evaluators

Three evaluators, each catching different failure modes:

### 1. `task_completeness` (LLM-as-judge, Claude Haiku)
Did the final answer contain the `required_info`? Catches hallucination: the agent may call the right agents but still produce a wrong answer. The judge compares the agent's output to the list of specific facts the answer must include.

### 2. `critical_agents_called` (code)
Were the agents in `cannot_complete_without` actually called? These are the agents that hold ground truth. Without them, any correct-sounding answer is hallucinated. This is a deterministic check against the trajectory — no LLM involved.

### 3. `sequence_respected` (code)
For `action_with_prerequisite` tasks: did `product_catalog_agent` appear before `cart_and_orders_agent` in the trajectory? The cart agent needs a real SKU or order_id — it cannot invent one. This enforces the one hard ordering constraint. It is only computed for the subset of examples where `requires_sequencing: True` (7–8 of 18 val examples).

**Why all three together:**
- An agent can call the right agents and still hallucinate → `task_completeness` catches this
- An agent can give a plausible answer without calling the critical agents → `critical_agents_called` catches this
- An agent can call all the right agents in the wrong order → `sequence_respected` catches this

---

## Results

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.50 | **0.61** | +0.11 |
| `critical_agents_called` | 0.64 | **0.94** | +0.30 |
| `sequence_respected` | 0.29 | **0.57** | +0.28 |

The biggest gain is `critical_agents_called` (+0.30): the optimized prompt explicitly tells the orchestrator which agent owns which data, so it stops answering from general knowledge on questions that require a live lookup. The minimum score went from 0 to 0.5 — the prompt is no longer completely missing on any example.

---

# Optimizer Agent Pipeline

The optimizer is a **LangGraph StateGraph** — a 5-node pipeline where each node has a defined responsibility and the reflection step runs on a stronger model.

```
pull_failures → analyze → reflect → generate → review → save
```

| Node | Model | What it does |
|---|---|---|
| `pull_failures` | — | Queries LangSmith for all baseline-train run scores via the Client API |
| `analyze` | claude-sonnet-4-6 | Identifies failure patterns across task types |
| `reflect` | **claude-opus-4-6** | Separates general patterns from example-specific overfitting risks |
| `generate` | **claude-opus-4-6** | Writes the optimized prompt from general patterns only |
| `review` | **claude-opus-4-6** | Final pass: strips anything that hardcodes training-specific details |
| `save` | — | Writes `prompts/optimized.md` |

**The reflection node is structurally separate** — not just another message in the same chain. Its job is to be a strict filter: for each insight from the analysis, it decides "does this generalize to unseen queries of the same type, or is it just patching a specific example?" Only insights that pass this filter are passed to the generate node.

**Human prior**: the optimizer is given a brief routing principles document (what makes routing prompts effective in general) as context — not a specific example prompt to copy. This anchors the optimizer toward known-good patterns without creating a risk of copying the prior.

**Anti-overfit mechanisms:**
1. The reflect node explicitly discards example-specific insights
2. The generate node is instructed not to hardcode product names, users, SKUs, or prices
3. The review node does a final check before saving

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

```
prompt-opt-demo/
├── data/
│   ├── shop.db               # SQLite: products, orders, users, reviews (generated by setup_db.py)
│   ├── chroma/               # vector store for product discovery
│   └── eval/
│       ├── train.jsonl       # 54 examples
│       ├── val.jsonl         # 18 examples
│       └── test.jsonl        # 18 examples (locked until final eval)
│
├── prompts/
│   ├── baseline.md           # starting prompt (1 sentence)
│   └── optimized.md          # optimizer output
│
└── src/
    ├── agents/               # Deep Agents orchestrator + 6 subagents
    ├── tools/                # DB tools, web search (Tavily), catalog tools
    ├── eval/
    │   ├── dataset.py        # 90 core examples, upload to LangSmith train/val/test
    │   ├── dataset_edge.py   # 10 edge cases, separate dataset, added progressively
    │   ├── evaluator.py      # task_completeness, critical_agents_called, sequence_respected
    │   ├── run_function.py   # wraps agent for LangSmith evaluate(), captures trajectory
    │   └── run_eval.py       # entry point: --prompt baseline|optimized --split train|val|test
    └── optimizer/
        └── run_optimizer.py  # LangGraph 5-node optimizer pipeline
```


---

# Status & TODOs

## Done
- [x] Shopping concierge with 6 subagents (Deep Agents)
- [x] 90-example dataset across 7 task types, uploaded to LangSmith (train/val/test)
- [x] 10 edge case examples in separate dataset
- [x] Three evaluators: task_completeness (LLM judge), critical_agents_called, sequence_respected
- [x] Trajectory capture from Deep Agents stream (correct chunk structure)
- [x] Baseline eval on train + val
- [x] LangGraph optimizer pipeline (5 nodes, reflection on opus)
- [x] Optimized prompt generated and saved
- [x] Optimized eval on val: 0.50→0.61 / 0.64→0.94 / 0.29→0.57

## Remaining
- [ ] Run optimized on train (sanity check — should not be lower than val, would flag overfitting)
- [ ] Second optimization round (if val still has room)
- [ ] Final eval on test set (run exactly once)
- [ ] Prompt diff write-up: what changed and why
- [ ] Add graphs (baseline vs optimized across all three metrics)
- [ ] Edge case eval: run both prompts against shopping-concierge-routing-edge

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