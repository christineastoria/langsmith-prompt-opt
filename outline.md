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

### Prefer deterministic and specific evals

**The more deterministic and specific your evaluators, the better.** This is not just a quality preference — it directly affects whether your optimizer signal is real.

LLM-as-judge evaluators are useful for open-ended outputs but have two practical problems: they introduce noise (same response scores differently across runs), and they can be gamed (the optimized prompt learns to satisfy the judge rather than solve the task). Every LLM judge you can replace with a deterministic code check is a win.

For specific evals:
- `required_info` should contain verifiable facts (exact prices, dates, SKUs) — not vague descriptions like “external review sentiment” that any response can satisfy
- `cannot_complete_without` should be tight: only agents that are genuine ground-truth holders, not every agent you'd like to see called
- Trajectory checks (did agent X appear before agent Y?) are fully deterministic and carry strong signal about whether the agent reasoned correctly

In practice: use LLM judges for things that are genuinely hard to specify (response quality, tone, completeness of an open-ended answer) and deterministic checks for everything structural (routing, ordering, tool selection, identifier correctness). When in doubt, ask: “could I write a unit test for this?” If yes, do it.

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

**Edge cases** (`shopping-concierge-routing-edge`, 15 examples) live in a separate dataset, added progressively as optimization advances. They are never part of train/val/test splits.

### Key dataset design decisions

**`cannot_complete_without`** is tight — only agents that hold ground truth data without which a correct answer requires hallucination. If an agent is helpful but the LLM could answer correctly from general knowledge, it is not listed here.

**`requires_sequencing: True`** only when the cart action literally needs the output of the catalog call (real order_id or SKU). The cart agent cannot hallucinate these. Policy can always be called in parallel with catalog — it is never in `expected_sequence`.

**`expected_agents`** = exactly `cannot_complete_without` ∪ agents in `expected_sequence`. No others. If an agent is not required and not sequenced, it is not expected.

**No cart agent when the answer is "no"**: if return eligibility fails (non-returnable, out of window, not yet delivered, already returned), `cart_and_orders_agent` is not in expected_agents or sequence. The correct behavior is to stop at the eligibility check and inform the customer.

**`required_info` contains only verifiable facts** — specific prices, delivery dates, SKUs, policy rules. For discovery examples that require web research, the web result is deliberately excluded from `required_info` (it is inherently variable at runtime). The web agent call is enforced by `critical_agents_called` instead.

### Human judgment is load-bearing in your eval dataset

The most important investment in any eval-driven optimization workflow is in the dataset itself — and that investment is human time, not compute. Get this right before you run a single eval.

It is tempting to generate eval examples automatically: use an LLM to produce query/expected-output pairs, run them through the system, and call the dataset "done." This gets you to a benchmark quickly, but it has a fundamental problem: **LLM-generated examples encode the LLM's assumptions about what correct behavior looks like, not your system's actual requirements.** If those assumptions are wrong, your optimizer will optimize toward the wrong target.

Every label in a dataset is a claim about what correct system behavior looks like. Those claims need to be reviewed by someone who understands both the task requirements and the system architecture — before you run optimization, not after. The optimizer learns from your labels. Wrong labels produce wrong prompts, reliably.

**What goes wrong without upfront review.** Here are label errors we caught in this dataset that would have silently degraded results:

- **Over-constrained routing labels.** Early examples included `product_comparison_agent` in `cannot_complete_without` for simple head-to-head queries. The orchestrator can reason from catalog data directly — the comparison agent is only strictly required for structured ranking across many items. Keeping it required created false negatives: the evaluator penalized correct behavior, and the optimizer would have learned to call the comparison agent unnecessarily.
- **Cart agent included on ineligible returns.** Several action examples listed `cart_and_orders_agent` as expected even when the return was ineligible (non-returnable item, outside window, not yet delivered). The correct behavior is to stop at the eligibility check. Labeling the cart agent as expected would teach the optimizer to attempt returns it should refuse.
- **Ambiguous queries pinned to one path.** "Is the Dyson worth buying?" can route through `product_catalog_agent` (check the user's order history) or `product_discovery_agent` (browse catalog). Both are reasonable. Listing only one penalizes the other — labeling a valid behavior as wrong.
- **Ownership assumptions baked into general questions.** "What warranty does the Sony XM6 come with?" was initially labeled as requiring `product_catalog_agent` to confirm the user owns the product. But the query is general — ownership is not the question. The label was projecting an assumption that isn't in the query.

**The recommended workflow for building a golden dataset:**

The best source of examples is your system's actual behavior — not synthetic queries invented from scratch. The recommended pipeline:

1. **Generate candidates from traces or with an agent.** Run your system on a sample of real or representative queries and export the traces. Alternatively, use an LLM agent to generate candidate examples with draft labels. LangSmith makes both easy: you can export traces directly and use the dataset generation tools to structure them.

2. **Route candidates to a human annotation queue.** In LangSmith, this means adding examples to an annotation queue where reviewers can inspect each input, the system's actual output, and the draft expected output side by side.

3. **Have humans correct inputs and outputs before approving.** Reviewers should ask for each example: *Is this query realistic? Is the expected routing actually correct? Are any labels over- or under-constrained? Does the required_info contain verifiable facts or vague descriptions?* Fix what's wrong, reject what can't be fixed cleanly.

4. **Promote reviewed examples to a golden dataset.** Only examples that have been explicitly approved go into train/val/test. The golden dataset is the source of truth for all optimization runs.

This pipeline keeps automation in its lane (generating volume, structuring candidates) while keeping humans in the loop on what "correct" actually means. The upfront cost is real — but it is far cheaper than discovering after several optimization rounds that your evaluator was measuring the wrong thing.

### Edge cases as a progressive stress test

The edge case dataset is not a second val set. It serves a different purpose: **progressive stress testing**.

The three-split train/val/test structure works well when you know your failure modes upfront. In practice, optimization reveals new failure patterns you didn't anticipate when writing the initial dataset. The edge case dataset is the mechanism for handling this — a living test suite that grows alongside the optimizer.

**How it works:**

1. Run baseline eval on train and val to identify failure clusters.
2. Run edge cases to check whether easy wins on train generalize to harder inputs.
3. Analyze per-example failures in the edge set — which patterns is the optimizer not improving on?
4. **Add new edge cases targeting those specific gaps.** Don't add them to train — keep them as a diagnostic layer.
5. In the next optimization round, feed the optimizer failures from both train AND the edge set. More failure signal = more general prompt improvements.
6. Repeat: re-run edge cases after each round to see if the new cases are now handled.

This is analogous to how software engineers add regression tests when they find a bug — you don't just fix the bug, you add a test that would have caught it. Here, you don't just fix the prompt, you add examples that would catch a regression.

**What we discovered and added:**

After round 1 optimization, analysis of `sequence_respected` failures on train showed a specific pattern: the optimizer improved routing to required agents (`critical_agents_called` +0.15) but made little progress on comparison-agent sequencing (+0.06). Digging into per-example scores revealed that `product_comparison_agent` was being skipped entirely on head-to-head comparison queries — even when the correct answer required one product to be identified as the winner before adding it to the cart.

The original edge cases (BATCH 1) covered policy exceptions, timing, cross-user references, and ambiguous intent — all BATCH 1 patterns. None specifically targeted comparison + action sequences.

**BATCH 2 was added to fill that gap:** 5 examples where the agent must look up products, compare them (explicitly or implicitly), and then act on the comparison result. These are harder than train comparisons because:
- The cart action depends strictly on the comparison outcome — the agent cannot add "the winner" without first determining which product that is
- Some include social influence ("my friend says Sony is better") that the agent must not take at face value
- One requires looking up another user's order history before the comparison can happen

**The evaluator had to be updated too.** When we added comparison examples to expected_sequence, we initially made `product_comparison_agent` required — which produced false negatives. The orchestrator *can* answer simple comparison questions using catalog data directly (it's a capable LLM). Only `product_catalog_agent` is strictly required; comparison agent is the right specialized tool but not the only path to a correct answer. The fix: weight `sequence_respected` scoring 80% on required agents and 20% as bonus for optional agents in the sequence. This eliminated the false negatives and gave the optimizer accurate signal on what's actually broken.

**The lesson: your evaluators and your dataset co-evolve.** Writing better examples often reveals evaluator blind spots, and fixing evaluators often reveals dataset gaps. Build in checkpoints to audit both together.

### A note on cross-user examples and auth

Adding cross-user examples surfaced a real production issue: in this system, `user_email` is an LLM-supplied argument to every user-scoped tool (`get_user_orders`, `add_to_cart`, `save_to_wishlist`, `initiate_return`). That means a sufficiently creative prompt could instruct the agent to look up a different user's order history or add items to someone else's cart.

The correct fix is **tool-layer scoping** — capture the authenticated user's email in a closure at session start and inject it into the tool, so the LLM never receives it as a parameter at all:

```python
def make_get_user_orders(user_email: str):
    @tool
    def get_user_orders(status: str = None) -> str:
        """Look up the current user's order history."""
        # user_email is captured from outer scope, not from LLM
        ...
    return get_user_orders
```

This is not a prompt engineering problem — prompt instructions can't reliably prevent a model from misusing parameters it controls. The right fix belongs at the tool layer.

**For the purposes of this demo**, we're leaving `user_email` as an LLM argument and testing whether the optimizer can learn at the prompt layer to avoid unnecessary cross-user calls. The expected behavior for cross-user examples is simple refusal — the agent should decline, not route to any subagent. The tool-layer fix should happen regardless; this is purely about whether prompt optimization can drive better behavior on top of it.

---

## Evaluators

Three evaluators, each catching different failure modes:

### 1. `task_completeness` (LLM-as-judge, Claude Haiku)
Did the final answer contain the `required_info`? Catches hallucination: the agent may call the right agents but still produce a wrong answer. The judge compares the agent's output to the list of specific facts the answer must include.

### 2. `critical_agents_called` (code)
Were the agents in `cannot_complete_without` actually called? These are the agents that hold ground truth. Without them, any correct-sounding answer is hallucinated. This is a deterministic check against the trajectory — no LLM involved.

### 3. `sequence_respected` (code)
For `requires_sequencing: True` examples: did agents in `expected_sequence` appear in the correct order in the trajectory? Only computed for the subset where ordering genuinely matters for correctness.

**Scoring is weighted by whether sequence agents are required or optional:**
- **Required** = agents in both `expected_sequence` AND `cannot_complete_without`. Their ordering drives 80% of the score. Missing or out-of-order required agents score low.
- **Optional** = agents in `expected_sequence` but NOT in `cannot_complete_without`. Calling them is good practice but not a correctness requirement. Contributes the remaining 20% as a bonus.

Example: `product_comparison_agent` is in `expected_sequence` for comparison queries but not in `cannot_complete_without` — the orchestrator can produce a correct answer using catalog data alone. An agent that calls catalog → cart in the right order gets 0.8. Calling comparison agent too gets 1.0.

**Why this matters:** If you treat all sequence agents as equally required, you get false negatives — the evaluator penalizes correct behavior. An agent that skips the comparison agent but produces the right answer from catalog data should not score 0. Over-constrained evaluators make the optimizer fight the metric instead of improving the prompt.

**Why all three together:**
- An agent can call the right agents and still hallucinate → `task_completeness` catches this
- An agent can give a plausible answer without calling the critical agents → `critical_agents_called` catches this
- An agent can call all the right agents in the wrong order → `sequence_respected` catches this

### The more deterministic and specific the evaluators, the better

LLM judges introduce noise: the judge can be inconsistent, can be gamed, and makes optimization harder to interpret. Replace LLM judgment with deterministic checks wherever the ground truth is knowable. In this system:
- `critical_agents_called` and `sequence_respected` are fully deterministic — no LLM involved
- `task_completeness` uses an LLM judge only because "did the answer include specific facts" is hard to check with regex

Even within LLM judges: the more specific the rubric, the more reliable the score. `required_info` lists exact prices, SKUs, and policy rules — not vague descriptions. A judge comparing "did the answer mention $249 for the AirPods Pro" is more consistent than "did the agent give good advice."

---

## Results

### Val set (held out during optimization)

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.50 | **0.61** | +0.11 |
| `critical_agents_called` | 0.64 | **0.94** | +0.30 |
| `sequence_respected` | 0.29 | **0.57** | +0.28 |

### Train set (seen by optimizer)

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.519 | **0.639** | +0.12 |
| `critical_agents_called` | 0.685 | **0.833** | +0.15 |
| `sequence_respected` | 0.222 | **0.278** | +0.06 |

### Reading the findings

**`critical_agents_called` is the clearest win.** The optimized prompt explicitly describes what data each subagent owns, so the orchestrator stops answering from general knowledge on questions that require a live lookup. Val gain (+0.30) is larger than train gain (+0.15) — the improvement generalized to unseen examples rather than memorizing training cases.

**`task_completeness` is consistent across splits** (+0.12 train, +0.11 val). The agent is more often producing answers grounded in actual lookups rather than plausible-sounding guesses. The consistency between splits is a good signal.

**`sequence_respected` is the weakest gain on train (+0.06 vs +0.28 on val).** The optimizer improved sequencing but didn't solve it — roughly 1 in 4 sequencing cases is still wrong on the train set. The small n (18 sequencing examples) makes this noisy; the val gain is more meaningful because it's clean data the optimizer never touched.

### Overfitting check

If the optimized prompt had overfit to training examples, you'd expect train scores to be noticeably higher than val scores. The opposite is true: val gains are at least as large as train gains on every metric. This indicates the prompt learned general routing principles rather than patching specific cases.

---

## Evaluator iteration — what changed and why

After analyzing the round 1 results, we discovered that `sequence_respected` was producing **false negatives on comparison examples**. The original evaluator treated every agent in `expected_sequence` as required — if `product_comparison_agent` wasn't called, the example scored 0.0, even when the required ordering (catalog → cart) was perfectly correct.

This matters because `product_comparison_agent` is not strictly required: the orchestrator can reason from catalog data directly. It's the right specialized tool, but not the only correct path. Scoring 0.0 for skipping it was punishing valid behavior.

**The fix:** updated `sequence_respected` to distinguish required agents (in `cannot_complete_without`) from optional agents (in `expected_sequence` only). Required ordering drives 80% of the score; calling optional agents is a 20% bonus.

We also added **BATCH 2 to the edge case dataset** — 5 harder comparison + action examples targeting the specific gap the analysis revealed: cases where the cart action depends on the comparison output, making it genuinely hard to get the right SKU without doing the comparison first.

### Re-run results with updated evaluator (train set)

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.528 | **0.602** | +0.07 |
| `critical_agents_called` | 0.676 | **0.870** | +0.19 |
| `sequence_respected` | 0.767 | **0.822** | +0.06 |

The `sequence_respected` scores are now much higher than previously reported (was 0.222/0.278 baseline/optimized). This is not an improvement in agent behavior — it's a correction to what we were measuring. The agent was already getting the required ordering (catalog → cart) right most of the time. The old metric was conflating "required ordering" with "called every agent we expected," which made it look worse than it was.

The delta is still meaningful: +0.06 represents the optimized prompt doing a better job of calling the optional comparison agent when appropriate — capturing the 20% bonus more often.

**`critical_agents_called` remains the strongest signal** at +0.19 on train. This metric is fully deterministic and measures the thing that matters most: is the agent grounding its answers in real data lookups rather than general knowledge?

### What this illustrates about eval iteration

The first version of any evaluator is a hypothesis about what "correct behavior" means. Running it against real agent outputs reveals where that hypothesis was wrong — either too strict (false negatives, like the comparison agent case) or too loose (false positives, where the agent passes but the answer is wrong). Iteration on evaluators is not a sign of a broken process; it is the process. Build in checkpoints to audit evaluator logic alongside dataset quality, especially after the first round of optimization when you have real failure data to reason from.

---

## Dataset restructuring before round 2

Before running round 2 optimization, we caught two problems with the dataset that needed fixing first.

**Problem 1: random split left product_comparison severely underrepresented in val.** The original `split_examples` used a flat shuffle — with only 12 comparison examples total, the random draw happened to assign 10 to train and 1 to val. Comparison tasks were the main failure cluster from round 1, so val was essentially blind to them. Optimization gains on comparison queries had no independent validation signal.

**Fix:** switched to stratified sampling — each task type is split proportionally within its group before the splits are combined. Result:

| Task type | Old train/val/test | New train/val/test |
|---|---|---|
| `product_comparison` | 10 / 1 / 1 | 7 / 2 / 3 |
| `action_with_prerequisite` | 11 / 7 / 5 | 13 / 4 / 6 |
| `return_eligibility` | 9 / 2 / 4 | 9 / 3 / 3 |
| `sizing_with_context` | 9 / 2 / 1 | 7 / 2 / 3 |
| `warranty_lookup` | 3 / 1 / 4 | 4 / 1 / 3 |

All previous eval results are invalidated by this change — new baselines were run on the restructured splits.

**Problem 2: val `sequence_respected` baseline was wildly understated.** Before the evaluator fix, val reported a baseline of 0.29. After correcting the false-negative issue in `sequence_respected` (required vs optional agent weighting), the corrected val baseline was 0.80 — meaning the agent was already getting sequencing right most of the time. The old number made round 1 look like a bigger win on sequencing than it actually was.

**The pattern:** data quality problems compound. A bad split + a bad evaluator together produce results that look meaningful but aren't. Fixing both before round 2 gives us a clean foundation: comparable train/val difficulty, accurate sequencing measurement, and a val set that actually tests the task types we're trying to improve.

**Progressive hardening plan:** once round 2 optimization converges on the current 90-example distribution, we add the 15 edge cases (split 8 to train, 7 to val) with a `difficulty_level` field to distinguish them from the core examples. Adding harder examples resets the baseline lower — that's expected and healthy. Round 3 then optimizes against the harder distribution. The test set is untouched until the final round.

---

## Evaluator generality — handling multiple valid execution paths

A second evaluator iteration came from a different direction: not a false negative in how we scored ordering, but a false negative in how we scored **which agents count as valid**.

The problem surfaced in `critical_agents_called`. For queries like "Add the AirPods Pro to my cart," the dataset listed `product_catalog_agent` in `cannot_complete_without`. But the agent sometimes solved these queries by calling `product_discovery_agent` (semantic search) instead — and got the right answer. The evaluator scored these 0.0. The metric was measuring "did the agent use the tool we expected" rather than "did the agent use a tool capable of completing the task."

This is a version of the same mistake as over-constraining `sequence_respected`: the evaluator's model of correct behavior was too narrow. In both cases, there are multiple valid execution paths to the right answer, and an evaluator that only accepts one path introduces noise that makes optimization harder and results harder to interpret.

**The fix:** OR semantics in `cannot_complete_without`. The field now supports a mixed list of items:
- A **string** → that exact agent must be called (AND semantics, unchanged)
- A **list of strings** → at least one agent in the group must be called (OR semantics)

Example: `[["product_catalog_agent", "product_discovery_agent"], "cart_and_orders_agent"]` scores 1.0 if either catalog or discovery was called, AND cart_and_orders was called.

The key question for each example is not "which agent did we expect?" but "which agents could actually complete this task correctly?" For queries that are pure product lookups — no order history, no price history, no internal reviews — both `product_catalog_agent` (structured SQL search) and `product_discovery_agent` (semantic vector search) can retrieve the product and its SKU. Either is correct. For queries that require order history, price history, or review data, only `product_catalog_agent` has those tools, so it stays as a hard AND requirement.

This generalizes a principle: **evaluators should model the task requirements, not the expected trajectory.** The expected trajectory is a hypothesis about how the agent will solve the problem. The task requirements describe what must be true about the solution. When those two things get conflated, the evaluator becomes brittle — it breaks as soon as the agent finds a valid path you didn't anticipate. Designing evaluators around requirements rather than trajectories makes them more durable through prompt changes, agent changes, and tool additions.

---

## Val metrics — full results log

Each row is one optimizer run. Dataset and evaluator state noted per run.

### Run 1 — original dataset (flat split), original evaluator

52 examples, flat random split, `sequence_respected` treated all sequence agents as equally required.

**Val:**

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.50 | **0.61** | +0.11 |
| `critical_agents_called` | 0.64 | **0.94** | +0.30 |
| `sequence_respected` | 0.29 | **0.57** | +0.28 |

**Train:**

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.519 | **0.639** | +0.12 |
| `critical_agents_called` | 0.685 | **0.833** | +0.15 |
| `sequence_respected` | 0.222 | **0.278** | +0.06 |

### Run 2 — original dataset (flat split), updated evaluator (required vs optional weighting)

Same 52 examples, same flat split. `sequence_respected` now weights required agents (80%) vs optional (20%).

**Train (re-run with corrected evaluator):**

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.528 | **0.602** | +0.07 |
| `critical_agents_called` | 0.676 | **0.870** | +0.19 |
| `sequence_respected` | 0.767 | **0.822** | +0.06 |

Val `sequence_respected` baseline corrected to **0.80** (was 0.29 — measurement fix, not agent improvement).

### Run 3 — updated dataset (stratified split, OR-group evaluator, full-sequence return examples), round 2

95 examples (55 train / 17 val / 23 test), stratified split, `critical_agents_called` supports OR groups, Allbirds bug fixed, 3 new catalog→policy→cart return examples added to train, Alo leggings 3-step return example in val.

**Val:**

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.559 | **0.618** | +0.06 |
| `critical_agents_called` | 0.637 | **0.980** | +0.34 |
| `sequence_respected` | 0.861 | **0.822** | -0.04 |

`critical_agents_called` at +0.34 is the strongest gain across all three rounds. Near-perfect (0.980) with low variance (std 0.08). The small `sequence_respected` dip reflects harder val examples (the new 3-step return) and a high baseline leaving little room — agent sequencing was already mostly correct.

### Run 5 — with-edge dataset (core + 14 edge examples in train, core-only val)

68 train (54 core + 14 edge) / 17 val (core only — edge examples have unique difficulty categories so all went to train). Edge examples cover adversarial cases: ambiguous references, cross-user auth, timing edge cases, personal history, competitor comparisons. Val set is identical to Run 4 so scores are directly comparable.

**Val:**

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.53 | **0.56** | +0.03 |
| `critical_agents_called` | 0.62 | **0.88** | +0.26 |
| `sequence_respected` | 0.50 | **0.90** | +0.40 |

`sequence_respected` is the standout: +0.40 gain and 0.90 mean with std 0.11 (very low variance). Training on edge sequencing examples made the agent dramatically more consistent about ordering. `critical_agents_called` also strong at +0.26. `task_completeness` gain is modest — edge examples trained the agent on harder routing but the val content is the same 17 core examples, so the ceiling on completeness gains here is limited.

### Run 6 — with-edge dataset, fixed task_completeness rubric for edge cases

Same 68 train / 17 val split as Run 5. `required_info` for auth refusals and personal history examples updated to describe outcomes rather than exact phrasing (e.g. "agent declines to access Jordan's account" not "agent should ask Alex which headphones"). Val set identical to Run 4 so directly comparable.

**Val:**

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.56 | **0.67** | +0.11 |
| `critical_agents_called` | 0.62 | **0.90** | +0.28 |
| `sequence_respected` | 0.58 | **0.88** | +0.30 |

Strong gains across all three metrics. Once the rubric accepted any correct refusal rather than a specific phrasing, `task_completeness` improved meaningfully. `sequence_respected` and `critical_agents_called` both near 0.90.

---

### Run 4 — v4 dataset (30-day Sony fix, AirPods not-delivered fix, return logic: cart OR conditions-confirmed, removed duplicate Uniqlo, judge reasoning added to optimizer input)

94 examples (54 train / 17 val / 23 test). Key data quality fixes: Sony electronics window corrected to 30 days, AirPods example fixed to reflect undelivered status, all return action examples updated so "conditions confirmed" counts as valid completion alongside "return initiated". LLM judge reasoning now fed into optimizer.

**Val:**

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| `task_completeness` | 0.47 | **0.71** | +0.24 |
| `critical_agents_called` | 0.59 | **0.97** | +0.38 |
| `sequence_respected` | 0.58 | **0.78** | +0.20 |

Best result across all runs. Optimizer experiment: `optimized-val-v4-354689a9`.

---

### Statistical Reliability — 6× repeated runs per condition

Each single-run result above is subject to LLM non-determinism: the same prompt can score slightly differently on different invocations because the agent (and in some cases the LLM judge) samples stochastically. To confirm the gains are real and not noise, we ran each condition 6 times on the same dataset and report mean ± standard deviation across runs.

**What this tells us:** If the standard deviation is small relative to the mean difference between baseline and optimized, the improvement is reliable — it isn't just a lucky draw. Conversely, high variance would mean the prompt is fragile and small changes in model sampling could reverse the result.

**Core dataset (v4 val), n=6 runs per condition:**

| Condition | task_completeness | critical_agents_called | sequence_respected |
|---|---|---|---|
| Baseline | 0.485 ± 0.052 | 0.578 ± 0.030 | 0.585 ± 0.042 |
| Optimized | **0.623 ± 0.063** | **0.926 ± 0.025** | **0.911 ± 0.017** |
| **Δ** | **+0.138** | **+0.348** | **+0.326** |

**With-edge dataset, n=6 runs per condition:**

| Condition | task_completeness | critical_agents_called | sequence_respected |
|---|---|---|---|
| Baseline | 0.520 ± 0.036 | 0.578 ± 0.030 | 0.568 ± 0.054 |
| Optimized | **0.686 ± 0.055** | **0.946 ± 0.012** | **0.894 ± 0.049** |
| **Δ** | **+0.166** | **+0.368** | **+0.326** |

**Key observations:**

- **`critical_agents_called` is the most reliable gain.** The optimized prompt's std dev is 0.025 (core) and 0.012 (edge) — essentially negligible variance. The gap between baseline and optimized (≈0.35) is 10–30× larger than the noise, making this the most statistically robust result.
- **`sequence_respected` is similarly stable** on the core dataset (std 0.017). The optimizer is consistently teaching the agent to respect tool ordering.
- **`task_completeness` has more variance** (~0.06) in both conditions. This is expected: it's an LLM-as-judge metric evaluating subjective response quality, so it inherits some sampling noise from the judge itself. The mean gains (+0.14 core, +0.17 edge) are still well above the noise floor.
- **Edge cases don't hurt.** The with-edge optimized scores are slightly *better* than core-only on all three metrics, suggesting the edge training examples improved generalization rather than overfitting.
- **The baseline is also reasonably stable** (std 0.03–0.05), confirming the dataset itself is not the source of noise — the gains are coming from the prompt, not lucky example ordering.

---

# Optimizer Agent Pipeline

The optimizer is a **LangGraph StateGraph** — a 6-node pipeline where each node has a defined responsibility and the heavier reasoning steps run on a stronger model.

```
pull_results → analyze → reflect → generate → review → save
```

| Node | Model | What it does |
|---|---|---|
| `pull_results` | — | Pulls **all** eval runs from LangSmith with full context: inputs, reference outputs, actual agent outputs, agents called, and per-metric scores |
| `analyze` | claude-sonnet-4-6 | Identifies failure patterns per metric across task types |
| `reflect` | **claude-opus-4-6** | Filters analysis to changes that will improve specific metrics on unseen examples |
| `generate` | **claude-opus-4-6** | Writes the optimized prompt incorporating only validated improvements |
| `review` | **claude-opus-4-6** | Final pass: strips anything that hardcodes training-specific details |
| `save` | — | Writes `prompts/optimized.md` |

## What the optimizer receives

The `pull_results` node fetches **all runs** from the baseline-train experiment — not just failures — with full context per example:

- `query` and `user_email` (inputs)
- `cannot_complete_without`, `expected_sequence`, `required_info` (reference outputs, looked up from train.jsonl)
- `agents_called` and `final_output` (actual agent outputs, truncated at 400 chars)
- Per-metric scores: `task_completeness`, `critical_agents_called`, `sequence_respected`

Pulling all results, not just failures, is important: the optimizer needs to see what the agent did right in order to avoid breaking it. And for examples that score 0.5, seeing the actual response alongside the reference output tells the optimizer far more than a score alone.

## Metric-driven analysis

The optimizer is given an `EVAL_METRICS` dict that maps each metric name to a diagnostic question:

```
task_completeness      → "What does the prompt fail to say about how to handle this task type?"
critical_agents_called → "What does the prompt fail to communicate about which agents hold ground truth?"
sequence_respected     → "What does the prompt fail to say about when to stop, what to check first, or when one result should gate the next?"
```

The **target is 1.0 on every metric for every example** — not a threshold like "fix examples below 0.6." Every score below 1.0 indicates a prompt gap. The analysis node is asked to ground every finding in this framing: which prompt gap explains why this specific metric fell short on this specific task type?

## The reflection node as an overfit filter

The **reflect node is structurally separate** — not just another message in the same chain. For each proposed change from the analysis, it applies two tests before passing it forward:

1. **Metric impact**: which specific metric does this change improve, and why? A proposed change with no clear metric tie is discarded.
2. **Generalization**: would this improvement apply to unseen queries of the same task type, or does it only patch the specific training examples shown? Anything that patches a specific example rather than a class of behavior is discarded.

Only changes that pass both tests are forwarded to the generate node. The goal is fewer, sharper changes — not a comprehensive list of every observation.

## Human prior

The optimizer is given a brief routing principles document as context — not a specific example prompt to copy. This document describes what makes routing prompts effective in general (what data each agent uniquely holds, when ordering genuinely matters, when to stop early). It anchors the optimizer toward known-good patterns without creating a risk of copying the prior's specifics.

## Anti-overfit mechanisms

Three layers prevent the optimizer from producing a prompt that memorizes training examples:

1. **Reflect node** explicitly discards example-specific insights before they reach generate
2. **Generate node** is instructed not to introduce constraints or behaviors beyond what the reflect node validated, and never to hardcode specific entities, identifiers, or values from training
3. **Review node** does a final sweep — every rule and instruction is checked: does it reference training-specific details? Would a reasonable unseen query of the same type be handled correctly?

## Why LangGraph for the optimizer

The linear node structure makes the optimizer's reasoning transparent and auditable. Each node's output is state in the graph — you can inspect what the analyze node concluded, what the reflect node kept or discarded, and what the generate node produced, independently. This is useful for debugging: when an optimized prompt is wrong, you can trace exactly which node introduced the error.

It also makes it easy to intervene. If the reflect output looks too conservative, you can re-run just the generate and review nodes with a different reflect output. The state is explicit; there's no hidden reasoning to untangle.

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
- [x] 15 edge case examples (BATCH 1 + BATCH 2) — BATCH 2 added after analyzing round 1 gaps
- [x] Three evaluators: task_completeness (LLM judge), critical_agents_called, sequence_respected
- [x] sequence_respected updated: required vs optional agent weighting (80/20) to eliminate false negatives
- [x] Trajectory capture from Deep Agents stream (correct chunk structure)
- [x] Baseline eval on train + val
- [x] LangGraph optimizer pipeline (5 nodes, reflection on opus)
- [x] Optimized prompt generated and saved
- [x] Optimized eval on val: 0.50→0.61 / 0.64→0.94 / 0.29→0.57
- [x] Optimized eval on train: 0.519→0.639 / 0.685→0.833 / 0.222→0.278 (no overfitting signal)
- [x] Edge dataset re-uploaded (15 examples), fresh train evals running with updated evaluator

## Remaining
- [ ] Round 2: fresh baseline on new stratified splits (in progress)
- [ ] Round 2: run optimizer on new train baseline failures → new optimized prompt
- [ ] Round 2: eval optimized prompt on train + val
- [ ] Add harder edge examples to train/val (split 8/7) with difficulty_level field
- [ ] Round 3: re-baseline on expanded train/val (harder distribution) → run optimizer → eval
- [ ] Final eval on test set (run exactly once, after all optimization rounds complete)
- [ ] Prompt diff write-up: what changed between baseline → round 2 → round 3
- [ ] Add graphs (baseline vs optimized across all three metrics)

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