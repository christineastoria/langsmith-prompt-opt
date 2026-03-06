"""
Prompt optimizer — LangGraph pipeline with a dedicated reflection node.

Nodes:
  pull_failures   → query LangSmith for low-scoring train runs
  analyze         → claude-sonnet-4-6: broad failure pattern analysis
  reflect         → claude-opus-4-6:   what generalizes vs. overfits?
  generate        → claude-opus-4-6:   write the optimized prompt
  review          → claude-opus-4-6:   strip anything example-specific
  save            → write prompts/optimized.md

    uv run python src/optimizer/run_optimizer.py
    uv run python src/optimizer/run_optimizer.py --experiment baseline-train-abc123
"""

import argparse
import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langsmith import Client

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
TRAIN_JSONL = Path(__file__).parent.parent.parent / "data" / "eval" / "train.jsonl"

# Models — reflection and generation use the strongest available
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# Human prior: general routing principles.
# Abstract enough that it anchors without causing overfitting to train examples.
# ---------------------------------------------------------------------------
ROUTING_PRINCIPLES = """\
Characteristics of effective routing prompts for multi-agent shopping concierges:

1. Each subagent is described by what it uniquely HOLDS (the data it owns),
   not just what it does. "Call X for order history" is more useful than "X handles orders".

2. Decision criteria are explicit: when is each agent NECESSARY vs. optional?
   The agent must understand which agents are the ground-truth holders for which facts.

3. Ordering constraints are stated only where they genuinely matter for correctness —
   e.g., you need a real SKU/order_id before a cart action; you can't hallucinate one.

4. The prompt distinguishes between "complete the action" and "check eligibility first".
   Some requests end at the eligibility check; not every query reaches a cart action.

5. For queries requiring external data (competitor prices, external reviews), the prompt
   makes clear that internal catalog data alone is insufficient.

Hard constraints for generated prompts:
- Do NOT hardcode specific product names, user names, SKUs, or prices.
- Every rule must generalize to unseen queries of the same type, not just training examples.
"""

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class OptimizerState(TypedDict):
    baseline_prompt: str
    experiment_name: str
    failure_summary: str        # formatted failures from LangSmith
    analysis: str               # Step 1: broad pattern analysis
    reflection: str             # Step 2: what generalizes vs. overfits
    draft_prompt: str           # Step 3: generated optimized prompt
    final_prompt: str           # Step 4: after anti-overfit review


# ---------------------------------------------------------------------------
# LangSmith helpers
# ---------------------------------------------------------------------------

def find_latest_experiment(client: Client, prefix: str = "baseline-train") -> str:
    projects = list(client.list_projects())
    matches = [p for p in projects if p.name.startswith(prefix)]
    if not matches:
        raise ValueError(
            f"No LangSmith project found starting with '{prefix}'. "
            "Run: uv run python src/eval/run_eval.py --prompt baseline --split train"
        )
    matches.sort(key=lambda p: p.start_time, reverse=True)
    return matches[0].name


def load_train_index() -> dict:
    index = {}
    for line in TRAIN_JSONL.read_text().splitlines():
        if line.strip():
            ex = json.loads(line)
            key = (ex["inputs"]["query"], ex["inputs"]["user_email"])
            index[key] = ex["outputs"]
    return index


def pull_and_format_results(client: Client, experiment_name: str) -> str:
    """Pull all train eval runs, format with scores + expected routing."""
    print(f"  Querying experiment '{experiment_name}'...")
    runs = list(client.list_runs(
        project_name=experiment_name,
        run_type="chain",
        is_root=True,
    ))
    print(f"  Found {len(runs)} runs")

    train_index = load_train_index()
    lines = []

    for run in runs:
        feedbacks = list(client.list_feedback(run_ids=[str(run.id)]))
        scores = {f.key: f.score for f in feedbacks if f.score is not None}

        query = (run.inputs or {}).get("query", "?")
        email = (run.inputs or {}).get("user_email", "?")
        expected = train_index.get((query, email), {})

        any_fail = any(
            scores.get(k, 1.0) < 0.6
            for k in ["task_completeness", "critical_agents_called"]
        )
        status = "FAIL" if any_fail else "pass"

        score_str = "  ".join(f"{k}: {v:.2f}" for k, v in sorted(scores.items()))
        lines.append(
            f"[{status}] \"{query}\" ({email})\n"
            f"  scores: {score_str}\n"
            f"  task_type: {expected.get('task_type', '?')}\n"
            f"  critical_agents: {', '.join(expected.get('cannot_complete_without', []))}\n"
            f"  failure_mode: {expected.get('failure_mode', '?')}"
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def node_pull_failures(state: OptimizerState) -> dict:
    print("\n[pull_failures] Querying LangSmith...")
    client = Client()

    experiment = state["experiment_name"]
    if not experiment:
        experiment = find_latest_experiment(client)
        print(f"  Auto-detected: {experiment}")

    summary = pull_and_format_results(client, experiment)
    return {"experiment_name": experiment, "failure_summary": summary}


def node_analyze(state: OptimizerState) -> dict:
    print("\n[analyze] Identifying failure patterns (sonnet)...")
    llm = ChatAnthropic(model=SONNET, temperature=0)

    response = llm.invoke([HumanMessage(content=f"""\
You are analyzing routing failures for a personal shopping concierge with these subagents:
- product_catalog_agent: owns order history, delivery status, prices, SKUs, internal reviews
- policy_and_sizing_agent: owns return windows, brand-specific rules, non-returnable items, sizing guides
- cart_and_orders_agent: executes cart actions (add, return, wishlist) — needs a real SKU/order_id
- product_comparison_agent: structures comparisons once catalog data is available
- product_discovery_agent: searches catalog by category or criteria
- web_research_agent: fetches external prices, reviews, trends

Current system prompt:
<prompt>
{state['baseline_prompt']}
</prompt>

Train eval results ({len(state['failure_summary'].splitlines())} lines, FAIL = metric < 0.6):
<results>
{state['failure_summary']}
</results>

{ROUTING_PRINCIPLES}

What routing patterns does the current prompt fail to support? \
Focus on patterns across task types — not individual examples. \
What decision logic or agent knowledge is the prompt missing?\
""")])

    return {"analysis": response.content}


def node_reflect(state: OptimizerState) -> dict:
    """Dedicated reflection node — uses the strongest model to separate
    general patterns from example-specific overfitting risks."""
    print("\n[reflect] Reflecting on generalizability (opus)...")
    llm = ChatAnthropic(model=OPUS, temperature=0)

    response = llm.invoke([HumanMessage(content=f"""\
You are reviewing a failure analysis for a routing prompt. Your job is to be a critical \
filter: separate what genuinely generalizes from what would overfit to the training examples.

Failure analysis:
<analysis>
{state['analysis']}
</analysis>

For each insight in the analysis, assess:
1. GENERAL — would this help the agent route correctly on unseen queries of the same type?
2. OVERFIT — would this only patch the exact training examples shown?

Then produce a clean list of ONLY the general, safe-to-incorporate insights, with \
a brief note on why each generalizes. Explicitly call out and discard anything overfit.

Be strict. A rule that mentions a specific product category is fine. \
A rule that only makes sense for one specific query is not.\
""")])

    return {"reflection": response.content}


def node_generate(state: OptimizerState) -> dict:
    print("\n[generate] Writing optimized prompt (opus)...")
    llm = ChatAnthropic(model=OPUS, temperature=0)

    response = llm.invoke([HumanMessage(content=f"""\
Write an optimized system prompt for the shopping concierge.

Original prompt:
<original>
{state['baseline_prompt']}
</original>

Validated general patterns to incorporate:
<patterns>
{state['reflection']}
</patterns>

{ROUTING_PRINCIPLES}

The optimized prompt must:
- Describe each subagent by what data it uniquely holds
- Give clear decision criteria for when each agent is necessary
- Specify ordering constraints only where genuinely required for correctness
- Handle the case where eligibility check blocks the action (don't call cart agent)
- Be a system prompt — concise and durable, not a case-by-case playbook

Write only the system prompt text.\
""")])

    return {"draft_prompt": response.content}


def node_review(state: OptimizerState) -> dict:
    """Anti-overfit review — final pass to strip anything example-specific."""
    print("\n[review] Anti-overfit review (opus)...")
    llm = ChatAnthropic(model=OPUS, temperature=0)

    response = llm.invoke([HumanMessage(content=f"""\
Review this optimized routing prompt for a shopping concierge:

<prompt>
{state['draft_prompt']}
</prompt>

Check every rule and instruction:
- Does it reference specific products, brands, users, prices, or SKUs?
- Is any constraint so narrow it would only apply to the training queries?
- Would a reasonable unseen query of the same type be handled correctly?

If the prompt is sound, output it unchanged.
If there are overfit elements, rewrite only those parts to be more general.

Output only the final prompt text — no commentary, no preamble.\
""")])

    return {"final_prompt": response.content}


def node_save(state: OptimizerState) -> dict:
    output_path = PROMPTS_DIR / "optimized.md"
    output_path.write_text(state["final_prompt"])
    print(f"\n[save] Written → {output_path}")
    print(f"\n{'='*60}\n{state['final_prompt']}\n{'='*60}")
    return {}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_optimizer_graph():
    graph = StateGraph(OptimizerState)

    graph.add_node("pull_failures", node_pull_failures)
    graph.add_node("analyze", node_analyze)
    graph.add_node("reflect", node_reflect)
    graph.add_node("generate", node_generate)
    graph.add_node("review", node_review)
    graph.add_node("save", node_save)

    graph.set_entry_point("pull_failures")
    graph.add_edge("pull_failures", "analyze")
    graph.add_edge("analyze", "reflect")
    graph.add_edge("reflect", "generate")
    graph.add_edge("generate", "review")
    graph.add_edge("review", "save")
    graph.add_edge("save", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default=None,
        help="LangSmith experiment name (default: auto-detect latest baseline-train-*)",
    )
    args = parser.parse_args()

    app = build_optimizer_graph()
    app.invoke({
        "baseline_prompt": (PROMPTS_DIR / "baseline.md").read_text().strip(),
        "experiment_name": args.experiment or "",
        "failure_summary": "",
        "analysis": "",
        "reflection": "",
        "draft_prompt": "",
        "final_prompt": "",
    })
