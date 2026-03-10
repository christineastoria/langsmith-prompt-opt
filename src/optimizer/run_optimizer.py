"""
Prompt optimizer — LangGraph pipeline with a dedicated reflection node.

Nodes:
  pull_eval_results → query LangSmith for low-scoring train runs
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
# Evaluation metrics — what each metric's failures indicate about prompt gaps.
# Customize this for your eval setup.
# ---------------------------------------------------------------------------
EVAL_METRICS = {
    "task_completeness": (
        "The agent's response did not fully satisfy what the user needed. "
        "Ask: what does the prompt fail to say about how to handle this task type? "
        "What decision logic, eligibility check, or response behavior is missing or ambiguous?"
    ),
    "critical_agents_called": (
        "The agent skipped one or more agents required to complete the task. "
        "Ask: what does the prompt fail to communicate about which agents are the sole source "
        "of truth for which data or actions? Are ordering constraints missing — e.g., the agent "
        "skipped a required lookup before attempting an action?"
    ),
    "sequence_respected": (
        "The agent called agents in the wrong order, or called unnecessary agents when it should "
        "have stopped early. Ask: what does the prompt fail to say about when to stop, what to "
        "check first, or when one result should gate the next step?"
    ),
}

# ---------------------------------------------------------------------------
# Human prior: general routing principles for multi-agent systems.
# NOTE: Only relevant when optimizing prompts for multi-agent routing tasks.
# ---------------------------------------------------------------------------
ROUTING_PRINCIPLES = """\
Characteristics of effective routing prompts for multi-agent systems:

1. Each agent is described by what it uniquely HOLDS (the data or capability it owns),
   not just what it does. "Call X to retrieve Y" is more useful than "X handles Y-related tasks".

2. Decision criteria are explicit: when is each agent NECESSARY vs. optional?
   The orchestrator must know which agents are the sole source of truth for which facts.

3. Ordering constraints are stated only where they genuinely matter for correctness —
   i.e., when a downstream step requires verified output from an upstream step to function at all.

4. The prompt distinguishes "check eligibility / retrieve state" from "execute an action".
   Some requests end at the check; not every query reaches execution. If the check reveals
   the action is impossible, stop early and explain — do not call the action agent.

5. For queries requiring data that exists outside the internal system, the prompt makes
   clear that internal data alone is insufficient and which agent handles external lookups.

Hard constraints for generated prompts:
- Do NOT hardcode specific entities, identifiers, or values from the training examples.
- Every rule must generalize to unseen queries of the same type, not just training examples.
- Use ONLY the agent names and tool names that appear in the training data. Do not rename, merge, or invent agents or tools.
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
    """Pull all train eval runs with full inputs, reference outputs, actual outputs, and scores."""
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
        comments = {f.key: f.comment for f in feedbacks if f.comment}

        inputs = run.inputs or {}
        outputs = run.outputs or {}
        query = inputs.get("query", "?")
        email = inputs.get("user_email", "?")
        ref = train_index.get((query, email), {})

        score_str = "  ".join(f"{k}: {v:.2f}" for k, v in sorted(scores.items()))

        actual_agents = outputs.get("agents_called", [])
        actual_response = (outputs.get("final_output") or "").strip()
        # Truncate long responses but keep enough for the optimizer to reason about
        if len(actual_response) > 400:
            actual_response = actual_response[:400] + "…"

        ref_agents_raw = ref.get("cannot_complete_without", [])
        expected_seq = ref.get("expected_sequence", [])

        # Format OR groups (nested lists) for human-readable optimizer input
        def _fmt_critical(agents: list) -> str:
            parts = []
            for item in agents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, list):
                    parts.append(f"({' OR '.join(item)})")
            return ", ".join(parts) or "none"

        comment_lines = "  ".join(
            f"\n  judge[{k}]: {v}" for k, v in sorted(comments.items()) if v
        )

        lines.append(
            f"query: \"{query}\" ({email})\n"
            f"  task_type: {ref.get('task_type', '?')}\n"
            f"  scores: {score_str}\n"
            f"  reference — critical_agents: {_fmt_critical(ref_agents_raw)}"
            + (f"  sequence: {' → '.join(expected_seq)}" if expected_seq else "") + "\n"
            f"  reference — required_info: {'; '.join(ref.get('required_info', []))}\n"
            f"  reference — failure_mode: {ref.get('failure_mode', '?')}\n"
            f"  actual    — agents_called: {', '.join(actual_agents) or 'none'}\n"
            f"  actual    — response: {actual_response or '(empty)'}"
            + comment_lines
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def node_pull_eval_results(state: OptimizerState) -> dict:
    print("\n[pull_eval_results] Pulling all eval results from LangSmith...")
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

    metric_descriptions = "\n".join(
        f"- {name}: {description}" for name, description in EVAL_METRICS.items()
    )

    response = llm.invoke([HumanMessage(content=f"""\
You are analyzing evaluation results to identify what is wrong with a routing prompt. \
The goal is to reach a perfect score on every metric for every example.

The evaluation uses these metrics — any score below 1.0 indicates a prompt gap:

{metric_descriptions}

Current system prompt:
<prompt>
{state['baseline_prompt']}
</prompt>

Eval results — all examples with scores, reference outputs, and actual agent outputs \
(scores are 0.0–1.0; target is 1.0 on all metrics for all examples):
<results>
{state['failure_summary']}
</results>

{ROUTING_PRINCIPLES}

For each metric, identify the specific prompt gaps that explain why any example falls short. \
Focus on patterns across task types — not individual examples. \
Ground your analysis in the critical_agents and task_type fields shown in the results.\
""")])

    return {"analysis": response.content}


def node_reflect(state: OptimizerState) -> dict:
    """Dedicated reflection node — filters analysis to changes that will improve
    specific metric scores on unseen examples without overfitting."""
    print("\n[reflect] Reflecting on metric impact (opus)...")
    llm = ChatAnthropic(model=OPUS, temperature=0)

    metric_names = ", ".join(EVAL_METRICS.keys())

    response = llm.invoke([HumanMessage(content=f"""\
You are reviewing a failure analysis for a routing prompt. Your goal is to decide which \
proposed changes will actually improve scores on the evaluation metrics ({metric_names}) \
for unseen examples — not just patch the specific training examples.

Failure analysis:
<analysis>
{state['analysis']}
</analysis>

For each proposed change, assess both dimensions:

1. METRIC IMPACT — which metric(s) does this change improve, and why? \
   A change with no clear metric tie is not worth including.

2. GENERALIZATION — would this improvement apply to unseen queries of the same type, \
   or does it only fix the exact examples shown? \
   Discard anything that patches a specific example rather than a class of behavior.

Produce a prioritized list of ONLY the changes that pass both tests, with a brief note on \
which metric each change targets and why it generalizes. Be strict — fewer, sharper changes \
are better than many vague ones.\
""")])

    return {"reflection": response.content}


def node_generate(state: OptimizerState) -> dict:
    print("\n[generate] Writing optimized prompt (opus)...")
    llm = ChatAnthropic(model=OPUS, temperature=0)

    response = llm.invoke([HumanMessage(content=f"""\
Write an optimized system prompt based on the validated improvements below.

Original prompt:
<original>
{state['baseline_prompt']}
</original>

Validated improvements to incorporate (each tied to a specific metric):
<improvements>
{state['reflection']}
</improvements>

{ROUTING_PRINCIPLES}

The optimized prompt must:
- Incorporate each validated improvement exactly as specified
- Remain a system prompt — concise and durable, not a case-by-case playbook
- Not introduce any new constraints or behaviors beyond what is validated above

Write only the system prompt text.\
""")])

    return {"draft_prompt": response.content}


def node_review(state: OptimizerState) -> dict:
    """Anti-overfit review — final pass to strip anything example-specific."""
    print("\n[review] Anti-overfit review (opus)...")
    llm = ChatAnthropic(model=OPUS, temperature=0)

    response = llm.invoke([HumanMessage(content=f"""\
Review this optimized routing prompt for overfit or example-specific language:

<prompt>
{state['draft_prompt']}
</prompt>

Check every rule and instruction:
- Does it reference specific entities, identifiers, or values from training examples?
- Is any constraint so narrow it would only apply to specific training queries?
- Would a reasonable unseen query of the same task type be handled correctly?

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

    graph.add_node("pull_eval_results", node_pull_eval_results)
    graph.add_node("analyze", node_analyze)
    graph.add_node("reflect", node_reflect)
    graph.add_node("generate", node_generate)
    graph.add_node("review", node_review)
    graph.add_node("save", node_save)

    graph.set_entry_point("pull_eval_results")
    graph.add_edge("pull_eval_results", "analyze")
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
