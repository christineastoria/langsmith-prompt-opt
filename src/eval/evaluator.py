"""
Evaluators for the shopping concierge routing eval.

Three targeted metrics:

  task_completeness     — LLM judge: did the answer actually complete the task?
                          Checks against required_info, not just agent calls.

  critical_agents_called — Code: were the agents marked cannot_complete_without called?
                           These are the ones whose absence makes the answer wrong.

  sequence_respected    — Code: for sequential tasks, did prerequisite agents fire
                          before the action agent? Wrong order = wrong answer.
"""

import asyncio
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic


# ---------------------------------------------------------------------------
# LLM judge setup
# ---------------------------------------------------------------------------

class CompletionGrade(TypedDict):
    reasoning: Annotated[str, ..., "Step-by-step reasoning for your score"]
    score: Annotated[float, ..., "0.0 = task not completed, 0.5 = partially completed, 1.0 = fully completed"]


_judge = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
).with_structured_output(CompletionGrade)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outputs(run) -> dict:
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def _expected(example) -> dict:
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


# ---------------------------------------------------------------------------
# Evaluator 1: Task completeness (LLM-as-judge)
# Checks whether the final answer actually satisfies the task.
# Uses required_info as the rubric — not which agents were called.
# ---------------------------------------------------------------------------

def task_completeness(run, example) -> dict:
    """
    Did the agent's final answer actually complete the task?
    Scored against required_info from the dataset, not agent trajectory.
    """
    run_out = _outputs(run)
    ex_out = _expected(example)

    final_answer = run_out.get("final_output", "")
    required_info = ex_out.get("required_info", [])
    task_type = ex_out.get("task_type", "unknown")
    query = run_out.get("query", "")

    if not final_answer:
        return {"score": 0.0, "comment": "No output captured from agent."}
    if not required_info:
        return {"score": None, "comment": "No required_info defined for this example."}

    required_str = "\n".join(f"- {r}" for r in required_info)

    grade: CompletionGrade = asyncio.run(_judge.ainvoke([{
        "role": "user",
        "content": (
            f"A customer asked: \"{query}\"\n\n"
            f"Task type: {task_type}\n\n"
            f"The agent's response was:\n{final_answer}\n\n"
            f"For this task to be complete, the response must include ALL of:\n{required_str}\n\n"
            f"Score 1.0 if all required information is present and the task is fully resolved. "
            f"Score 0.5 if some required information is present but the answer is incomplete. "
            f"Score 0.0 if the answer is missing critical information or is wrong."
        ),
    }]))

    return {"score": grade["score"], "comment": grade["reasoning"]}


# ---------------------------------------------------------------------------
# Helpers for OR-group semantics in cannot_complete_without
# ---------------------------------------------------------------------------

def _flatten_critical(critical: list) -> set:
    """Return the set of all agent names mentioned in cannot_complete_without (flattened)."""
    result = set()
    for item in critical:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, list):
            result.update(item)
    return result


# ---------------------------------------------------------------------------
# Evaluator 2: Critical agents called (code check)
# Only checks agents marked cannot_complete_without — not all expected agents.
# These are the ones where absence = factually wrong or impossible answer.
#
# cannot_complete_without supports two item formats:
#   - str  → that exact agent must be called (AND)
#   - list → at least one agent in the list must be called (OR group)
#
# Example: [["product_catalog_agent", "product_discovery_agent"], "web_research_agent"]
#   means (product_catalog_agent OR product_discovery_agent) AND web_research_agent.
# ---------------------------------------------------------------------------

def critical_agents_called(run, example) -> dict:
    """
    Were the agents strictly necessary for this task actually called?
    Only checks cannot_complete_without — not the full expected_agents list.
    Supports OR groups via nested lists.
    """
    actual: list = _outputs(run).get("agents_called", [])
    critical: list = _expected(example).get("cannot_complete_without", [])

    if not critical:
        return {"score": None, "comment": "No critical agents defined — skipped."}

    actual_set = set(actual)
    satisfied = []
    unsatisfied = []

    for item in critical:
        if isinstance(item, str):
            if item in actual_set:
                satisfied.append(item)
            else:
                unsatisfied.append(item)
        elif isinstance(item, list):
            label = f"({' OR '.join(item)})"
            if any(a in actual_set for a in item):
                satisfied.append(label)
            else:
                unsatisfied.append(label)

    score = len(satisfied) / len(critical)

    if not unsatisfied:
        return {
            "score": 1.0,
            "comment": f"All critical requirements satisfied: {satisfied}.",
        }

    return {
        "score": score,
        "comment": (
            f"Unsatisfied critical requirements: {unsatisfied}. "
            f"Satisfied: {satisfied}. "
            f"Called: {sorted(actual_set)}."
        ),
    }


# ---------------------------------------------------------------------------
# Evaluator 3: Sequence respected (code check)
# For sequential tasks: did the prerequisite agent fire before the action?
# Doesn't require exact order of all agents — just that lookups precede actions.
# ---------------------------------------------------------------------------

def sequence_respected(run, example) -> dict:
    """
    For sequential tasks: did prerequisite lookups happen before actions?

    Scoring is weighted by whether agents are required vs. optional:
    - Agents in both expected_sequence AND cannot_complete_without: required.
      Their ordering is the hard constraint — drives 80% of the score.
    - Agents in expected_sequence but NOT cannot_complete_without: optional.
      Bonus for calling them — drives the remaining 20%.

    If there are no optional agents, the full score comes from required ordering.
    Non-sequential examples return score=None (excluded from aggregation).
    """
    ex_out = _expected(example)

    if not ex_out.get("requires_sequencing", False):
        return {"score": None, "comment": "Non-sequential example — skipped."}

    actual: list = _outputs(run).get("agents_called", [])
    expected_seq: list = ex_out.get("expected_sequence", [])
    required_set: set = _flatten_critical(ex_out.get("cannot_complete_without", []))

    if len(expected_seq) < 2:
        return {"score": None, "comment": "Sequence has fewer than 2 steps — skipped."}

    if not actual:
        return {"score": 0.0, "comment": "No agents called."}

    required_seq = [a for a in expected_seq if a in required_set]
    optional_seq = [a for a in expected_seq if a not in required_set]

    def _ordering_score(seq: list, trajectory: list) -> tuple[float, str]:
        """Score how well a subsequence is ordered within the trajectory."""
        positions = {a: trajectory.index(a) for a in seq if a in trajectory}
        missing = [a for a in seq if a not in positions]
        if missing:
            # Partial: fraction present, capped at 0.5 since ordering can't be verified
            frac = len(positions) / len(seq) * 0.5
            return frac, f"missing: {missing}"
        ordered = [positions[a] for a in seq]
        correct = sum(1 for i in range(len(ordered) - 1) if ordered[i] < ordered[i + 1])
        return correct / (len(ordered) - 1), f"{correct}/{len(ordered) - 1} pairs in order"

    # Score required ordering
    if len(required_seq) >= 2:
        required_score, req_note = _ordering_score(required_seq, actual)
    else:
        required_score, req_note = 1.0, "no required ordering constraint"

    # Score optional agents (bonus: were they called at all?)
    if optional_seq:
        called_optional = [a for a in optional_seq if a in actual]
        optional_score = len(called_optional) / len(optional_seq)
        OPTIONAL_WEIGHT = 0.2
        score = required_score * (1 - OPTIONAL_WEIGHT) + optional_score * OPTIONAL_WEIGHT
        comment = (
            f"Required {required_seq}: {req_note}. "
            f"Optional {optional_seq}: {len(called_optional)}/{len(optional_seq)} called. "
            f"Score={score:.2f} (80% required + 20% optional)."
        )
    else:
        score = required_score
        comment = f"Required {required_seq}: {req_note}. Score={score:.2f}."

    return {"score": score, "comment": comment}
