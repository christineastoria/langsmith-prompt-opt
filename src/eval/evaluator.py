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
# Evaluator 2: Critical agents called (code check)
# Only checks agents marked cannot_complete_without — not all expected agents.
# These are the ones where absence = factually wrong or impossible answer.
# ---------------------------------------------------------------------------

def critical_agents_called(run, example) -> dict:
    """
    Were the agents strictly necessary for this task actually called?
    Only checks cannot_complete_without — not the full expected_agents list.
    """
    actual: list = _outputs(run).get("agents_called", [])
    critical: list = _expected(example).get("cannot_complete_without", [])

    if not critical:
        return {"score": None, "comment": "No critical agents defined — skipped."}

    actual_set = set(actual)
    critical_set = set(critical)
    missing = critical_set - actual_set

    if not missing:
        return {
            "score": 1.0,
            "comment": f"All critical agents called: {sorted(critical_set)}.",
        }

    score = len(critical_set & actual_set) / len(critical_set)
    return {
        "score": score,
        "comment": (
            f"Missing critical agents: {sorted(missing)}. "
            f"Without these, the task cannot be correctly completed. "
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
    Checks that expected_sequence[0] appears before expected_sequence[-1] in
    the actual trajectory. Non-sequential examples return score=None.
    """
    ex_out = _expected(example)

    if not ex_out.get("requires_sequencing", False):
        return {"score": None, "comment": "Non-sequential example — skipped."}

    actual: list = _outputs(run).get("agents_called", [])
    expected_seq: list = ex_out.get("expected_sequence", [])

    if len(expected_seq) < 2:
        return {"score": None, "comment": "Sequence has fewer than 2 steps — skipped."}

    if not actual:
        return {"score": 0.0, "comment": "No agents called."}

    # Check that each agent in expected_sequence appears in the correct relative order
    positions = {agent: actual.index(agent) for agent in expected_seq if agent in actual}
    missing_from_actual = [a for a in expected_seq if a not in positions]

    if missing_from_actual:
        return {
            "score": 0.0,
            "comment": (
                f"Required agents not called: {missing_from_actual}. "
                f"Actual trajectory: {actual}."
            ),
        }

    # Verify ordering: each agent must appear after the previous one
    ordered_positions = [positions[a] for a in expected_seq]
    is_ordered = all(ordered_positions[i] < ordered_positions[i + 1] for i in range(len(ordered_positions) - 1))

    if is_ordered:
        return {
            "score": 1.0,
            "comment": f"Correct sequence. Trajectory: {actual}.",
        }

    # Partial credit: how many consecutive pairs are in the right order
    correct_pairs = sum(
        1 for i in range(len(ordered_positions) - 1)
        if ordered_positions[i] < ordered_positions[i + 1]
    )
    score = correct_pairs / (len(ordered_positions) - 1)
    return {
        "score": score,
        "comment": (
            f"{correct_pairs}/{len(ordered_positions)-1} sequential pairs in correct order. "
            f"Expected: {expected_seq}. Actual trajectory: {actual}."
        ),
    }
