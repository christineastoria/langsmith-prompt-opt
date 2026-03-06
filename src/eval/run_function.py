"""
Run function for the shopping concierge routing eval.

Runs a query through the concierge and captures which subagents were called.
Follows the LangSmith evaluator skill's "inspect before implementing" principle —
set DEBUG=true to print raw chunk structure before relying on trajectory extraction.

Usage:
    uv run python src/eval/run_function.py  # smoke test with debug output
"""

import os
import uuid
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def make_run_fn(prompt_variant: str = "baseline"):
    """
    Returns a run function bound to a specific prompt variant.
    Pass prompt_variant="optimized" to evaluate the optimised prompt.
    """
    # Import here so the module can be imported without side effects
    from agents.shopping_assistant import create_concierge, get_user_context

    agent = create_concierge(prompt_variant=prompt_variant)

    def run_concierge(inputs: dict) -> dict:
        query: str = inputs["query"]
        user_email: str = inputs["user_email"]
        thread_id = f"eval-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        user_context = get_user_context(user_email)
        messages = [
            {"role": "system", "content": user_context},
            {"role": "user", "content": query},
        ]

        agents_called: list[str] = []
        final_output: str = ""

        # Stream with debug mode to capture task tool calls
        for chunk in agent.stream(
            {"messages": messages},
            config=config,
            stream_mode="debug",
            subgraphs=True,
        ):
            if DEBUG:
                print(f"DEBUG chunk type={type(chunk).__name__}: {str(chunk)[:300]}")

            # chunk is (namespace, event_dict) when subgraphs=True
            if isinstance(chunk, tuple):
                _, event = chunk
            else:
                event = chunk

            # Extract subagent calls and final output from top-level events only
            # (namespace == () means top-level graph, not a subgraph)
            if isinstance(chunk, tuple) and chunk[0] != ():
                continue

            if isinstance(event, dict):
                event_type = event.get("type", "")
                payload = event.get("payload", {})

                # task events with tool_call_with_context carry the subagent name
                if event_type == "task" and payload.get("name") == "tools":
                    input_data = payload.get("input", {})
                    if input_data.get("__type") == "tool_call_with_context":
                        tc = input_data.get("tool_call", {})
                        if tc.get("name") == "task":
                            agent_name = tc.get("args", {}).get("subagent_type")
                            if agent_name and agent_name not in agents_called:
                                agents_called.append(agent_name)

                # task_result from the model node = final answer
                if event_type == "task_result" and payload.get("name") == "model":
                    result_data = payload.get("result", {})
                    for msg in result_data.get("messages", []):
                        if hasattr(msg, "content") and isinstance(msg.content, str):
                            final_output = msg.content

        if DEBUG:
            print(f"\nDEBUG agents_called: {agents_called}")
            print(f"DEBUG final_output[:200]: {final_output[:200]}")

        return {
            "agents_called": agents_called,
            "final_output": final_output,
            "query": query,
            "user_email": user_email,
        }

    return run_concierge


# ---------------------------------------------------------------------------
# Smoke test — run directly to inspect output structure
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["DEBUG"] = "true"

    run_fn = make_run_fn("baseline")
    result = run_fn({
        "query": "Can I return the Alo leggings I ordered?",
        "user_email": "mia@example.com",
    })

    print("\n--- Final result ---")
    print(f"agents_called: {result['agents_called']}")
    print(f"final_output: {result['final_output'][:300]}")
