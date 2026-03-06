"""
Edge case examples for the shopping concierge eval.

These are harder cases added progressively as we optimize the prompt.
They test routing behavior that is ambiguous, multi-step, or requires
the agent to reason about exceptions and conflicts — not just happy-path routing.

Upload to a separate LangSmith dataset (not the train/val/test splits):
    uv run python src/eval/dataset_edge.py
    uv run python src/eval/dataset_edge.py --replace

Why separate?
    Core examples test that the agent routes correctly for clear-cut tasks.
    Edge cases are added one batch at a time as we optimize — they let us
    check whether prompt improvements generalize to harder inputs without
    contaminating the main eval splits.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "shopping-concierge-routing-edge"

# ---------------------------------------------------------------------------
# Edge cases
# Add new batches below as optimization progresses.
# Each case has a 'difficulty' tag explaining what makes it hard.
# ---------------------------------------------------------------------------

EDGE_EXAMPLES = [

    # ------------------------------------------------------------------
    # BATCH 1 — Policy exceptions the agent must know, not assume
    # ------------------------------------------------------------------

    {
        "inputs": {"query": "I want to return my Skims bodysuit", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "policy_exception",
            "difficulty_note": "Standard return logic says 30 days. Skims bodysuits are non-returnable. Agent must check policy, not assume.",
            "required_info": [
                "Skims Fits Everybody Bodysuit delivered 20 days ago",
                "Skims bodysuits are final sale — explicitly non-returnable",
                "cannot initiate return — must inform customer clearly",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Agent applies standard 30-day window without checking Skims-specific non-returnable rule.",
        },
    },
    {
        "inputs": {"query": "Return my MacBook — I bought it 3 weeks ago", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "policy_exception",
            "difficulty_note": "Customer states '3 weeks ago'. Electronics have a 14-day window. Agent must catch this is out-of-window.",
            "required_info": [
                "MacBook Pro delivered 22 days ago",
                "electronics return window is 14 days — not the standard 30",
                "not eligible — outside window, correct customer",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Agent applies 30-day standard window instead of the 14-day electronics rule.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 — Customer's own history changes the answer
    # ------------------------------------------------------------------

    {
        "inputs": {"query": "Is the Dyson worth buying? I'm thinking of getting one.", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "difficulty": "personal_history",
            "difficulty_note": "Sam previously bought AND returned the Dyson. Agent should surface this, not just give generic advice.",
            "required_info": [
                "Sam purchased and returned the Dyson V15 (35 days ago) — cited battery life issues",
                "external review context on the Dyson V15",
                "agent should flag Sam's own return history as highly relevant signal",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent gives generic recommendation without checking Sam's own return history.",
        },
    },
    {
        "inputs": {"query": "I want the same headphones Jordan ordered", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "cross_user_reference",
            "difficulty_note": "User references another user's order. Agent needs to look up Jordan's orders, not Alex's.",
            "required_info": [
                "Jordan's most recent headphone order: Sony WH-1000XM6 (ELEC-003)",
                "correct action: add ELEC-003 to Alex's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't look up Jordan's orders to identify the product.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 — Multi-step with 3+ agents
    # ------------------------------------------------------------------

    {
        "inputs": {
            "query": "Find me the best-reviewed jacket you carry, check if it's returnable, and add it to my cart",
            "user_email": "jordan@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "multi_step_4_agents",
            "difficulty_note": "Requires: catalog (reviews) → comparison (which is best) → policy (returnable?) → cart. 4-agent chain.",
            "required_info": [
                "Arc'teryx Beta AR has best internal review (5 stars)",
                "Arc'teryx has lifetime warranty — returnable within standard window",
                "correct action: add CLTH-003 to Jordan's cart",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no review data. Without policy: can't answer the return question. Without comparison: 'best-reviewed' is arbitrary.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 — Timing edge cases
    # ------------------------------------------------------------------

    {
        "inputs": {"query": "Can I still return my AirPods? I just got them delivered.", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "difficulty": "timing_edge",
            "difficulty_note": "AirPods are 'processing' — not delivered yet. Customer thinks they were just delivered. Agent must check actual status.",
            "required_info": [
                "AirPods Pro status is 'processing' — not yet delivered",
                "customer may be mistaken about delivery status",
                "cannot return — item not yet delivered per our records",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent takes customer's word ('just delivered') instead of checking actual order status.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 — Conflicting intent
    # ------------------------------------------------------------------

    {
        "inputs": {
            "query": "I want to cancel my Aritzia Super Puff order and get the Arc'teryx instead",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "conflicting_intent",
            "difficulty_note": "Two actions: cancel current order + add different item. Agent must handle both and check cancellation policy.",
            "required_info": [
                "Aritzia Super Puff is 'processing' — can be cancelled before shipment",
                "Arc'teryx Beta AR SKU is CLTH-003, price $799",
                "cancel the Aritzia order and add Arc'teryx to cart",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Agent only handles one of the two actions, or tries to cancel a shipped item.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 — Price match / refund for price drop
    # ------------------------------------------------------------------

    {
        "inputs": {
            "query": "The Dyson dropped in price since I bought it — can I get a refund for the difference?",
            "user_email": "sam@example.com",
        },
        "outputs": {
            "task_type": "return_eligibility",
            "difficulty": "price_match_policy",
            "difficulty_note": "Sam returned the Dyson already. Agent needs to find this AND know the price match policy.",
            "required_info": [
                "Sam already returned the Dyson 35 days ago — refund was already processed",
                "price match policy: we honor price adjustments within 14 days of purchase",
                "moot point — Dyson was already returned",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent answers the price match question without first checking that Sam already returned the item.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 — Ambiguous product reference
    # ------------------------------------------------------------------

    {
        "inputs": {"query": "Add the candle I always buy to my cart", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "ambiguous_reference",
            "difficulty_note": "Mia has ordered Diptyque Baies before. Agent must look up her order history to identify 'the candle she always buys'.",
            "required_info": [
                "Mia's order history includes Diptyque Baies (BEAU-001) from 6 days ago",
                "Baies is the candle she has purchased",
                "correct action: add BEAU-001 to Mia's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: agent guesses which candle or asks unnecessarily when order history has the answer.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 — External info needed to answer internal question
    # ------------------------------------------------------------------

    {
        "inputs": {
            "query": "Is the Arc'teryx Beta AR jacket worth $799 — I see it for less at REI?",
            "user_email": "jordan@example.com",
        },
        "outputs": {
            "task_type": "price_assessment",
            "difficulty": "price_context_conflict",
            "difficulty_note": "Customer claims REI is cheaper. Agent must verify our price AND get real REI price, not assume customer is right.",
            "required_info": [
                "our Arc'teryx Beta AR price: $799 (no historical price changes)",
                "REI current price for Arc'teryx Beta AR from web",
                "comparison: confirm or refute customer's claim",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent accepts customer's claim about REI price without verifying either our price or the competitor price.",
        },
    },
]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_edge(client: Client, examples: list[dict], replace: bool = False) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        if not replace:
            print(f"  '{DATASET_NAME}' already exists — skipping. Pass --replace to overwrite.")
            return
        client.delete_dataset(dataset_name=DATASET_NAME)
        print(f"  Deleted existing '{DATASET_NAME}'")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Shopping concierge routing — edge cases, added progressively during optimization",
    )
    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id,
    )
    print(f"  Uploaded {len(examples)} edge examples → '{DATASET_NAME}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="Delete and re-upload")
    args = parser.parse_args()

    print(f"\n{len(EDGE_EXAMPLES)} edge examples\n")
    client = Client()
    upload_edge(client, EDGE_EXAMPLES, replace=args.replace)
    print("\nDone.")
