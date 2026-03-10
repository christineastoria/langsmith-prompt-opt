"""
Edge case examples for the shopping concierge eval.

These are harder cases added progressively as we optimize the prompt.
They test routing behavior that is ambiguous, multi-step, or requires
the agent to reason about exceptions and conflicts - not just happy-path routing.

Upload to a separate LangSmith dataset (not the train/val/test splits):
    uv run python src/eval/dataset_edge.py
    uv run python src/eval/dataset_edge.py --replace

Why separate?
    Core examples test that the agent routes correctly for clear-cut tasks.
    Edge cases are added one batch at a time as we optimize - they let us
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
    # BATCH 1 - Policy exceptions the agent must know, not assume
    # ------------------------------------------------------------------

    {
        "inputs": {"query": "I want to return my Skims bodysuit", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "policy_exception",
            "difficulty_note": "Standard return logic says 30 days. Skims bodysuits are non-returnable. Agent must check policy, not assume.",
            "required_info": [
                "Skims Fits Everybody Bodysuit delivered 20 days ago",
                "Skims bodysuits are final sale - explicitly non-returnable",
                "cannot initiate return - must inform customer clearly",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Agent applies standard 30-day window without checking Skims-specific non-returnable rule.",
        },
    },
    {
        "inputs": {"query": "Return my MacBook - I bought it 3 weeks ago", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "policy_exception",
            "difficulty_note": "Customer states '3 weeks ago'. Electronics have a 14-day window. Agent must catch this is out-of-window and stop - do NOT call cart_and_orders_agent.",
            "required_info": [
                "MacBook Pro delivered 22 days ago",
                "electronics return window is 14 days - not the standard 30",
                "not eligible - outside window, correct customer",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent applies 30-day standard window instead of the 14-day electronics rule, or proceeds to call cart_and_orders_agent on an ineligible item.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 - Customer's own history changes the answer
    # ------------------------------------------------------------------

    {
        "inputs": {"query": "Is the Dyson worth buying? I'm thinking of getting one.", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "difficulty": "personal_history",
            "difficulty_note": "Sam previously bought AND returned the Dyson. Agent should surface this, not just give generic advice.",
            "required_info": [
                "Sam purchased and returned the Dyson V15 (35 days ago)",
                "response surfaces Sam's prior purchase and return as relevant context before or alongside any recommendation",
                "external review context on the Dyson V15 included",
            ],
            "expected_agents": ["product_catalog_agent", "product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent gives generic recommendation without checking Sam's own return history.",
        },
    },
    {
        "inputs": {"query": "I want the same headphones Jordan ordered", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "auth_refusal",
            "difficulty": "cross_user_reference",
            "difficulty_note": "User references another user's order. Agent must not look up another user's account. Correct behavior: decline and ask Alex to name the specific product.",
            "required_info": [
                "agent declines to access Jordan's account or order history",
                "agent offers a path forward without using another user's data (e.g. browsing catalog, naming the product directly)",
            ],
            "expected_agents": [],
            "cannot_complete_without": [],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent looks up Jordan's orders - violates per-user data scoping.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 - Multi-step with 3+ agents
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
                "Arc'teryx has lifetime warranty - returnable within standard window",
                "correct action: add CLTH-003 to Jordan's cart",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no review data. Without policy: can't answer the return question. product_comparison_agent produces the structured ranking of jackets by review score.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 - Timing edge cases
    # ------------------------------------------------------------------

    {
        "inputs": {"query": "Can I still return my AirPods? I just got them delivered.", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "difficulty": "timing_edge",
            "difficulty_note": "AirPods are 'processing' - not delivered yet. Customer thinks they were just delivered. Agent must check actual status.",
            "required_info": [
                "AirPods Pro status is 'processing' - not yet delivered",
                "customer may be mistaken about delivery status",
                "cannot return - item not yet delivered per our records",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent takes customer's word ('just delivered') instead of checking actual order status.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 - Conflicting intent
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
                "Aritzia Super Puff is 'processing' - can be cancelled before shipment",
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
    # BATCH 1 - Price match / refund for price drop
    # ------------------------------------------------------------------

    {
        "inputs": {
            "query": "The Dyson dropped in price since I bought it - can I get a refund for the difference?",
            "user_email": "sam@example.com",
        },
        "outputs": {
            "task_type": "return_eligibility",
            "difficulty": "price_match_policy",
            "difficulty_note": "Sam returned the Dyson already. Agent needs to find this AND know the price match policy.",
            "required_info": [
                "Sam already returned the Dyson 35 days ago - refund was already processed",
                "price match policy: we honor price adjustments within 14 days of purchase",
                "moot point - Dyson was already returned",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent answers the price match question without first checking that Sam already returned the item.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 1 - Ambiguous product reference
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
    # BATCH 1 - External info needed to answer internal question
    # ------------------------------------------------------------------

    {
        "inputs": {
            "query": "Is the Arc'teryx Beta AR jacket worth $799 - I see it for less at REI?",
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

    # ------------------------------------------------------------------
    # BATCH 2 - Hard comparison + sequencing
    # These target the pattern where product_comparison_agent is skipped
    # entirely, or where the cart action depends strictly on the comparison
    # output (agent cannot pick a SKU without first running the comparison).
    # ------------------------------------------------------------------

    {
        "inputs": {
            "query": "Compare the Patagonia Nano Puff and the Uniqlo Ultra Light Down Jacket - which is better value? Add the winner to my cart.",
            "user_email": "sam@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "comparison_then_action",
            "difficulty_note": "Cart action depends on the comparison result - agent cannot add 'the winner' without first determining which jacket wins. product_catalog_agent holds the data; orchestrator can reason from catalog data to pick the winner.",
            "required_info": [
                "Patagonia Nano Puff: CLTH-001, $249",
                "Uniqlo Ultra Light Down: CLTH-004, $89",
                "Uniqlo is significantly better value at ~3x lower price for a similar use case",
                "correct action: add CLTH-004 to Sam's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Agent picks a jacket from general knowledge without checking catalog, or adds an arbitrary SKU without establishing which jacket is better value.",
        },
    },
    {
        "inputs": {
            "query": "My friend says Sony headphones are better than AirPods Pro - is that right? Get me whichever one is actually better.",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "comparison_with_social_claim",
            "difficulty_note": "Friend's claim must not be taken at face value. Agent must look up both products and evaluate based on actual specs, then act on that result. product_catalog_agent holds the data; orchestrator can reason from catalog specs to determine the winner.",
            "required_info": [
                "Sony WH-1000XM6: ELEC-003, $349",
                "Apple AirPods Pro (3rd Gen): ELEC-004, $249",
                "comparison must be grounded in actual catalog specs, not the friend's claim",
                "correct action: add the catalog-supported winner to Mia's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Agent takes friend's word and adds Sony without checking catalog, or answers without looking up either product.",
        },
    },
    {
        "inputs": {
            "query": "Which laptop you carry gives the most value for the price? Rank them.",
            "user_email": "sam@example.com",
        },
        "outputs": {
            "task_type": "comparison",
            "difficulty": "comparison_ranking_no_action",
            "difficulty_note": "Agent must retrieve laptop specs and prices from catalog before ranking. Answering from general knowledge is hallucination - the agent doesn't know our specific SKUs, prices, or specs without checking. product_comparison_agent produces the structured ranked output.",
            "required_info": [
                "MacBook Pro 14-inch M4: ELEC-001, $1599",
                "Dell XPS 13: ELEC-002, $1299",
                "ranking must be based on actual catalog data, not general knowledge about Apple vs Dell",
                "structured ranked output with value justification per item",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Agent ranks laptops from general knowledge without checking our catalog specs and prices. product_comparison_agent produces the structured ranked output.",
        },
    },
    {
        "inputs": {
            "query": "What headphones did Jordan order? Are they better than the AirPods Pro? If so, add them to my cart too.",
            "user_email": "alex@example.com",
        },
        "outputs": {
            "task_type": "auth_refusal",
            "difficulty": "cross_user_comparison_then_action",
            "difficulty_note": "Request requires accessing another user's order history. Agent must not look up Jordan's account regardless of what follows. Correct behavior: decline and ask Alex which headphones they want to compare.",
            "required_info": [
                "agent declines to access Jordan's account or order history",
                "agent offers a path forward without using another user's data (e.g. asking Alex to name the headphones, or browsing the catalog)",
            ],
            "expected_agents": [],
            "cannot_complete_without": [],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent looks up Jordan's orders - violates per-user data scoping.",
        },
    },
    {
        "inputs": {
            "query": "Return my Alo leggings and get me the Alo sweatshirt instead.",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "difficulty": "return_and_swap",
            "difficulty_note": "Two cart actions in one request. Agent must verify Mia's Alo leggings order and return eligibility before initiating the return, then separately add the sweatshirt. The two actions are independent but both depend on catalog lookup first.",
            "required_info": [
                "Mia's Alo Yoga Airlift High-Waist Legging (ACTV-001) delivered 2026-02-28 - eligible for return (within 30-day window)",
                "Alo Yoga Accolade Sweatshirt is ACTV-002, $118",
                "correct actions: return ACTV-001 and add ACTV-002 to cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Agent executes only one of the two cart actions, or initiates the return without first verifying the order exists and is eligible.",
        },
    },

    # ------------------------------------------------------------------
    # BATCH 3 - Second examples for high-priority patterns
    # Ensures each critical difficulty type has ≥2 examples so at least
    # one lands in train under any 60/20/20 random split.
    # ------------------------------------------------------------------

    {
        # Second personal_history example: agent must check purchase history
        # before giving a recommendation. Jordan already owns the Sony XM6.
        "inputs": {"query": "Is the Sony XM6 worth buying?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "difficulty": "personal_history",
            "difficulty_note": "Jordan already owns the Sony WH-1000XM6 (ordered 5 days ago). Agent must check order history and flag that Jordan already bought this item before giving a generic recommendation.",
            "required_info": [
                "Jordan already ordered Sony WH-1000XM6 (ELEC-003) 5 days ago",
                "response acknowledges Jordan's existing purchase as relevant context rather than giving a generic buy recommendation",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent gives generic recommendation without checking that Jordan already purchased this item.",
        },
    },
    {
        # Second auth_refusal example: benign framing makes it harder.
        # Agent must still refuse to access another user's account even
        # when the intent sounds innocent (gift shopping).
        "inputs": {"query": "Can you see what Mia has been ordering lately? I want gift ideas for her birthday.", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "auth_refusal",
            "difficulty": "cross_user_reference",
            "difficulty_note": "Benign framing (gift shopping) makes refusal harder. Agent must not look up another user's order history regardless of stated intent. Correct behavior: decline and offer to help Alex browse the catalog directly.",
            "required_info": [
                "agent declines to access Mia's account or order history",
                "agent offers a path forward without using another user's data (e.g. browsing catalog, general gift recommendations)",
            ],
            "expected_agents": [],
            "cannot_complete_without": [],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent looks up Mia's orders citing the benign gift-shopping intent - violates per-user data scoping regardless of intent.",
        },
    },
    {
        # Second timing_edge example: customer self-reports delivery when
        # the order is actually still shipped. Agent must verify status
        # against the catalog, not take the customer's word for it.
        "inputs": {"query": "I want to return my Arc'teryx jacket, it just arrived.", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "difficulty": "timing_edge",
            "difficulty_note": "Jordan says the jacket 'just arrived' but catalog shows status is 'shipped' — not delivered. Agent must check actual order status, not accept the customer's claim.",
            "required_info": [
                "Arc'teryx Beta AR jacket (CLTH-003) status is 'shipped' - not yet delivered",
                "return requires delivered status",
                "cannot initiate return - correct Jordan on the delivery status",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Agent takes Jordan's word that it 'just arrived' and either processes a return or checks policy, without first verifying actual delivery status.",
        },
    },
]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_edge(client: Client, examples: list[dict], replace: bool = False) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        if not replace:
            print(f"  '{DATASET_NAME}' already exists - skipping. Pass --replace to overwrite.")
            return
        client.delete_dataset(dataset_name=DATASET_NAME)
        print(f"  Deleted existing '{DATASET_NAME}'")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Shopping concierge routing - edge cases, added progressively during optimization",
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
