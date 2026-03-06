"""
Routing eval dataset for the shopping concierge.

Every example is designed so the task genuinely cannot be completed correctly
without the full required trajectory. Each example specifies:

  required_info   — what the answer MUST contain to be correct
  expected_agents — which agents provide each piece of required_info
  cannot_complete_without — subset of agents whose absence makes the task impossible

This means evaluators can check answer quality, not just tool call presence.

Run once to upload to LangSmith:
    uv run python src/eval/dataset.py
"""

import json
import random
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "eval"
DATASET_NAME = "shopping-concierge-routing"

# ---------------------------------------------------------------------------
# Examples
#
# Each example documents WHY the trajectory is required:
#   - required_info: specific facts the answer must contain
#   - expected_agents: which agent supplies each fact
#   - cannot_complete_without: agents whose absence makes the answer wrong/incomplete
#   - failure_mode: what the baseline does wrong
# ---------------------------------------------------------------------------

EXAMPLES = [
    # ------------------------------------------------------------------
    # PRICE ASSESSMENT
    # Requires internal price history + external competitor prices.
    # Neither alone answers "is this a good deal" — you need both.
    # ------------------------------------------------------------------
    {
        "inputs": {
            "query": "Is the Sony WH-1000XM6 a good deal right now?",
            "user_email": "jordan@example.com",
        },
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "internal price history showing it dropped from $399 to $349",
                "current competitor/market price for comparison",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Baseline calls only product_catalog_agent — gives price history but no market context. Or calls only web_research_agent — gives competitor prices but no internal history. Neither alone answers whether it's a good deal.",
        },
    },
    {
        "inputs": {
            "query": "Is our price on the MacBook Pro competitive with other stores?",
            "user_email": "alex@example.com",
        },
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "our current price for the MacBook Pro ($1599)",
                "prices at other major retailers for comparison",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without product_catalog_agent, doesn't know our price. Without web_research_agent, can't compare. A hallucinated answer with only one side is confidently wrong.",
        },
    },

    # ------------------------------------------------------------------
    # RETURN ELIGIBILITY
    # Requires order lookup (actual date + status) + policy (window + conditions).
    # Policy alone can't confirm eligibility without knowing the actual order date.
    # Order data alone doesn't tell you what the return conditions are.
    # ------------------------------------------------------------------
    {
        "inputs": {
            "query": "Can I return the Alo leggings I ordered?",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "order status is 'delivered' and order was placed 6 days ago",
                "return policy for activewear is 30 days, unworn with tags",
                "correct eligibility conclusion: yes, within window",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent"],
            "failure_mode": "Calling only policy_and_sizing_agent gives the policy but not the order date — can't confirm eligibility. Calling only product_catalog_agent gives the date but not the policy window.",
        },
    },
    {
        "inputs": {
            "query": "Can I still return the leggings from 2 weeks ago? It's been a while.",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "actual order date (14 days ago — delivered)",
                "return window is 30 days for activewear",
                "correct conclusion: still eligible, 16 days remaining",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent"],
            "failure_mode": "Baseline checks policy without looking up the actual order date — gives a policy answer but can't confirm '2 weeks' is still within the window for this specific order.",
        },
    },
    {
        "inputs": {
            "query": "What's the warranty on the Arc'teryx jacket I ordered, and how do I make a claim?",
            "user_email": "jordan@example.com",
        },
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "confirm Jordan has the Arc'teryx Beta AR jacket in their order history",
                "Arc'teryx has a lifetime warranty against defects",
                "claim process: warranty.arcteryx.com with photos",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without product_catalog_agent, can't confirm they actually own it. Without policy_and_sizing_agent, can't give accurate warranty length or claims process.",
        },
    },

    # ------------------------------------------------------------------
    # PRODUCT COMPARISON
    # Requires fetching specs from catalog + running structured comparison.
    # Catalog alone returns raw specs; comparison agent formats and reasons over them.
    # ------------------------------------------------------------------
    {
        "inputs": {
            "query": "Compare the MacBook Pro 14 and the Dell XPS 13 — which is better value?",
            "user_email": "alex@example.com",
        },
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "specs for both laptops (storage, memory, display, battery)",
                "prices for both ($1599 MacBook, $1299 Dell)",
                "a value judgment supported by the spec comparison",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": ["product_catalog_agent", "product_comparison_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Without product_catalog_agent, comparison agent has no data to compare. Without product_comparison_agent, the answer is just a raw spec dump with no structured analysis.",
        },
    },
    {
        "inputs": {
            "query": "Show me all the running shoes you carry and rank them by value",
            "user_email": "jordan@example.com",
        },
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "all running shoes in catalog with prices (Ultraboost $190, NB 990v6 $185, Tree Runner $128)",
                "a ranked comparison explaining value for each",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": ["product_catalog_agent", "product_comparison_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Without catalog lookup, comparison agent works with incomplete or hallucinated product list. Without comparison agent, the answer is a raw list with no value analysis.",
        },
    },

    # ------------------------------------------------------------------
    # DISCOVERY + EXTERNAL VALIDATION
    # Requires semantic discovery (vibe match) + web validation (real opinions).
    # Semantic search finds candidates; web search validates with real reviews.
    # ------------------------------------------------------------------
    {
        "inputs": {
            "query": "Find me a good outdoor jacket under $300 — what do reviewers say about the top options?",
            "user_email": "alex@example.com",
        },
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "catalog candidates for outdoor jacket under $300 (Patagonia Nano Puff $249, Uniqlo $89)",
                "external review sentiment or expert opinion on the top candidates",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Web search alone returns general results without knowing our catalog. Discovery alone returns products without any external validation. The customer asked for both.",
        },
    },
    {
        "inputs": {
            "query": "Is the Dyson worth $749? What do people actually think of it?",
            "user_email": "sam@example.com",
        },
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "internal customer reviews of the Dyson V15 (mixed — Sam returned it, noted battery life issues)",
                "external reviews or expert opinion from web",
                "price context ($749 is standard retail price, no price drops)",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without product_catalog_agent, misses internal reviews and Sam's own return history. Without web_research_agent, no external opinion. Combined they give a complete picture.",
        },
    },

    # ------------------------------------------------------------------
    # SIZING WITH PRODUCT CONTEXT
    # Requires knowing what the customer ordered + sizing guidance for that brand.
    # Sizing advice without knowing the product is generic and potentially wrong.
    # ------------------------------------------------------------------
    {
        "inputs": {
            "query": "I just ordered a Skims bodysuit — what size should I have gotten? Did I pick right?",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Mia's Skims order (Fits Everybody Bodysuit, ordered 2x in size from her order history)",
                "Skims Fits Everybody sizing: true to size, stretches 3x",
                "verdict on whether her size choice was correct given her measurements/preferences",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without order lookup, the agent doesn't know what she ordered or what size. Without sizing docs, can't give Skims-specific guidance.",
        },
    },
    {
        "inputs": {
            "query": "I'm thinking of getting the Aritzia Super Puff — what size given I normally wear XS but want it roomy?",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Aritzia Super Puff sizing: runs true to size with relaxed boxy fit",
                "for roomy fit: TTS is already generous, could go XS or 2XS",
                "size chart reference for Mia's XS frame",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy_and_sizing_agent, the agent gives generic sizing advice without the Aritzia-specific guidance that it runs generous.",
        },
    },

    # ------------------------------------------------------------------
    # SEQUENTIAL — action requires prior lookup
    # The action tool needs specific IDs from the lookup. Without the lookup,
    # cart_and_orders_agent cannot be called with correct arguments.
    # ------------------------------------------------------------------
    {
        "inputs": {
            "query": "I want to return the jacket I ordered. Can I, and if so please start the return.",
            "user_email": "jordan@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Jordan's jacket order (Arc'teryx Beta AR, order_id and sku from DB)",
                "return eligibility: shipped status — needs to be delivered first",
                "correct outcome: cannot initiate return yet, explain why",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Without product_catalog_agent first, cart_and_orders_agent has no order_id or sku to call initiate_return with — it would hallucinate IDs or fail.",
        },
    },
    {
        "inputs": {
            "query": "Buy me the cheaper of the two Diptyque candles",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "both Diptyque candles are $75 (same price — they're both $75)",
                "correct response: inform they're the same price, ask which she prefers",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without lookup, agent either hallucinates which is cheaper or adds one arbitrarily. The correct answer (they're the same price) requires fetching actual prices first.",
        },
    },
    {
        "inputs": {
            "query": "Add the best-reviewed headphones you carry to my cart",
            "user_email": "jordan@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "internal reviews: Sony XM6 has 4.5 avg, AirPods Pro also reviewed positively",
                "Sony WH-1000XM6 is the best-reviewed (highest avg rating)",
                "correct action: add Sony XM6 (ELEC-003) to cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without checking reviews first, agent adds an arbitrary or hallucinated 'best' headphone. The SKU passed to add_to_cart must be grounded in actual review data.",
        },
    },
    {
        "inputs": {
            "query": "Return my most recent delivered order",
            "user_email": "mia@example.com",
        },
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Mia's most recent delivered order: Alo leggings + Diptyque Baies (order placed 6 days ago)",
                "policy check: candles cannot be returned if opened; leggings can within 30 days",
                "correct action: can only return the leggings — Baies is a candle (14-day unopened only)",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Without order lookup: no order_id/sku. Without policy check: might attempt to return the candle, which would fail or be incorrect.",
        },
    },
]


# ---------------------------------------------------------------------------
# Split + export
# ---------------------------------------------------------------------------

def split_examples(examples: list[dict], train: float = 0.6, val: float = 0.2) -> tuple:
    random.seed(42)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples → {path.name}")


def upload_split(client: Client, examples: list[dict], split: str) -> None:
    name = f"{DATASET_NAME}-{split}"
    if client.has_dataset(dataset_name=name):
        print(f"  '{name}' already exists — skipping.")
        return
    dataset = client.create_dataset(
        dataset_name=name,
        description=f"Shopping concierge routing eval — {split} split",
    )
    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id,
    )
    print(f"  Uploaded {len(examples)} examples → '{name}'")


if __name__ == "__main__":
    train, val, test = split_examples(EXAMPLES)
    print(f"\n{len(EXAMPLES)} examples → {len(train)} train / {len(val)} val / {len(test)} test\n")

    print("Writing JSONL...")
    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(val, DATA_DIR / "val.jsonl")
    write_jsonl(test, DATA_DIR / "test.jsonl")

    print("\nUploading to LangSmith...")
    client = Client()
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        upload_split(client, split_data, split_name)

    print("\nDone.")
