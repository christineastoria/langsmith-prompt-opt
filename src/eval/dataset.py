"""
Core routing eval dataset for the shopping concierge.

What we measure and why
-----------------------
task_completeness (LLM judge)
    Did the final answer contain the required_info? This catches hallucination:
    the agent may call the right agents but still give a wrong answer.

critical_agents_called (code)
    Were the agents in cannot_complete_without actually called?
    These are the agents that HOLD the ground truth - without them, any
    correct-sounding answer is hallucinated.

sequence_respected (code)
    For action tasks: did the catalog lookup happen before the cart action?
    The cart_and_orders_agent needs real order_id/sku from the catalog to work.
    requires_sequencing is True ONLY for action_with_prerequisite examples.

Sequencing rules
----------------
- requires_sequencing: True ONLY when cart action depends on catalog output
  (needs actual order_id or sku to call initiate_return / add_to_cart)
- Everything else is False - parallel independent lookups don't have a
  required order even if both are needed

Run to upload (replaces existing datasets):
    uv run python src/eval/dataset.py
    uv run python src/eval/dataset.py --replace   # delete and re-upload
"""

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "eval"
DATASET_NAME = "shopping-concierge-routing"

# ---------------------------------------------------------------------------
# DB reference (from setup_db.py)
# ---------------------------------------------------------------------------
# Users:
#   alex@example.com   - delivered MacBook (22 days), processing AirPods (2 days)
#   jordan@example.com - delivered Sony XM6 (5 days), shipped Arc'teryx (8 days)
#   sam@example.com    - delivered Allbirds+Uniqlo (14 days), returned Dyson (35 days)
#   mia@example.com    - delivered Alo leggings+Diptyque Baies (6 days),
#                        delivered Skims bodysuit x2 (20 days),
#                        processing Aritzia Super Puff (1 day)
#
# Key prices: MacBook $1599, Dell XPS $1299, iPad Pro $1099, Sony XM6 $349,
#   AirPods Pro $249, Samsung TV $899, Dyson $749, Instant Pot $99, YETI $45,
#   Patagonia $249, Arc'teryx $799, Uniqlo Down $89, Levi's 501 $79,
#   Alo leggings $128, Alo sweatshirt $118, Skims bodysuit $62, Skims dress $88,
#   Aritzia Super Puff $325, Diptyque Baies $75, Diptyque Feu de Bois $75,
#   Rhode lip treatment $20, Drunk Elephant $68, Glossier You $72,
#   Nike Air Max $150, Ultraboost $190, Allbirds $128, Timberland $230, NB 990v6 $185
#
# Return policy: standard 30 days unworn; electronics 30 days (Apple products: 14 days
#   for opened); candles 14 days if unopened; beauty/skincare no returns once opened;
#   Skims bodysuits non-returnable (final sale)
# ---------------------------------------------------------------------------

EXAMPLES = [

    # ================================================================
    # PRICE ASSESSMENT
    # cannot_complete_without catalog for internal price data.
    # Add web_research_agent when the question requires market comparison.
    # requires_sequencing: False (independent lookups)
    # ================================================================

    {
        "inputs": {"query": "Did the Sony XM6 price drop recently?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": ["Sony WH-1000XM6 dropped from $399 to $349 over the past 3 months"],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: agent hallucinates price history.",
        },
    },
    {
        "inputs": {"query": "What's the current price on the MacBook Pro?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": ["MacBook Pro 14-inch M4 is currently $1599"],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: agent hallucinates current price.",
        },
    },
    {
        "inputs": {"query": "Has the Dyson ever been on sale here?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": ["Dyson V15 dropped from $799 to $749"],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no access to internal price history.",
        },
    },
    {
        "inputs": {"query": "Was the Nike Air Max ever cheaper than it is now?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": ["Nike Air Max 270 was $120 on sale ~45 days ago, now back to $150"],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: can't see internal price history.",
        },
    },
    {
        "inputs": {"query": "Is the Sony WH-1000XM6 a good deal right now?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "internal price history: dropped from $399 to $349",
                "current competitor/market price for comparison",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Catalog alone can't confirm if $349 is competitive. Web alone doesn't know our price history.",
        },
    },
    {
        "inputs": {"query": "Is our MacBook Pro price competitive with other stores?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "our MacBook Pro 14 M4 price ($1599)",
                "prices at other major retailers for comparison",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know our price. Without web: can't compare.",
        },
    },
    {
        "inputs": {"query": "How does the Sony XM6 compare to Bose QuietComfort in price?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "Sony WH-1000XM6 price ($349)",
                "Bose QuietComfort price from web for comparison",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: might hallucinate Sony price. Without web: no Bose price.",
        },
    },
    {
        "inputs": {"query": "Is the iPad Pro worth buying here vs the Apple Store?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "our iPad Pro 13-inch M4 price ($1099)",
                "Apple Store / competitor pricing for the same model",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know our price. Without web: can't compare to Apple Store.",
        },
    },
    {
        "inputs": {"query": "Is the Dyson worth $749 or can I get it cheaper somewhere else?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "our Dyson V15 price ($749, dropped from $799)",
                "Dyson V15 prices at other retailers",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: can't confirm our price or show the drop. Without web: no competitor context.",
        },
    },
    {
        "inputs": {"query": "Is the Arc'teryx at $799 fair? What does it go for at REI?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "price_assessment",
            "required_info": [
                "our Arc'teryx Beta AR price ($799, no change in price history)",
                "REI / competitor pricing for Arc'teryx Beta AR",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: might hallucinate our price. Without web: no REI context.",
        },
    },

    # ================================================================
    # RETURN ELIGIBILITY
    # Both catalog (order date/status) + policy (window + conditions).
    # Neither depends on the other - requires_sequencing: False.
    # Policy-only examples included for general questions.
    # ================================================================

    {
        "inputs": {"query": "Can I return the Alo leggings I ordered?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Alo leggings delivered 6 days ago",
                "activewear return window is 30 days, unworn with tags",
                "eligible - within window",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no order date to verify window. Without policy: no return conditions.",
        },
    },
    {
        "inputs": {"query": "Can I still return the leggings? It's been about 2 weeks.", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Alo leggings delivered 6 days ago (not 14 - correct the customer)",
                "30-day activewear window",
                "eligible - 24 days remaining",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: agent accepts '2 weeks' as fact instead of checking actual date.",
        },
    },
    {
        "inputs": {"query": "Can I return the MacBook I bought here?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "MacBook delivered 22 days ago",
                "electronics return window is 14 days",
                "not eligible - outside 14-day electronics window",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no order date. Without policy: might assume 30-day standard window instead of 14-day electronics rule.",
        },
    },
    {
        "inputs": {"query": "Can I return the Arc'teryx jacket?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Arc'teryx jacket status is 'shipped' - not yet delivered",
                "return policy requires delivered status before a return can be started",
                "cannot return yet - not delivered",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know status is 'shipped'. Without policy: doesn't know delivery is required.",
        },
    },
    {
        "inputs": {"query": "Can I return my Allbirds sneakers?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Allbirds Tree Runner delivered 14 days ago",
                "footwear return window is 30 days, unworn",
                "eligible - 16 days remaining",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no delivery date. Without policy: no window to verify eligibility.",
        },
    },
    {
        "inputs": {"query": "Can I return the Uniqlo jacket?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Uniqlo Ultra Light Down delivered 14 days ago",
                "clothing return window is 30 days, unworn with tags",
                "eligible - 16 days remaining",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no order date. Without policy: wrong return window.",
        },
    },
    {
        "inputs": {"query": "Can I return the Diptyque candle I ordered?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Diptyque Baies delivered 6 days ago",
                "candles: 14-day window, only if unopened",
                "eligible IF unopened - within 14-day candle window",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no delivery date. Without policy: misses candle-specific 14-day rule.",
        },
    },
    {
        "inputs": {"query": "How long do I have to return my Sony headphones?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Sony XM6 delivered 5 days ago",
                "electronics return window is 30 days",
                "25 days remaining to return",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no delivery date to calculate remaining time. Without policy: wrong window.",
        },
    },
    {
        "inputs": {"query": "I want to return my AirPods - can I?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "AirPods Pro status is 'processing' - not yet shipped or delivered",
                "return policy requires delivered status",
                "cannot return yet - order not delivered",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know status is 'processing'. Without policy: doesn't know delivery is required.",
        },
    },
    {
        "inputs": {"query": "Can I return the Skims bodysuit?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Skims Fits Everybody Bodysuit delivered 20 days ago",
                "Skims bodysuits are final sale - non-returnable",
                "not eligible - Skims bodysuits cannot be returned",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't confirm she has it. Without policy: misses the non-returnable rule.",
        },
    },
    {
        "inputs": {"query": "Can I return the Aritzia Super Puff I just ordered?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Aritzia Super Puff status is 'processing' - just ordered 1 day ago",
                "return policy requires delivered status",
                "cannot return yet - not delivered",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know it's still processing. Without policy: might not know delivery is required.",
        },
    },
    {
        "inputs": {"query": "I got the wrong size Alo leggings - can I exchange them?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Alo Airlift leggings delivered 6 days ago",
                "exchanges treated same as returns: 30 days, unworn with tags",
                "eligible for exchange - within window",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no order date. Without policy: no exchange terms.",
        },
    },
    {
        "inputs": {"query": "What's the return policy for electronics?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "electronics return window is 30 days from delivery",
                "must be unopened OR opened but unused, with all original accessories included",
                "Apple products (MacBook, iPhone, AirPods) are 14 days for opened items",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: agent gives incorrect or hallucinated return window.",
        },
    },
    {
        "inputs": {"query": "Is my Dyson return fully processed?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Dyson V15 return was processed 35 days ago - status is 'returned'",
                "return is complete",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no way to confirm return status from Sam's order history.",
        },
    },
    {
        "inputs": {"query": "What if I want to return something I already wore?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "items must be unworn with original tags to qualify for return",
                "worn items are not eligible",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: agent might give a more lenient answer than policy allows.",
        },
    },

    # ================================================================
    # WARRANTY LOOKUP
    # Catalog to verify ownership, policy for warranty terms.
    # Policy-only for general questions.
    # requires_sequencing: False
    # ================================================================

    {
        "inputs": {"query": "What's the warranty on my Arc'teryx jacket, and how do I make a claim?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "Jordan has the Arc'teryx Beta AR jacket (order in 'shipped' status)",
                "Arc'teryx lifetime warranty against defects in materials and workmanship",
                "claim process: warranty.arcteryx.com with photos of the issue",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: can't confirm Jordan owns it. Without policy: can't give accurate warranty terms.",
        },
    },
    {
        "inputs": {"query": "Is my MacBook Pro still under warranty?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "MacBook Pro purchased 22 days ago",
                "Apple 1-year limited warranty",
                "still under warranty - well within 1-year period",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no purchase date to verify period. Without policy: no warranty terms.",
        },
    },
    {
        "inputs": {"query": "What warranty does the Sony XM6 come with?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "Sony 1-year limited warranty",
                "covered for manufacturing defects within 1 year of purchase",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: can't give accurate Sony warranty terms.",
        },
    },
    {
        "inputs": {"query": "My Allbirds are falling apart - what's the warranty?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "Sam's Allbirds Tree Runner purchased 14 days ago",
                "Allbirds 1-year warranty on manufacturing defects",
                "claim: contact Allbirds customer service directly",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no purchase date. Without policy: wrong warranty terms.",
        },
    },
    {
        "inputs": {"query": "Do you carry anything with a lifetime warranty?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "Arc'teryx offers a lifetime warranty on defects",
                "this is the only product in our catalog with a lifetime warranty",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: can't confirm which brands offer lifetime warranties.",
        },
    },
    {
        "inputs": {"query": "My MacBook is acting strange - am I still covered and how do I claim?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "MacBook Pro purchased 22 days ago - still under Apple's 1-year warranty",
                "claim: Apple Store or apple.com/support",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no purchase date. Without policy: wrong claim process.",
        },
    },
    {
        "inputs": {"query": "What's covered under a standard product warranty here?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "standard warranty covers manufacturing defects, not damage from misuse",
                "typical period is 1 year for most brands",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: agent gives hallucinated warranty terms.",
        },
    },
    {
        "inputs": {"query": "Arc'teryx lifetime warranty - what exactly does it cover?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "warranty_lookup",
            "required_info": [
                "Arc'teryx lifetime warranty covers defects in materials and workmanship",
                "does not cover normal wear and tear or damage from misuse",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: hallucinated coverage details.",
        },
    },

    # ================================================================
    # PRODUCT COMPARISON
    # catalog fetches data → comparison_agent structures the analysis.
    # requires_sequencing: True (comparison needs catalog data to run).
    # cannot_complete_without: ["product_catalog_agent"] - without real
    # specs/prices/reviews the comparison works on hallucinated data.
    # ================================================================

    {
        "inputs": {"query": "Compare the MacBook Pro 14 and Dell XPS 13 - which is better value?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "MacBook Pro: 16GB RAM, 512GB SSD, 24hr battery, $1599",
                "Dell XPS 13: 16GB RAM, 512GB SSD, 13hr battery, $1299",
                "value judgment supported by the specs",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: comparison uses hallucinated specs.",
        },
    },
    {
        "inputs": {"query": "Show me all running shoes and rank them by value", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "running shoes in catalog: Adidas Ultraboost 24 ($190, subcategory Running), New Balance 990v6 ($185, subcategory Running)",
                "Allbirds Tree Runner is subcategory Casual - not a running shoe",
                "structured ranked output with value justification per item",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Without catalog: incomplete or hallucinated product list. product_comparison_agent produces the structured ranked output.",
        },
    },
    {
        "inputs": {"query": "Sony XM6 vs AirPods Pro - which is better for commuting?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "Sony XM6: over-ear, 30hr battery, $349, adaptive ANC",
                "AirPods Pro 3rd Gen: in-ear, 6hr (30 with case), $249, H2 chip",
                "recommendation for commuting with reasoning",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: hallucinated specs.",
        },
    },
    {
        "inputs": {"query": "MacBook Pro vs iPad Pro - which is better for content creation?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "MacBook Pro 14 M4: $1599, full laptop, 512GB, 24hr battery",
                "iPad Pro 13 M4: $1099, tablet, 256GB, Apple Pencil Pro support",
                "comparison for content creation use case",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: specs are hallucinated.",
        },
    },
    {
        "inputs": {"query": "Patagonia Nano Puff vs Uniqlo Ultra Light - which is worth more?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "Patagonia Nano Puff: $249, PrimaLoft insulation, 298g, Fair Trade",
                "Uniqlo Ultra Light Down: $89, 90% down, 195g, packable",
                "value comparison based on specs and price difference",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: can't get accurate prices or specs.",
        },
    },
    {
        "inputs": {"query": "Show all jackets under $300 ranked by value", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "jackets under $300: Patagonia Nano Puff ($249), Uniqlo Ultra Light ($89)",
                "Arc'teryx ($799) excluded as over budget",
                "structured ranked output with value justification per item",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Without catalog: might include wrong products or prices. product_comparison_agent produces the structured ranked output.",
        },
    },
    {
        "inputs": {"query": "Which headphones do you carry - rank them by what customers prefer", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "Sony WH-1000XM6 has internal reviews (Jordan: 5★, Alex: 4★ - avg 4.5)",
                "AirPods Pro 3rd Gen has NO internal reviews in our system",
                "Sony XM6 wins on customer sentiment; AirPods Pro is unranked internally",
                "structured ranked output by customer preference",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Without catalog: can't get actual review data. product_comparison_agent produces the structured ranking by customer preference.",
        },
    },
    {
        "inputs": {"query": "What's the best value laptop you carry?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "MacBook Pro 14 M4: $1599, 24hr battery, M4 chip",
                "Dell XPS 13: $1299, 13hr battery, Intel Core Ultra 7",
                "iPad Pro 13: $1099 (tablet, not traditional laptop)",
                "structured ranked output with value justification per item",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Without catalog: incomplete or hallucinated product list. product_comparison_agent produces the structured ranked output.",
        },
    },
    {
        "inputs": {"query": "Compare the Diptyque Baies and Feu de Bois candles - which should I get?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "Diptyque Baies: $75, blackcurrant + rose, 60hr burn",
                "Diptyque Feu de Bois: $75, smoky wood, 60hr burn",
                "same price and specs - comparison on fragrance profile",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: can't get accurate fragrance profiles.",
        },
    },
    {
        "inputs": {"query": "Which Alo item has better reviews - the leggings or the sweatshirt?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "Alo Airlift leggings: 5-star internal review",
                "Alo Accolade Sweatshirt: no reviews yet in our system",
                "leggings win on reviews; sweatshirt is unreviewed",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: hallucinated review data.",
        },
    },
    {
        "inputs": {"query": "Nike Air Max vs Adidas Ultraboost - which is better for running?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "Nike Air Max 270: $150, Max Air cushioning, lifestyle/light activity",
                "Adidas Ultraboost 24: $190, Boost midsole, dedicated performance running shoe",
                "Ultraboost is better for running; Air Max better for casual use",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: wrong specs.",
        },
    },
    {
        "inputs": {"query": "What's the best Skims item based on customer reviews?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "product_comparison",
            "required_info": [
                "Skims Fits Everybody Bodysuit: multiple 5-star reviews",
                "Skims Soft Lounge Slip Dress: no internal reviews yet",
                "bodysuit is the clear winner by reviews",
                "structured ranked output by customer review score",
            ],
            "expected_agents": ["product_catalog_agent", "product_comparison_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "product_comparison_agent"],
            "failure_mode": "Without catalog: no actual review data. product_comparison_agent produces the structured ranking by review score.",
        },
    },

    # ================================================================
    # DISCOVERY + EXTERNAL VALIDATION
    # product_discovery_agent for catalog matches, web_research_agent for
    # external sentiment/reviews. Both needed: discovery without web = no
    # external validation; web without discovery = doesn't match our catalog.
    # requires_sequencing: False
    # ================================================================

    {
        "inputs": {"query": "Find me a good outdoor jacket under $300 - what do reviewers say?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "catalog candidates: Patagonia Nano Puff ($249), Uniqlo Ultra Light ($89)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery alone: products without external validation. Web alone: general results without our catalog context.",
        },
    },
    {
        "inputs": {"query": "Is the Dyson worth $749? What do people actually think of it?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "internal: Sam returned it, noted battery issues; price dropped from $799",
            ],
            "expected_agents": ["product_catalog_agent", "web_research_agent"],
            "cannot_complete_without": ["product_catalog_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: misses Sam's own return and internal reviews. Without web: no external opinion.",
        },
    },
    {
        "inputs": {"query": "I need noise-cancelling headphones - what's the consensus?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "our noise-cancelling options: Sony XM6 ($349), AirPods Pro ($249)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: products without external validation. Web only: might not match our inventory.",
        },
    },
    {
        "inputs": {"query": "Any good casual sneakers you carry? What do people online say about them?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "casual sneaker options: Nike Air Max 270 ($150), Allbirds Tree Runner ($128)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: our products without outside context. Web only: general results without our catalog.",
        },
    },
    {
        "inputs": {"query": "I need a laptop for video editing - what are professionals saying is best?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "our laptops: MacBook Pro 14 M4 ($1599), Dell XPS 13 ($1299)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: our products without editorial context. Web only: might not reference our models.",
        },
    },
    {
        "inputs": {"query": "I'm looking for skincare - are the options you carry actually worth it?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "our skincare: Rhode Peptide Lip Treatment ($20), Drunk Elephant Protini ($68)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: products without external validation. Web only: general advice without our catalog.",
        },
    },
    {
        "inputs": {"query": "Looking for running shoes - what do the experts say?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "our running shoes: Ultraboost 24 ($190), NB 990v6 ($185)",
            ],
            "expected_agents": ["product_catalog_agent", "product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: our options without expert context. Web only: might not match our actual inventory.",
        },
    },
    {
        "inputs": {"query": "What's a good winter coat for NYC - what are people loving right now?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "our coat options: Aritzia Super Puff ($325), Arc'teryx Beta AR ($799), Patagonia Nano Puff ($249)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: catalog without trend context. Web only: general trends without our inventory.",
        },
    },
    {
        "inputs": {"query": "What's a good gift under $100 - and what do people say about it?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "catalog options under $100: Uniqlo Down ($89), YETI Tumbler ($45), Rhode ($20), Levi's ($79)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: our options without gift/review context. Web only: general ideas without our catalog.",
        },
    },
    {
        "inputs": {"query": "I want a candle or fragrance - what are people actually loving right now?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "discovery_with_validation",
            "required_info": [
                "our fragrance/candle options: Diptyque Baies ($75), Feu de Bois ($75), Glossier You ($72)",
            ],
            "expected_agents": ["product_discovery_agent", "web_research_agent"],
            "cannot_complete_without": ["product_discovery_agent", "web_research_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Discovery only: catalog without trend context. Web only: general fragrance trends without our inventory.",
        },
    },

    # ================================================================
    # SIZING WITH CONTEXT
    # Policy-only for general sizing questions.
    # Catalog + policy when the question is about a specific order
    # (need to verify what size they actually ordered).
    # requires_sequencing: False
    # ================================================================

    {
        "inputs": {"query": "I just ordered the Skims bodysuit - did I pick the right size?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Mia ordered Skims Fits Everybody Bodysuit (size XS, 2 units, 20 days ago)",
                "Skims Fits Everybody stretches to 3x, true to size",
                "verdict: XS is correct for Mia's XS frame",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know what size she ordered. Without policy: no Skims-specific sizing guidance.",
        },
    },
    {
        "inputs": {"query": "I'm getting the Aritzia Super Puff - what size given I'm XS but want it roomy?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Aritzia Super Puff runs true to size with a relaxed boxy fit",
                "for roomy look: TTS is already generous; XS or 2XS both work",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: generic advice missing that Aritzia runs generous.",
        },
    },
    {
        "inputs": {"query": "Are the Allbirds Tree Runners true to size? I'm usually an 8.", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Allbirds Tree Runner: runs slightly narrow, suggest sizing up half for wide feet",
                "for size 8 normal width: TTS is fine",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: generic TTS advice missing the narrow-fit nuance.",
        },
    },
    {
        "inputs": {"query": "My Alo leggings feel kind of tight - did I order the right size?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Mia ordered Alo Airlift leggings (XS, delivered 6 days ago)",
                "Alo Airlift runs true to size - XS is correct for an XS frame",
                "tightness is expected - Airlift is a compressive performance fabric",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know what size she ordered. Without policy: can't give Alo-specific sizing context.",
        },
    },
    {
        "inputs": {"query": "I want the Alo sweatshirt in an oversized look - what size?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Alo Accolade Sweatshirt already has a relaxed crop fit",
                "for oversized look: size up one (XS to S) for extra room",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: generic sizing advice missing that Alo already has a relaxed fit.",
        },
    },
    {
        "inputs": {"query": "Does the Patagonia Nano Puff run large or true to size?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Patagonia Nano Puff runs true to size",
                "for layering: order regular size or size up if wearing over thick sweaters",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: hallucinated sizing guidance.",
        },
    },
    {
        "inputs": {"query": "New Balance 990v6 - should I size up or is it TTS?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "New Balance 990v6 runs true to size",
                "available in wide widths - go up half size if between sizes or wide footed",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: generic advice missing NB width options.",
        },
    },
    {
        "inputs": {"query": "Timberland 6-inch boot - how do they fit?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Timberland 6-inch boot runs slightly large - size down half",
                "wide toe box: good for wide feet at true size",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: might recommend TTS when Timberlands run large.",
        },
    },
    {
        "inputs": {"query": "How does the Arc'teryx Beta AR fit - I usually wear L", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Arc'teryx Beta AR: athletic/trim fit, designed for layering over midlayers",
                "L in other brands: L in Arc'teryx is correct; size up if wearing thick base + mid layers",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: generic sizing advice missing Arc'teryx athletic-fit specifics.",
        },
    },
    {
        "inputs": {"query": "I want to order the Skims Soft Lounge Slip Dress - what size as an XS?", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Skims Soft Lounge: stretchy, true to size or size down for fitted look",
                "for XS: XXS for body-con, XS for standard, S for relaxed",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: generic advice missing Skims Soft Lounge stretch guidance.",
        },
    },
    {
        "inputs": {"query": "Do the Levi's 501 run large or true to size?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Levi's 501: 100% cotton, will shrink slightly with first wash",
                "size up one in waist if between sizes; inseam runs true",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: misses cotton-shrink nuance and Levi's size-up recommendation.",
        },
    },
    {
        "inputs": {"query": "I'm buying a Skims bodysuit in XS as a gift - will it fit someone who's usually a S?", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "sizing_with_context",
            "required_info": [
                "Skims Fits Everybody fabric stretches to 3x its size",
                "XS fits roughly XXS-S range - will work for a S but be more fitted",
                "recommend S for a standard fit; XS will be snug",
            ],
            "expected_agents": ["policy_and_sizing_agent"],
            "cannot_complete_without": ["policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without policy: can't give Skims-specific stretch guidance.",
        },
    },

    # ================================================================
    # ACTION WITH PREREQUISITE
    # catalog lookup MUST precede cart action.
    # The cart_and_orders_agent needs real order_id / sku from catalog.
    # Without catalog: any id passed to add_to_cart / initiate_return is
    # hallucinated and will fail or produce wrong results.
    # requires_sequencing: True for all.
    # cannot_complete_without: always includes product_catalog_agent;
    # also policy_and_sizing_agent when return eligibility must be checked.
    # ================================================================

    {
        "inputs": {"query": "Buy me the cheaper of the two Diptyque candles", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "both Diptyque candles are $75 - same price",
                "correct response: inform they're the same price, ask which she prefers",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: hallucinates a price difference or picks one arbitrarily.",
        },
    },
    {
        "inputs": {"query": "Add the best-reviewed headphones to my cart", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Sony WH-1000XM6 has the highest internal review rating (4.5 avg)",
                "correct action: add Sony XM6 (ELEC-003) to Jordan's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't determine actual review ratings; SKU passed to add_to_cart is hallucinated.",
        },
    },
    {
        "inputs": {"query": "Return my most recent delivered order", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "most recent delivered: Alo leggings + Diptyque Baies (6 days ago)",
                "policy: candles returnable only if unopened within 14 days; leggings within 30 days",
                "can only return the leggings - candle requires unopened condition",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order_id/sku. Without policy: might attempt to return the candle incorrectly.",
        },
    },
    {
        "inputs": {"query": "I want to return my jacket - can you start the return?", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Arc'teryx jacket status is 'shipped' - not delivered",
                "return requires delivered status",
                "cannot initiate return yet - not delivered",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know the jacket is only shipped, not delivered.",
        },
    },
    {
        "inputs": {"query": "Add the cheapest laptop to my wishlist", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Dell XPS 13 is the cheapest laptop at $1299 (MacBook is $1599)",
                "correct action: save Dell XPS 13 (ELEC-002) to Alex's wishlist",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't determine actual prices; SKU would be hallucinated.",
        },
    },
    {
        "inputs": {"query": "Start a return for my Allbirds", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Allbirds Tree Runner (FOOT-003) delivered 14 days ago - within 30-day footwear window",
                "return initiated for FOOT-003, or eligibility confirmed and conditions communicated to user",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order_id or sku for initiate_return.",
        },
    },
    {
        "inputs": {"query": "Return my Sony headphones", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Sony XM6 (ELEC-003) delivered 5 days ago - within 30-day electronics window",
                "return initiated for ELEC-003, or eligibility confirmed and conditions communicated to user",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order details or SKU to verify delivery or initiate return.",
        },
    },
    {
        "inputs": {"query": "Add the most popular Diptyque candle to my wishlist", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Diptyque Baies has a 5-star review and is described as iconic",
                "correct action: save Diptyque Baies (BEAU-001) to Mia's wishlist",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't determine which candle has better reviews.",
        },
    },
    {
        "inputs": {"query": "I want to return the Skims bodysuit I ordered", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Skims Fits Everybody Bodysuit delivered 20 days ago",
                "Skims bodysuits are non-returnable - final sale",
                "cannot initiate return - Skims bodysuits are excluded",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no order details. Without policy: misses Skims non-returnable rule.",
        },
    },
    {
        "inputs": {"query": "Return my MacBook", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "MacBook Pro delivered 22 days ago",
                "electronics return window is 14 days",
                "not eligible - outside the 14-day electronics window",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: no order date. Without policy: might apply 30-day standard window instead of 14-day electronics rule.",
        },
    },
    {
        "inputs": {"query": "Add the cheapest item in Beauty & Wellness to my cart", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "cheapest Beauty & Wellness item: Rhode Peptide Lip Treatment at $20",
                "correct action: add BEAU-003 to Mia's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't find actual prices across Beauty & Wellness.",
        },
    },
    {
        "inputs": {"query": "Save the Aritzia Super Puff to my wishlist", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Aritzia Super Puff Long Jacket SKU is FASH-003, price $325",
                "correct action: save FASH-003 to Mia's wishlist",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no SKU to pass to save_to_wishlist.",
        },
    },
    {
        "inputs": {"query": "Return my Dyson", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Sam's Dyson V15 was already returned 35 days ago - status is 'returned'",
                "correct response: inform Sam the return was already processed",
            ],
            "expected_agents": ["product_catalog_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: doesn't know return was already processed - policy check is irrelevant once status is 'returned'.",
        },
    },
    {
        "inputs": {"query": "Add the most expensive item I've ever ordered to my wishlist", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Alex's most expensive order: MacBook Pro at $1599",
                "correct action: save MacBook Pro (ELEC-001) to Alex's wishlist",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't look up Alex's order history.",
        },
    },
    {
        "inputs": {"query": "Add the AirPods Pro to my cart", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Apple AirPods Pro 3rd Gen SKU is ELEC-004, price $249",
                "correct action: add ELEC-004 to Alex's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: SKU passed to add_to_cart is hallucinated.",
        },
    },
    {
        "inputs": {"query": "Return both items from my last order", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Sam's last delivered order: Allbirds + Uniqlo Down Jacket (14 days ago)",
                "both within 30-day window, both eligible",
                "returns initiated for FOOT-003 and CLTH-004, or eligibility confirmed and conditions communicated to user",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order details or SKUs for initiate_return.",
        },
    },
    {
        "inputs": {"query": "Add the Sony headphones to my cart", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Sony WH-1000XM6 SKU is ELEC-003, price $349",
                "correct action: add ELEC-003 to Alex's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: SKU would be hallucinated.",
        },
    },
    {
        "inputs": {"query": "Return everything I've ordered that's eligible", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Mia's delivered: Alo leggings + Diptyque Baies (6 days ago), Skims bodysuit (20 days ago)",
                "eligible: Alo leggings only - Skims non-returnable, candle requires unopened",
                "only the leggings can be returned",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't enumerate orders. Without policy: can't determine which are actually eligible.",
        },
    },
    {
        "inputs": {"query": "Wishlist the cheapest home item you carry", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Home items by price: YETI Tumbler ($45), Instant Pot ($99), Dyson V15 ($749)",
                "cheapest: YETI Rambler 30oz at $45 (HOME-003)",
                "correct action: save HOME-003 to Sam's wishlist",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: can't get actual prices to find cheapest.",
        },
    },
    {
        "inputs": {"query": "Add the Rhode lip treatment to my cart", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Rhode Peptide Lip Treatment SKU is BEAU-003, price $20",
                "correct action: add BEAU-003 to Mia's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"]],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: SKU would be hallucinated.",
        },
    },
    {
        "inputs": {"query": "Return my Allbirds and add a YETI tumbler to my cart", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Allbirds (FOOT-003) delivered 14 days ago, eligible for return",
                "YETI Rambler 30oz SKU is HOME-003, price $45",
                "return initiated for FOOT-003 or eligibility confirmed; add HOME-003 to cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order details for return, no SKU for cart.",
        },
    },
    {
        "inputs": {"query": "Add the Patagonia Nano Puff to my cart", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Patagonia Nano Puff Jacket SKU is CLTH-001, price $249",
                "correct action: add CLTH-001 to Alex's cart",
            ],
            "expected_agents": ["product_catalog_agent", "cart_and_orders_agent"],
            "cannot_complete_without": [["product_catalog_agent", "product_discovery_agent"], "cart_and_orders_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: SKU would be hallucinated.",
        },
    },

    # ================================================================
    # TIMELINE CORRECTION
    # Customer states a wrong timeframe - agent must look up actual
    # delivery date and correct them rather than accepting the claim.
    # ================================================================

    {
        "inputs": {"query": "I've had my Allbirds for over a month - am I past the return window?", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "return_eligibility",
            "required_info": [
                "Sam's Allbirds Tree Runner delivered 14 days ago - not over a month, correct the customer",
                "footwear return window is 30 days, unworn",
                "eligible - 16 days remaining",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": False,
            "expected_sequence": [],
            "failure_mode": "Without catalog: agent accepts 'over a month' as fact instead of verifying actual delivery date. Without policy: can't apply the correct return window.",
        },
    },
    {
        "inputs": {"query": "Return my Alo leggings", "user_email": "mia@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Alo Airlift leggings (ACTV-001) delivered 6 days ago",
                "activewear return window is 30 days, unworn with tags - eligible",
                "return initiated for ACTV-001, or eligibility confirmed and conditions communicated to user",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order details or SKU. Without policy: can't confirm eligibility or conditions.",
        },
    },

    # ================================================================
    # FULL 3-STEP RETURN SEQUENCE (catalog → policy → cart)
    # These require all three agents in order:
    #   1. catalog: find the order and delivery date
    #   2. policy: verify return eligibility for the category
    #   3. cart_and_orders: execute the return
    # Without all three in sequence, the return is either wrong or
    # never happens. These test that the agent doesn't skip policy
    # or act before verifying eligibility.
    # ================================================================

    {
        "inputs": {"query": "Return my AirPods", "user_email": "alex@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "AirPods Pro (ELEC-004) status is 'processing' - not yet delivered",
                "return policy requires item to be delivered before a return can be initiated",
                "cannot initiate return - item not yet received",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent"],
            "failure_mode": "Without catalog: no way to know AirPods are still processing. Without policy: might attempt return on undelivered item.",
        },
    },
    {
        "inputs": {"query": "Return my Uniqlo jacket", "user_email": "sam@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Uniqlo Ultra Light Down Jacket (CLTH-004) delivered 14 days ago",
                "outerwear return window is 30 days, unworn with tags - eligible",
                "return initiated for CLTH-004, or eligibility confirmed and conditions communicated to user",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order_id or SKU for CLTH-004. Without policy: can't confirm 30-day window or eligibility conditions.",
        },
    },
    {
        "inputs": {"query": "I want to return my Sony headphones - please start the return", "user_email": "jordan@example.com"},
        "outputs": {
            "task_type": "action_with_prerequisite",
            "required_info": [
                "Sony WH-1000XM6 (ELEC-003) delivered 5 days ago - within 30-day electronics window",
                "return initiated for ELEC-003, or eligibility confirmed and conditions communicated to user",
            ],
            "expected_agents": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "cannot_complete_without": ["product_catalog_agent", "policy_and_sizing_agent"],
            "requires_sequencing": True,
            "expected_sequence": ["product_catalog_agent", "policy_and_sizing_agent", "cart_and_orders_agent"],
            "failure_mode": "Without catalog: no order_id or SKU for ELEC-003. Without policy: can't confirm 30-day electronics window or eligibility conditions.",
        },
    },
]


# ---------------------------------------------------------------------------
# Split + export
# ---------------------------------------------------------------------------

def split_examples(examples: list[dict], train: float = 0.6, val: float = 0.2) -> tuple:
    """Stratified split - each task_type is split proportionally across train/val/test."""
    random.seed(42)
    from collections import defaultdict
    by_type = defaultdict(list)
    for ex in examples:
        by_type[ex["outputs"]["task_type"]].append(ex)

    train_out, val_out, test_out = [], [], []
    for task_type, group in by_type.items():
        shuffled = group.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train))
        n_val = max(1, int(n * val))
        train_out.extend(shuffled[:n_train])
        val_out.extend(shuffled[n_train:n_train + n_val])
        test_out.extend(shuffled[n_train + n_val:])

    return train_out, val_out, test_out


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples → {path.name}")


def upload_split(client: Client, examples: list[dict], split: str, replace: bool = False, suffix: str = "") -> None:
    base = f"{DATASET_NAME}{suffix}"
    name = f"{base}-{split}"
    if client.has_dataset(dataset_name=name):
        if not replace:
            print(f"  '{name}' already exists - skipping. Pass --replace to overwrite.")
            return
        client.delete_dataset(dataset_name=name)
        print(f"  Deleted existing '{name}'")
    dataset = client.create_dataset(
        dataset_name=name,
        description=f"Shopping concierge routing eval - {split} split",
    )
    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id,
    )
    print(f"  Uploaded {len(examples)} examples → '{name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="Delete and re-upload existing datasets")
    parser.add_argument("--suffix", default="", help="Suffix appended to dataset name, e.g. '-v2' → shopping-concierge-routing-v2-train")
    args = parser.parse_args()

    train, val, test = split_examples(EXAMPLES)
    print(f"\n{len(EXAMPLES)} examples → {len(train)} train / {len(val)} val / {len(test)} test\n")

    print("Writing JSONL...")
    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(val, DATA_DIR / "val.jsonl")
    write_jsonl(test, DATA_DIR / "test.jsonl")

    print("\nUploading to LangSmith...")
    client = Client()
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        upload_split(client, split_data, split_name, replace=args.replace, suffix=args.suffix)

    print("\nDone.")
