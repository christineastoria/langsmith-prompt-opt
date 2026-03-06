"""
Subagent definitions for the personal shopping concierge.

Each subagent is specialized for one type of work. The main orchestrator
delegates to these via the `task` tool based on what the query requires.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain.tools import tool

from tools.sql_tools import (
    get_inventory_status,
    get_price_history,
    get_product_details,
    get_product_reviews,
    get_user_orders,
    search_products,
)
from tools.search_tools import semantic_search
from tools.web_tools import web_search
from tools.action_tools import add_to_cart, save_to_wishlist, initiate_return

DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "docs"
DB_PATH = Path(__file__).parent.parent.parent / "data" / "shop.db"


# ---------------------------------------------------------------------------
# Docs tools (scoped to data/docs/)
# ---------------------------------------------------------------------------

@tool
def list_docs() -> str:
    """
    List all available policy and guide documents.
    Use this first if you're unsure which document to read.
    Returns a list of available document filenames.
    """
    files = [f.name for f in DOCS_DIR.glob("*.md")]
    return json.dumps({"documents": files})


@tool
def read_doc(filename: str) -> str:
    """
    Read a policy or guide document in full.

    Available documents:
    - return_policy.md: Return windows, conditions, non-returnable items, brand-specific rules
    - size_guide.md: Sizing advice and measurement charts for every brand we carry
    - warranty_info.md: Manufacturer warranty length and claims process for all categories
    - brand_guide.md: Brand positioning, who each brand is best for, recommendation tips

    Args:
        filename: Name of the document to read (e.g. "return_policy.md").

    Returns:
        Full text content of the document.
    """
    path = DOCS_DIR / filename
    if not path.exists():
        available = [f.name for f in DOCS_DIR.glob("*.md")]
        return json.dumps({"error": f"Document '{filename}' not found.", "available": available})
    return path.read_text()


@tool
def search_doc(filename: str, query: str) -> str:
    """
    Search within a document for a specific term without reading the whole thing.

    Use this when you need a specific section (e.g. Skims return rules, Arc'teryx
    sizing, Dyson warranty length) rather than the entire document.

    Args:
        filename: Document to search (e.g. "size_guide.md").
        query: Term or brand to search for (e.g. "Skims", "Arc'teryx", "14 days").

    Returns:
        Matching lines with surrounding context.
    """
    path = DOCS_DIR / filename
    if not path.exists():
        return json.dumps({"error": f"Document '{filename}' not found."})

    lines = path.read_text().splitlines()
    query_lower = query.lower()
    results = []
    for i, line in enumerate(lines):
        if query_lower in line.lower():
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            results.append({
                "line": i + 1,
                "context": "\n".join(lines[start:end]),
            })

    if not results:
        return json.dumps({"message": f"No matches for '{query}' in {filename}."})
    return json.dumps({"matches": results})


# ---------------------------------------------------------------------------
# Code execution tool
# ---------------------------------------------------------------------------

@tool
def run_python(code: str) -> str:
    """
    Execute a Python script for product analysis and return the output.

    The script has read-only access to the shopping SQLite database via the
    DB_PATH variable (pre-injected). Use the sqlite3 module to query it.

    Good uses:
    - Side-by-side product spec comparisons (e.g. MacBook vs Dell XPS)
    - Price trend calculations and discount percentages
    - Filtering and ranking products by multiple attributes at once
    - Formatted comparison tables

    Args:
        code: Valid Python code. Must use print() for output — return values
              are not captured. Import sqlite3 and use DB_PATH to query the db.

    Returns:
        stdout from the script, or an error message if execution fails.
    """
    preamble = f'DB_PATH = r"{DB_PATH}"\n'
    full_code = preamble + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Error:\n{result.stderr[:1000]}"
        return result.stdout[:3000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: script timed out after 15 seconds."
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Subagent definitions
# ---------------------------------------------------------------------------

PRODUCT_CATALOG_AGENT = {
    "name": "product_catalog_agent",
    "description": (
        "Looks up structured product and customer data from the internal database. "
        "Use for: searching products by name, brand, or category; fetching full product specs "
        "and pricing; checking price history; looking up inventory/stock levels; retrieving a "
        "customer's order history and order status; reading internal customer reviews."
    ),
    "system_prompt": (
        "You are a catalog and customer data specialist for a personal shopping concierge. "
        "Use the provided tools to retrieve accurate product details, pricing, order history, "
        "and internal reviews. If the task requires multiple lookups (e.g. product details "
        "plus its reviews), call all relevant tools and combine the results clearly."
    ),
    "tools": [
        search_products,
        get_product_details,
        get_price_history,
        get_user_orders,
        get_product_reviews,
        get_inventory_status,
    ],
}

PRODUCT_DISCOVERY_AGENT = {
    "name": "product_discovery_agent",
    "description": (
        "Finds products using natural language and semantic similarity — not exact names. "
        "Use for vague, descriptive, or vibe-based queries: 'something cozy for winter', "
        "'a gift for someone who loves the outdoors', 'wireless headphones for commuting', "
        "'a candle that smells like a fancy hotel'. Do NOT use for exact product name lookups."
    ),
    "system_prompt": (
        "You are a product discovery specialist for a personal shopping concierge. "
        "Use semantic_search to find products that best match the customer's natural language "
        "description. Return the top matches with the product name, brand, price, and a brief "
        "explanation of why each one fits what they're looking for."
    ),
    "tools": [semantic_search],
}

WEB_RESEARCH_AGENT = {
    "name": "web_research_agent",
    "description": (
        "Searches the web for current, external information. Use for: competitor pricing at "
        "other retailers, expert and editorial reviews from external sources, current sales or "
        "promotions, brand news and announcements, product recalls, or comparing our price "
        "against the market. Do NOT use for information already in our internal catalog."
    ),
    "system_prompt": (
        "You are a web research specialist for a personal shopping concierge. "
        "Use web_search to find current pricing, expert reviews, deals, and brand news. "
        "Always cite sources. Prioritize reputable sources (major retailers, established review "
        "sites like Wirecutter, RTINGS, Vogue, etc.). Be concise and actionable."
    ),
    "tools": [web_search],
}

POLICY_AND_SIZING_AGENT = {
    "name": "policy_and_sizing_agent",
    "description": (
        "Answers questions using internal policy and product knowledge documents. "
        "Use for: whether an item is returnable and how to do it, sizing advice and fit "
        "recommendations for any brand we carry, warranty coverage and how to make a claim, "
        "and brand background or recommendation guidance. "
        "Has access to: return_policy.md, size_guide.md, warranty_info.md, brand_guide.md."
    ),
    "system_prompt": (
        "You are a policy and product knowledge specialist for a personal shopping concierge. "
        "Use read_doc and search_doc to answer questions about return policies, sizing, warranties, "
        "and brand guidance. Always cite which document your answer comes from. Surface brand-specific "
        "exceptions and edge cases (e.g. Skims bodysuits are non-returnable, Apple has a 14-day "
        "return window for opened products)."
    ),
    "tools": [list_docs, read_doc, search_doc],
}

PRODUCT_COMPARISON_AGENT = {
    "name": "product_comparison_agent",
    "description": (
        "Runs Python analysis for structured comparisons, calculations, and multi-criteria filtering. "
        "Use for: comparing two or more products side by side on specs and price, calculating "
        "discount percentages or price trends, filtering the catalog by multiple attributes at once, "
        "or generating a formatted comparison table the customer can read easily."
    ),
    "system_prompt": (
        "You are a data analysis specialist for a personal shopping concierge. "
        "Use run_python to write and execute scripts that query the SQLite database and compute "
        "results. Always use print() for output. Use sqlite3 and the injected DB_PATH variable. "
        "Format outputs as clean, readable tables or bullet-point summaries."
    ),
    "tools": [run_python],
}

CART_AND_ORDERS_AGENT = {
    "name": "cart_and_orders_agent",
    "description": (
        "Performs write actions on the customer's account. Use ONLY when the customer explicitly "
        "requests an action: adding a specific item to their cart, saving an item to their wishlist, "
        "or initiating a return on a delivered order. Never call this for browsing, research, "
        "or questions about products."
    ),
    "system_prompt": (
        "You are the order and cart action specialist for a personal shopping concierge. "
        "Execute customer-requested actions: add to cart, save to wishlist, initiate returns. "
        "Confirm what was done and provide clear next steps. For returns, always confirm the "
        "item name, order ID, and reason before executing."
    ),
    "tools": [add_to_cart, save_to_wishlist, initiate_return],
}

ALL_SUBAGENTS = [
    PRODUCT_CATALOG_AGENT,
    PRODUCT_DISCOVERY_AGENT,
    WEB_RESEARCH_AGENT,
    POLICY_AND_SIZING_AGENT,
    PRODUCT_COMPARISON_AGENT,
    CART_AND_ORDERS_AGENT,
]
