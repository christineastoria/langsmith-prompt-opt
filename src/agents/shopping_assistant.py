"""
Personal shopping concierge — main orchestrator.

Creates the top-level Deep Agent that receives customer messages and delegates
to specialized subagents. The system prompt being optimized lives in prompts/.

Usage:
    from agents.shopping_assistant import create_concierge, get_user_context

    agent = create_concierge()
    config = {"configurable": {"thread_id": "user-mia@example.com"}}

    # Inject user context as the first system message
    context = get_user_context("mia@example.com")
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": "Can I return the Alo leggings I ordered?"},
        ]
    }, config=config)
"""

import json
import sqlite3
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from agents.subagents import ALL_SUBAGENTS

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
DB_PATH = Path(__file__).parent.parent.parent / "data" / "shop.db"


def load_system_prompt(variant: str = "baseline") -> str:
    """Load a system prompt variant from prompts/. Defaults to baseline.md."""
    path = PROMPTS_DIR / f"{variant}.md"
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text().strip()


def get_user_context(user_email: str) -> str:
    """
    Build a user context string to inject at the start of each session.
    Includes name, preferences, sizes, and recent order history.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    user = conn.execute(
        "SELECT id, name, preferences FROM users WHERE email = ?", (user_email,)
    ).fetchone()

    if not user:
        conn.close()
        return f"Customer email: {user_email} (no profile found)"

    prefs = json.loads(user["preferences"])

    recent_orders = conn.execute(
        """
        SELECT o.id AS order_id, o.status, o.created_at,
               p.name AS product_name, p.sku, oi.quantity
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        WHERE o.user_id = ?
        ORDER BY o.created_at DESC
        LIMIT 6
        """,
        (user["id"],),
    ).fetchall()

    conn.close()

    orders_summary = []
    seen_orders: set = set()
    for row in recent_orders:
        oid = row["order_id"]
        if oid not in seen_orders:
            seen_orders.add(oid)
            orders_summary.append(
                f"  - Order #{oid} ({row['status']}): {row['product_name']}"
            )

    context_lines = [
        f"## Current customer",
        f"Name: {user['name']}",
        f"Email: {user_email}",
    ]

    if prefs.get("sizes"):
        size_parts = ", ".join(f"{k}: {v}" for k, v in prefs["sizes"].items())
        context_lines.append(f"Sizes: {size_parts}")

    if prefs.get("brands"):
        context_lines.append(f"Favorite brands: {', '.join(prefs['brands'])}")

    if prefs.get("categories"):
        context_lines.append(f"Shops for: {', '.join(prefs['categories'])}")

    if orders_summary:
        context_lines.append("Recent orders:")
        context_lines.extend(orders_summary)

    return "\n".join(context_lines)


def create_concierge(prompt_variant: str = "baseline") -> object:
    """
    Create the shopping concierge Deep Agent.

    Args:
        prompt_variant: Which system prompt to load from prompts/.
                        Defaults to "baseline". Use "optimized" after running
                        the optimizer.

    Returns:
        A compiled Deep Agent ready to invoke.
    """
    system_prompt = load_system_prompt(prompt_variant)

    return create_deep_agent(
        model="claude-haiku-4-5-20251001",
        system_prompt=system_prompt,
        subagents=ALL_SUBAGENTS,
        checkpointer=MemorySaver(),
    )
