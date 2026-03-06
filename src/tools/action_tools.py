"""
Write/action tools for cart, wishlist, and returns.
Used by the action_agent — the only subagent that modifies state.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from langchain.tools import tool

DB_PATH = Path(__file__).parent.parent.parent / "data" / "shop.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@tool
def add_to_cart(user_email: str, product_sku: str, quantity: int = 1) -> str:
    """
    Add a product to the user's shopping cart.

    Use this ONLY when the customer explicitly says they want to buy, add to
    cart, or purchase an item. Do not call this for browsing, comparing, or
    asking about products.

    Args:
        user_email: The customer's email address (e.g. "mia@example.com").
        product_sku: The product SKU (e.g. "FASH-001"). Must be an exact SKU
                     from the catalog. Use search_products to find the SKU first
                     if you don't have it.
        quantity: Number of units to add (default 1).

    Returns:
        Confirmation message with product name and updated cart quantity.
    """
    with _conn() as conn:
        user = conn.execute(
            "SELECT id, name FROM users WHERE email = ?", (user_email,)
        ).fetchone()
        if not user:
            return json.dumps({"error": f"No user found: {user_email}"})

        product = conn.execute(
            "SELECT id, name, price FROM products WHERE sku = ?", (product_sku,)
        ).fetchone()
        if not product:
            return json.dumps({"error": f"Product not found: {product_sku}"})

        inventory = conn.execute(
            "SELECT quantity FROM inventory WHERE product_id = ?", (product["id"],)
        ).fetchone()
        if not inventory or inventory["quantity"] < quantity:
            return json.dumps({"error": f"Insufficient stock for {product['name']}. Available: {inventory['quantity'] if inventory else 0}"})

        conn.execute(
            """
            INSERT INTO cart (user_id, product_id, quantity, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, product_id) DO UPDATE SET quantity = quantity + excluded.quantity
            """,
            (user["id"], product["id"], quantity, datetime.now().isoformat()),
        )
        conn.commit()

    return json.dumps({
        "success": True,
        "message": f"Added {quantity}x {product['name']} to {user['name']}'s cart.",
        "product": product["name"],
        "price_each": product["price"],
        "subtotal": product["price"] * quantity,
    })


@tool
def save_to_wishlist(user_email: str, product_sku: str) -> str:
    """
    Save a product to the user's wishlist for later.

    Use this when the customer says they want to save something, bookmark it,
    add it to their wishlist, or come back to it later — but is NOT ready to
    buy right now.

    Args:
        user_email: The customer's email address.
        product_sku: The product SKU (e.g. "BEAU-001").

    Returns:
        Confirmation message.
    """
    with _conn() as conn:
        user = conn.execute(
            "SELECT id, name FROM users WHERE email = ?", (user_email,)
        ).fetchone()
        if not user:
            return json.dumps({"error": f"No user found: {user_email}"})

        product = conn.execute(
            "SELECT id, name FROM products WHERE sku = ?", (product_sku,)
        ).fetchone()
        if not product:
            return json.dumps({"error": f"Product not found: {product_sku}"})

        conn.execute(
            """
            INSERT OR IGNORE INTO wishlist (user_id, product_id, added_at)
            VALUES (?, ?, ?)
            """,
            (user["id"], product["id"], datetime.now().isoformat()),
        )
        conn.commit()

    return json.dumps({
        "success": True,
        "message": f"Saved {product['name']} to {user['name']}'s wishlist.",
    })


@tool
def initiate_return(user_email: str, order_id: int, product_sku: str, reason: str) -> str:
    """
    Initiate a return for an item in a delivered order.

    Use this ONLY when the customer explicitly asks to return or exchange an
    item, AND after confirming the item is eligible (within return window,
    not a non-returnable category). Check the return policy via the docs tools
    and verify the order via get_user_orders before calling this.

    Args:
        user_email: The customer's email address.
        order_id: The order ID containing the item to return. Get this from
                  get_user_orders first.
        product_sku: SKU of the specific item to return (e.g. "CLTH-004").
        reason: Customer's reason for return. Options: "wrong_size",
                "changed_mind", "defective", "not_as_described", "arrived_damaged",
                "found_better_price".

    Returns:
        Return authorization with instructions, or an error if ineligible.
    """
    with _conn() as conn:
        user = conn.execute(
            "SELECT id, name FROM users WHERE email = ?", (user_email,)
        ).fetchone()
        if not user:
            return json.dumps({"error": f"No user found: {user_email}"})

        order = conn.execute(
            """
            SELECT o.id, o.status, o.created_at, p.name AS product_name, p.sku,
                   p.category, oi.price_at_purchase
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            WHERE o.id = ? AND o.user_id = ? AND p.sku = ?
            """,
            (order_id, user["id"], product_sku),
        ).fetchone()

        if not order:
            return json.dumps({"error": f"Order {order_id} not found or does not contain {product_sku}"})

        if order["status"] not in ("delivered", "shipped"):
            return json.dumps({"error": f"Cannot return order with status '{order['status']}'. Only delivered orders can be returned."})

        # Check non-returnable categories
        non_returnable = ["Bodysuits"]
        if order["category"] in non_returnable or order["sku"] in ("FASH-001",):
            return json.dumps({
                "error": f"{order['product_name']} cannot be returned. Bodysuits are non-returnable for hygiene reasons."
            })

        # Update order status
        conn.execute(
            "UPDATE orders SET status = 'return_initiated' WHERE id = ?",
            (order_id,),
        )
        conn.commit()

    return json.dumps({
        "success": True,
        "return_authorization": f"RA-{order_id}-{product_sku}",
        "product": order["product_name"],
        "refund_amount": order["price_at_purchase"],
        "reason": reason,
        "instructions": (
            "Your return has been authorized. A prepaid shipping label will be emailed to you within 1 hour. "
            "Drop off at any UPS or USPS location within 7 days. "
            "Refunds are processed within 5-7 business days of receiving the return."
        ),
    })
