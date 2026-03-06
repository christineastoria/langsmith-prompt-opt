"""
SQL tools for querying the SQLite shopping database.
Used by the sql_agent for structured product, inventory, pricing, order, and review lookups.
"""

import json
import sqlite3
from pathlib import Path

from langchain.tools import tool

DB_PATH = Path(__file__).parent.parent.parent / "data" / "shop.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


@tool
def search_products(
    query: str,
    category: str = None,
    brand: str = None,
    min_price: float = None,
    max_price: float = None,
    limit: int = 10,
) -> str:
    """
    Search for products by keyword in the product catalog.

    Searches product names, descriptions, and brands. Use this for structured
    catalog lookups where you know what the customer is looking for. For fuzzy
    or natural-language searches ("something cozy for winter"), prefer the
    semantic_search tool instead.

    Args:
        query: Keyword(s) to search in product name, brand, and description.
        category: Filter by category. Options: "Electronics", "Footwear",
                  "Clothing", "Activewear", "Fashion", "Beauty & Wellness", "Home".
        brand: Filter by brand name (e.g. "Apple", "Skims", "Alo Yoga").
        min_price: Minimum price filter in USD.
        max_price: Maximum price filter in USD.
        limit: Max number of results to return (default 10).

    Returns:
        JSON list of matching products with id, sku, name, brand, category,
        price, and a short description snippet.
    """
    with _conn() as conn:
        sql = """
            SELECT id, sku, name, brand, category, subcategory, price,
                   substr(description, 1, 150) AS description_snippet
            FROM products
            WHERE (name LIKE ? OR description LIKE ? OR brand LIKE ?)
        """
        params: list = [f"%{query}%", f"%{query}%", f"%{query}%"]

        if category:
            sql += " AND category = ?"
            params.append(category)
        if brand:
            sql += " AND brand = ?"
            params.append(brand)
        if min_price is not None:
            sql += " AND price >= ?"
            params.append(min_price)
        if max_price is not None:
            sql += " AND price <= ?"
            params.append(max_price)

        sql += " LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return json.dumps({"results": [], "message": f"No products found matching '{query}'"})
    return json.dumps({"results": _rows_to_dicts(rows)})


@tool
def get_product_details(identifier: str) -> str:
    """
    Get full details for a single product including specs, price, and inventory.

    Use this when you need complete information about a specific product —
    full description, all specs, current price, and stock level.

    Args:
        identifier: Either a product SKU (e.g. "ELEC-003") or product name
                    (e.g. "Sony WH-1000XM6"). Partial name matches work.

    Returns:
        JSON object with full product details including specs (JSON), current
        price, and inventory quantity. Returns an error message if not found.
    """
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT p.*, i.quantity, i.warehouse
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            WHERE p.sku = ? OR p.name LIKE ?
            LIMIT 1
            """,
            (identifier, f"%{identifier}%"),
        ).fetchone()

    if not row:
        return json.dumps({"error": f"Product not found: {identifier}"})

    result = dict(row)
    result["specs"] = json.loads(result["specs"])
    return json.dumps(result)


@tool
def get_price_history(identifier: str) -> str:
    """
    Get the price history for a product to assess whether the current price
    is a good deal.

    Use this when a customer asks whether a price is good, whether an item
    has gone on sale before, or whether they should wait for a deal.

    Args:
        identifier: Product SKU (e.g. "ELEC-003") or partial product name.

    Returns:
        JSON object with current price and a list of historical prices with
        dates, ordered from oldest to most recent.
    """
    with _conn() as conn:
        product = conn.execute(
            "SELECT id, name, price FROM products WHERE sku = ? OR name LIKE ? LIMIT 1",
            (identifier, f"%{identifier}%"),
        ).fetchone()

        if not product:
            return json.dumps({"error": f"Product not found: {identifier}"})

        history = conn.execute(
            """
            SELECT price, recorded_at
            FROM price_history
            WHERE product_id = ?
            ORDER BY recorded_at ASC
            """,
            (product["id"],),
        ).fetchall()

    return json.dumps({
        "product": product["name"],
        "current_price": product["price"],
        "price_history": _rows_to_dicts(history),
    })


@tool
def get_user_orders(user_email: str, status: str = None) -> str:
    """
    Look up a user's order history by their email address.

    Use this to answer questions about past purchases, order status, delivery
    dates, or whether a customer is eligible to return something.

    Args:
        user_email: The customer's email address (e.g. "mia@example.com").
        status: Optional filter by order status. Options: "pending",
                "processing", "shipped", "delivered", "returned", "cancelled".
                Omit to return all orders.

    Returns:
        JSON list of orders with order id, status, date, total, and line items
        (product name, quantity, price paid).
    """
    with _conn() as conn:
        user = conn.execute(
            "SELECT id, name FROM users WHERE email = ?", (user_email,)
        ).fetchone()

        if not user:
            return json.dumps({"error": f"No user found with email: {user_email}"})

        sql = """
            SELECT o.id AS order_id, o.status, o.created_at, o.total,
                   p.name AS product_name, p.sku, oi.quantity, oi.price_at_purchase
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            WHERE o.user_id = ?
        """
        params: list = [user["id"]]

        if status:
            sql += " AND o.status = ?"
            params.append(status)

        sql += " ORDER BY o.created_at DESC"
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        msg = f"No orders found for {user['name']}"
        if status:
            msg += f" with status '{status}'"
        return json.dumps({"user": user["name"], "orders": [], "message": msg})

    # Group order items under each order
    orders: dict = {}
    for row in rows:
        oid = row["order_id"]
        if oid not in orders:
            orders[oid] = {
                "order_id": oid,
                "status": row["status"],
                "created_at": row["created_at"],
                "total": row["total"],
                "items": [],
            }
        orders[oid]["items"].append({
            "product": row["product_name"],
            "sku": row["sku"],
            "quantity": row["quantity"],
            "price_paid": row["price_at_purchase"],
        })

    return json.dumps({"user": user["name"], "orders": list(orders.values())})


@tool
def get_product_reviews(identifier: str, limit: int = 5) -> str:
    """
    Get customer reviews for a product from our internal review database.

    Use this when a customer asks what people think of a product, wants to
    know about common issues, or is comparing products. For broader sentiment
    or recent external reviews, use the web_search tool instead.

    Args:
        identifier: Product SKU (e.g. "ELEC-003") or partial product name.
        limit: Maximum number of reviews to return (default 5, max 20).

    Returns:
        JSON object with product name, average rating, and list of reviews
        (rating, text, date). Ordered by most recent first.
    """
    with _conn() as conn:
        product = conn.execute(
            "SELECT id, name FROM products WHERE sku = ? OR name LIKE ? LIMIT 1",
            (identifier, f"%{identifier}%"),
        ).fetchone()

        if not product:
            return json.dumps({"error": f"Product not found: {identifier}"})

        reviews = conn.execute(
            """
            SELECT r.rating, r.body, r.created_at, u.name AS reviewer
            FROM reviews r
            JOIN users u ON r.user_id = u.id
            WHERE r.product_id = ?
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (product["id"], min(limit, 20)),
        ).fetchall()

        avg = conn.execute(
            "SELECT ROUND(AVG(rating), 1) AS avg_rating, COUNT(*) AS total FROM reviews WHERE product_id = ?",
            (product["id"],),
        ).fetchone()

    return json.dumps({
        "product": product["name"],
        "average_rating": avg["avg_rating"],
        "total_reviews": avg["total"],
        "reviews": _rows_to_dicts(reviews),
    })


@tool
def get_inventory_status(identifier: str) -> str:
    """
    Check the current inventory / stock level for a product.

    Use this when a customer asks if something is in stock, available,
    or how many units remain.

    Args:
        identifier: Product SKU (e.g. "FASH-003") or partial product name.

    Returns:
        JSON object with product name, current stock quantity, and warehouse.
    """
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT p.name, p.sku, i.quantity, i.warehouse, i.last_updated
            FROM products p
            JOIN inventory i ON p.id = i.product_id
            WHERE p.sku = ? OR p.name LIKE ?
            LIMIT 1
            """,
            (identifier, f"%{identifier}%"),
        ).fetchone()

    if not row:
        return json.dumps({"error": f"Product not found: {identifier}"})

    result = dict(row)
    result["in_stock"] = result["quantity"] > 0
    return json.dumps(result)
