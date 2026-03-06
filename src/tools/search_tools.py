"""
Semantic search tool for fuzzy/natural-language product queries via Chroma.
Used by the semantic_agent for "find me something like X" queries.
"""

import json
from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = Path(__file__).parent.parent.parent / "data" / "chroma"

_store: Chroma | None = None


def _get_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name="products",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=str(CHROMA_PATH),
        )
    return _store


@tool
def semantic_search(
    query: str,
    category: str = None,
    max_price: float = None,
    limit: int = 5,
) -> str:
    """
    Search for products using natural language and semantic similarity.

    Use this for fuzzy, descriptive, or vibe-based queries where the customer
    doesn't know the exact product name — e.g. "something cozy for winter",
    "a gift for someone who loves the outdoors", or "wireless headphones good
    for commuting". For exact product lookups by name or SKU, use search_products.

    Args:
        query: Natural language description of what the customer is looking for.
        category: Optional category filter. Options: "Electronics", "Footwear",
                  "Clothing", "Activewear", "Fashion", "Beauty & Wellness", "Home".
        max_price: Optional maximum price in USD.
        limit: Number of results to return (default 5).

    Returns:
        JSON list of semantically similar products with name, brand, category,
        price, and a relevance score (lower = more similar).
    """
    store = _get_store()

    where: dict = {}
    if category:
        where["category"] = category
    if max_price is not None:
        where["price"] = {"$lte": max_price}

    results = store.similarity_search_with_score(
        query,
        k=limit,
        filter=where if where else None,
    )

    hits = []
    for doc, score in results:
        hits.append({
            "name": doc.metadata["name"],
            "brand": doc.metadata["brand"],
            "category": doc.metadata["category"],
            "subcategory": doc.metadata["subcategory"],
            "sku": doc.metadata["sku"],
            "price": doc.metadata["price"],
            "relevance_score": round(score, 3),
            "description_snippet": doc.page_content[:200],
        })

    if not hits:
        return json.dumps({"results": [], "message": "No matching products found."})
    return json.dumps({"results": hits})
