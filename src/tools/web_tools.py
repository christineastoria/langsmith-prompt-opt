"""
Web search tool via Tavily.
Used by the web_agent for current prices, external reviews, and competitor info.
"""

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

_tavily = TavilySearch(max_results=5)


@tool
def web_search(query: str, search_type: str = "general") -> str:
    """
    Search the web for current information about products, prices, and reviews.

    Use this for:
    - Competitor pricing and availability ("how much is the Sony XM6 at Best Buy?")
    - Recent external reviews and expert opinions ("rtings.com review Sony XM6")
    - Current deals, promotions, or sales events
    - Product availability at other retailers
    - News about a brand or product (recalls, new releases, lawsuits)

    Do NOT use this for:
    - Information available in our internal product catalog (use search_products)
    - Customer's own order history (use get_user_orders)
    - Our own return or warranty policies (use the docs tools)

    Args:
        query: The search query. Be specific — include brand and product name
               for best results (e.g. "Sony WH-1000XM6 price Amazon 2025").
        search_type: Type of search to guide query framing. Options:
                     "general" (default), "price_comparison", "reviews",
                     "news", "availability".

    Returns:
        String summary of top web search results with sources.
    """
    if search_type == "price_comparison":
        query = f"{query} price buy online"
    elif search_type == "reviews":
        query = f"{query} review expert opinion"
    elif search_type == "news":
        query = f"{query} news 2025"
    elif search_type == "availability":
        query = f"{query} in stock available"

    result = _tavily.invoke(query)
    return str(result)
