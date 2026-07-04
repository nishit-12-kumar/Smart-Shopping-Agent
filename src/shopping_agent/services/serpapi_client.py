from typing import List, Dict, Any
from serpapi import GoogleSearch
from src.shopping_agent.config import SERPAPI_API_KEY
from src.shopping_agent.utils.logger import agent_logger
from src.shopping_agent.utils.exceptions import SerpAPIError


class SerpApiClient:
    def __init__(self):
        self.api_key = SERPAPI_API_KEY

    def search_google_shopping(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetches product data from Google Shopping via SerpAPI.

        Raises SerpAPIError (rather than silently swallowing the problem)
        whenever SerpAPI itself reports a failure — e.g. quota exhausted,
        invalid key, or a connection failure. This is deliberately NOT
        caught here: search_products_node decides how to surface it to the
        user (a clear "search quota exhausted" message), instead of this
        client silently returning an empty/fake result set that looks like
        "no products matched" when the real problem is "the API is down".
        """
        agent_logger.info(f"Initiating SerpAPI Google Shopping search for: '{query}'")

        params = {
            "engine": "google_shopping",
            "q": query,
            "hl": "en",
            "gl": "in",  # Setting geolocation to India for relevant local pricing
            "api_key": self.api_key,
            "num": 5     # Limit results to save LLM tokens downstream
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as e:
            agent_logger.error(f"Failed to connect to SerpAPI: {str(e)}", exc_info=True)
            raise SerpAPIError(f"Could not reach SerpAPI: {str(e)}") from e

        if "error" in results:
            error_msg = results["error"]
            agent_logger.error(f"SerpAPI returned an error: {error_msg}")
            raise SerpAPIError(error_msg)

        shopping_results = results.get("shopping_results", [])

        if not shopping_results:
            agent_logger.warning("SerpAPI returned empty shopping results.")
            return []

        # Clean and structure the payload
        parsed_results = []
        for item in shopping_results[:5]:
            # NOTE: for the `google_shopping` engine, SerpAPI returns the
            # product URL under `product_link`, NOT `link` — `link` only
            # shows up on inline/related/featured result types, which we
            # never request here. Using the wrong key silently produced
            # `None` for every single product (confirmed in logs), which
            # made every "View deal" button fall back to a generic
            # constructed search URL instead of the real listing.
            link = item.get("product_link") or item.get("link")

            parsed_results.append({
                "title": item.get("title"),
                "price": item.get("extracted_price"), # Extracts clean float/int
                "link": link,
                "source": item.get("source"),
                "rating": item.get("rating"),
                "reviews": item.get("reviews"),
                "thumbnail": item.get("thumbnail")   # ADD THIS — product image URL
            })

            if not link:
                agent_logger.warning(
                    f"No product_link/link found for '{item.get('title')}' — "
                    "UI will fall back to a constructed search URL for this item."
                )

        agent_logger.info(f"Successfully retrieved {len(parsed_results)} products from SerpAPI.")
        return parsed_results