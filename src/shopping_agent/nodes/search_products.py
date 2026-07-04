from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.serpapi_client import SerpApiClient
from src.shopping_agent.utils.exceptions import SerpAPIError
from src.shopping_agent.utils.logger import agent_logger

# Substrings SerpAPI tends to use when the account has run out of searches.
# Not exhaustive, but lets us give a precise message for the common case
# instead of a generic "search failed" message.
_QUOTA_INDICATORS = ("run out of searches", "exceeded", "quota", "usage limit")


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(indicator in lowered for indicator in _QUOTA_INDICATORS)


def search_products_node(state: ShoppingState) -> ShoppingState:
    """
    Executes the product search using the SerpApiClient.
    """
    agent_logger.info("Entering search_products_node.")
    
    try:
        # Prevent search if the previous node decided we need more clarification
        if state.get("clarification_needed"):
            agent_logger.info("Clarification needed. Skipping search.")
            return state

        # Safely extract the search query
        search_params = state.get("search_params") or {}
        search_query = search_params.get("query")
        
        if not search_query:
            agent_logger.warning("No search query found in state. Skipping search.")
            # Initialize errors list if it doesn't exist
            if "errors" not in state: state["errors"] = []
            state["errors"].append("Missing search parameters.")
            return state

        # Execute the search
        client = SerpApiClient()

        try:
            products = client.search_google_shopping(search_query)
        except SerpAPIError as e:
            # Distinct from "no products found" — this means SerpAPI itself
            # couldn't be reached or refused the request (commonly: search
            # quota exhausted). We surface this explicitly rather than
            # letting it look like a normal empty result.
            error_text = str(e)
            agent_logger.error(f"SerpAPI unavailable: {error_text}")

            if _is_quota_error(error_text):
                state["serpapi_error_message"] = (
                    "⚠️ SerpAPI search quota has been used up for now, so I can't fetch "
                    "live product data. Please try again later once the quota resets."
                )
            else:
                state["serpapi_error_message"] = (
                    f"⚠️ SerpAPI is currently unavailable ({error_text}). "
                    "Please try again in a moment."
                )

            if "errors" not in state: state["errors"] = []
            state["errors"].append(f"SerpAPI failure: {error_text}")
            state["raw_products"] = []
            return state

        # Update the state
        state["raw_products"] = products
        
        if not products:
            if "errors" not in state: state["errors"] = []
            state["errors"].append("No products found for the given query.")

    except Exception as e:
        agent_logger.error(f"Error in search_products_node: {str(e)}", exc_info=True)
        if "errors" not in state: state["errors"] = []
        state["errors"].append(f"Search node failed: {str(e)}")
        state["raw_products"] = []
        
    return state