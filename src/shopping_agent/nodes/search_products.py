from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.serpapi_client import SerpApiClient
from src.shopping_agent.utils.logger import agent_logger

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
        products = client.search_google_shopping(search_query)
        
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