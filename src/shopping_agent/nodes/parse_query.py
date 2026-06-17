from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.utils.logger import agent_logger

def parse_query_node(state: ShoppingState) -> ShoppingState:
    """
    Decides whether the query has enough detail to search directly,
    or needs clarification — works for ANY product, not just
    hardcoded categories like laptop/AC.
    """
    agent_logger.info(f"Entering parse_query_node with query: {state['user_query']}")

    try:
        query = state.get("user_query", "").strip()
        query_lower = query.lower()

        # Signals that the user ALREADY gave specific configuration details,
        # so we don't need to ask again. This is generic, not category-specific.
        has_numeric_spec = any(char.isdigit() for char in query)  # e.g. "60k", "16GB", "1.5 ton"
        has_spec_keywords = any(word in query_lower for word in [
            "gb", "ton", "star", "inch", "ram", "processor", "storage",
            "size", "color", "colour", "with", "under", "budget"
        ])

        # Word count check: very short queries like "I want to buy a shoe"
        # almost never contain real configuration info
        meaningful_word_count = len(query.split())

        needs_clarification = not (has_numeric_spec or has_spec_keywords) or meaningful_word_count <= 6

        if needs_clarification:
            agent_logger.warning("Query lacks specific configuration. Requesting clarification UI.")
            state["clarification_needed"] = True
            state["search_params"] = None
        else:
            agent_logger.info("Query already has enough specs. Proceeding to search.")
            state["clarification_needed"] = False
            state["search_params"] = {"query": query}

    except Exception as e:
        agent_logger.error(f"Error in parse_query_node: {str(e)}", exc_info=True)
        # Default to asking for clarification rather than guessing wrong
        state["clarification_needed"] = True
        state["search_params"] = None

    return state


