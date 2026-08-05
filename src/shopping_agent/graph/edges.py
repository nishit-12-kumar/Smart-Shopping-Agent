from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.utils.logger import agent_logger

def route_after_parse(state: ShoppingState) -> str:
    if state.get("clarification_needed"):
        agent_logger.info("Clarification needed. Routing -> END (UI handles the spec form).")
        return "end_clarification"
    
    agent_logger.info("Routing -> search_products")
    return "search_products"

def route_after_intent(state: ShoppingState) -> str:

    if state.get("intent") == "FOLLOW_UP":
        agent_logger.info("Routing -> answer_followup")
        return "answer_followup"

    agent_logger.info("Routing -> parse_query")
    return "parse_query"


def route_after_message_type(state: ShoppingState) -> str:

    if state.get("message_type") == "CHITCHAT":
        agent_logger.info("Routing -> END (chitchat handled directly)")
        return "end_chitchat"
    agent_logger.info("Routing -> classify_intent")
    return "classify_intent"