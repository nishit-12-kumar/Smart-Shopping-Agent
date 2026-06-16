from langgraph.graph import StateGraph, END
from src.shopping_agent.graph.state import ShoppingState

# Import Nodes
from src.shopping_agent.nodes.parse_query import parse_query_node
from src.shopping_agent.nodes.search_products import search_products_node
from src.shopping_agent.nodes.validate_deals import validate_deals_node
from src.shopping_agent.nodes.price_validity import price_validity_node
from src.shopping_agent.nodes.synthesize import synthesize_node

from src.shopping_agent.nodes.classify_intent import classify_intent_node
from src.shopping_agent.nodes.answer_followup import answer_followup_node
from src.shopping_agent.graph.edges import route_after_parse, route_after_intent, route_after_message_type
from src.shopping_agent.nodes.classify_message_type import classify_message_type_node

# Import Routing Logic
from src.shopping_agent.utils.logger import agent_logger

def build_graph():
    """Compiles and returns the LangGraph state machine."""
    agent_logger.info("Compiling the LangGraph state machine.")
    
    builder = StateGraph(ShoppingState)

    # 1. Add all nodes
    builder.add_node("classify_message_type", classify_message_type_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("answer_followup", answer_followup_node)
    
    builder.add_node("parse_query", parse_query_node)
    builder.add_node("search_products", search_products_node)
    builder.add_node("validate_deals", validate_deals_node)
    builder.add_node("price_validity", price_validity_node) # <-- Added node
    builder.add_node("synthesize", synthesize_node)

    # 2. Define the execution flow
    builder.set_entry_point("classify_message_type")

    builder.add_conditional_edges(
        "classify_message_type",
        route_after_message_type,
        {
            "end_chitchat": END,
            "classify_intent": "classify_intent"
        }
    )

    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "answer_followup": "answer_followup",
            "parse_query": "parse_query"
        }
    )

    builder.add_edge("answer_followup", END)
    
    builder.add_conditional_edges(
        "parse_query", 
        route_after_parse,
        {
            "end_clarification": END,
            "search_products": "search_products"
        }
    )
    
    # Linear flow: Search -> Validate -> Check Prices
    builder.add_edge("search_products", "validate_deals")
    builder.add_edge("validate_deals", "price_validity") # <-- New Edge
    builder.add_edge("price_validity", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()

