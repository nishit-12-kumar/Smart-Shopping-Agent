from typing import TypedDict, List, Dict, Any, Optional

class ShoppingState(TypedDict):
    """
    Represents the state of the shopping agent throughout its execution.
    """
    user_query: str
    clarification_needed: bool
    search_params: Optional[Dict[str, Any]]
    raw_products: List[Dict[str, Any]]
    validated_deals: List[Dict[str, Any]]
    message_type: str   # "CHITCHAT" or "SHOPPING"
    final_recommendation: str
    structured_recommendation: Optional[Dict[str, Any]]   # ADDED — card-based response data
    errors: List[str]
    # --- Multi-turn memory fields ---
    conversation_history: List[Dict[str, Any]]
    last_shown_deals: List[Dict[str, Any]]
    intent: str   # "NEW" or "FOLLOW_UP"
