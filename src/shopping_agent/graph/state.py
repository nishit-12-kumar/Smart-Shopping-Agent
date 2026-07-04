from typing import TypedDict, List, Dict, Any, Optional

class ShoppingState(TypedDict):
    """
    Represents the state of the shopping agent throughout its execution.
    """
    user_query: str
    clarification_needed: bool
    search_params: Optional[Dict[str, Any]]
    raw_products: List[Dict[str, Any]]
    serpapi_error_message: Optional[str] 
    validated_deals: List[Dict[str, Any]]
    message_type: str   # "CHITCHAT" or "SHOPPING"
    final_recommendation: str
    structured_recommendation: Optional[Dict[str, Any]]   
    errors: List[str]

    # --- Multi-turn memory fields ---
    conversation_history: List[Dict[str, Any]]
    last_shown_deals: List[Dict[str, Any]]
    intent: str   
