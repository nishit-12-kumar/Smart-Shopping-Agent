from typing import TypedDict, List, Dict, Any, Optional

class ShoppingState(TypedDict):

    user_query: str                                # Raw text of the user's latest message
    clarification_needed: bool                     # True if parse_query couldn't extract enough specs to search
    search_params: Optional[Dict[str, Any]]        # Structured specs (category, budget, etc.) extracted by parse_query
    raw_products: List[Dict[str, Any]]             # Unprocessed listings returned by search_products (SerpAPI or mock fallback)
    serpapi_error_message: Optional[str]           # Set if the SerpAPI call fails, so the UI/logs can surface the reason
    validated_deals: List[Dict[str, Any]]          # Products after confidence scoring (validate_deals) and price-sanity checks (price_validity)
    message_type: str   # "CHITCHAT" or "SHOPPING"  -- set by classify_message_type_node, drives the first routing decision
    final_recommendation: str                      # LLM-generated reasoning text produced by synthesize_node
    structured_recommendation: Optional[Dict[str, Any]]   # Deterministic top pick + alternatives assembled in Python by synthesize_node
    errors: List[str]                              # Accumulated non-fatal error messages from any node in the pipeline

    # --- Multi-turn memory fields ---
    conversation_history: List[Dict[str, Any]]     # Full chat history for the session, used for follow-up context
    last_shown_deals: List[Dict[str, Any]]         # Most recently shown product set, referenced by answer_followup_node
    intent: str   # "NEW" or "FOLLOW_UP" -- set by classify_intent_node, drives routing after message-type classification
