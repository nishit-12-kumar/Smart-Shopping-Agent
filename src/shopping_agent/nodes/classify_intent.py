from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


class IntentClassification(BaseModel):
    intent: str = Field(description="Either 'FOLLOW_UP' if the message is about a product already shown, or 'NEW' if it's a fresh search request.")
    referenced_product_id: Optional[str] = Field(
        default=None,
        description="The id (e.g. 'p1', 'p2') of the specific product the user is referring to, "
                    "from the ids listed in the product history below. Null if intent is NEW, or "
                    "if it's a FOLLOW_UP that doesn't point at one specific product (e.g. 'show me cheaper options')."
    )


def _format_history_for_prompt(search_history: list) -> str:
    blocks = []
    for turn in search_history:
        lines = "\n".join(
            f"  [{p.get('id')}] {p.get('title')} — ₹{p.get('price')}"
            for p in turn.get("products", [])
        )
        blocks.append(f"Search: \"{turn.get('query')}\"\n{lines}")
    return "\n\n".join(blocks)


def classify_intent_node(state: ShoppingState) -> ShoppingState:
    agent_logger.info("Entering classify_intent_node.")

    search_history = state.get("search_history", [])
    user_query = state.get("user_query", "")

    if not search_history:
        state["intent"] = "NEW"
        agent_logger.info("No previous search history in memory. Classified as NEW.")
        return state

    try:
        llm = get_groq_llm()
        structured_llm = llm.with_structured_output(IntentClassification)

        history_text = _format_history_for_prompt(search_history)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are classifying a user's shopping message. Below are the last few searches "
             "already shown to the user, each with an id in brackets before every product "
             "(e.g. [p2]). Decide if the new message is a FOLLOW_UP question about one or more "
             "of these specific products (e.g. 'tell me more about the LG one', 'is the first "
             "one good for gaming', 'compare option 2 and 3'), or a NEW, unrelated search "
             "request. If it's a FOLLOW_UP about one clear product, return that product's id "
             "as referenced_product_id — do not return its title, only the id shown in brackets."
            ),
            ("human", "Recent search history:\n{history}\n\nNew message: {query}")
        ])

        chain = prompt | structured_llm
        result: IntentClassification = chain.invoke({"history": history_text, "query": user_query})

        agent_logger.info(
            f"Intent classification result: intent={result.intent}, "
            f"referenced_product_id={result.referenced_product_id}"
        )

        state["intent"] = result.intent if result.intent in ("FOLLOW_UP", "NEW") else "NEW"

        state["search_params"] = state.get("search_params") or {}
        state["search_params"]["referenced_product_id"] = result.referenced_product_id
        """
        State looks like this after classification:

        state = {
            "intent": "FOLLOW_UP",
            "search_params": {
                "referenced_product_id": "p2"
            }
        }

        Downstream (answer_followup_node), "p2" is looked up directly against
        the product summaries in search_history — no fuzzy title matching.
        """
    except Exception as e:
        agent_logger.error(f"Intent classification failed: {str(e)}", exc_info=True)
        # Fail safe: treat as a NEW search rather than getting stuck
        state["intent"] = "NEW"

    return state