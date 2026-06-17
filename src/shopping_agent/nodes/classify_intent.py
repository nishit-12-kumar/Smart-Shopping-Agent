import json
import re
from langchain_core.prompts import ChatPromptTemplate
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import GroqClient
from src.shopping_agent.utils.logger import agent_logger


def _extract_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
    raw_text = re.sub(r"```$", "", raw_text).strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in classification response")
    return json.loads(match.group(0))


def classify_intent_node(state: ShoppingState) -> ShoppingState:
    """
    Runs BEFORE parse_query. Decides if the new message is a NEW search
    or a FOLLOW_UP question about products already shown in the last turn.
    """
    agent_logger.info("Entering classify_intent_node.")

    last_shown_deals = state.get("last_shown_deals", [])
    user_query = state.get("user_query", "")

    # No previous deals shown yet -> nothing to follow up on, always NEW
    if not last_shown_deals:
        state["intent"] = "NEW"
        agent_logger.info(f"No previous deals in memory (count={len(last_shown_deals)}). Classified as NEW.")
        return state


    try:
        llm = GroqClient().get_llm()

        products_list = "\n".join(
            f"- {d.get('title')} (₹{d.get('price')})" for d in last_shown_deals
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are classifying a user's shopping message. Below is a list of products "
             "that were already shown to the user in the previous turn. Decide if the new "
             "message is a FOLLOW_UP question about one of those specific products "
             "(e.g. 'tell me more about the LG one', 'is the first one good for gaming', "
             "'compare option 2 and 3'), or a NEW, unrelated search request.\n\n"
             "Respond with ONLY raw JSON in this exact shape: "
             "{{\"intent\": \"FOLLOW_UP\" or \"NEW\", \"referenced_product\": \"exact product title or null\"}}"
            ),
            ("human", "Previously shown products:\n{products}\n\nNew message: {query}")
        ])

        chain = prompt | llm
        result = chain.invoke({"products": products_list, "query": user_query})

        agent_logger.info(f"RAW intent classification output: {result.content!r}")
        parsed = _extract_json(result.content)

        state["intent"] = parsed.get("intent", "NEW")
        state["search_params"] = state.get("search_params") or {}
        state["search_params"]["referenced_product"] = parsed.get("referenced_product")

        agent_logger.info(f"Classified intent: {state['intent']} | referenced: {parsed.get('referenced_product')}")

    except Exception as e:
        agent_logger.error(f"Intent classification failed: {str(e)}", exc_info=True)
        # Fail safe: treat as a NEW search rather than getting stuck
        state["intent"] = "NEW"

    return state