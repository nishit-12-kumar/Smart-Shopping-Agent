from langchain_core.prompts import ChatPromptTemplate
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


def _resolve_referenced_product(last_shown_deals: list, product_id: str) -> dict | None:
    if not product_id:
        return None
    return next((d for d in last_shown_deals if d.get("id") == product_id), None)


def _format_history_for_prompt(search_history: list) -> str:
    blocks = []
    for turn in search_history[:-1]:  # exclude the most recent turn, that's covered by last_shown_deals in full detail
        lines = "\n".join(f"  [{p.get('id')}] {p.get('title')} — ₹{p.get('price')}" for p in turn.get("products", []))
        blocks.append(f"Earlier search: \"{turn.get('query')}\"\n{lines}")
    return "\n\n".join(blocks) if blocks else "None"


def answer_followup_node(state: ShoppingState) -> ShoppingState:
    agent_logger.info("Entering answer_followup_node.")

    state["structured_recommendation"] = None

    try:
        user_query = state.get("user_query", "")
        last_shown_deals = state.get("last_shown_deals", [])
        search_history = state.get("search_history", [])
        referenced_id = (state.get("search_params") or {}).get("referenced_product_id")

        referenced_product = _resolve_referenced_product(last_shown_deals, referenced_id)

        deals_text = "\n".join(
            f"- [{d.get('id')}] {d.get('title')} | ₹{d.get('price')} | Score: {d.get('confidence_score')}/100 "
            f"| Reasoning: {d.get('reasoning')} | Source: {d.get('source')}"
            for d in last_shown_deals
        )

        earlier_history_text = _format_history_for_prompt(search_history)

        if referenced_product:
            referenced_text = f"[{referenced_product.get('id')}] {referenced_product.get('title')}"
        elif referenced_id:
            # classify_intent gave an id, but it didn't match anything in last_shown_deals
            # (e.g. it referred to an earlier search) — say so plainly instead of guessing.
            referenced_text = f"id {referenced_id} (not in the most recent search results — check earlier searches below)"
        else:
            referenced_text = "unclear — use best judgement from the question and the product list"

        llm = get_groq_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "The user is asking a follow-up question about products already shown to them. "
             "Answer directly and conversationally using ONLY the data provided below. "
             "Do not search for new products or invent details not present in the data."),
            ("human",
             "Most recently shown products:\n{deals}\n\n"
             "Earlier searches this session (for context only, less detail):\n{earlier_history}\n\n"
             "User is likely referring to: {referenced}\n\n"
             "User's follow-up question: {query}")
        ])

        chain = prompt | llm

        agent_logger.info("Streaming follow-up answer from memory, no new search performed.")

        full_response = ""
        for chunk in chain.stream({
            "deals": deals_text,
            "earlier_history": earlier_history_text,
            "referenced": referenced_text,
            "query": user_query
        }):
            full_response += chunk.content

        state["final_recommendation"] = full_response
        agent_logger.info("Follow-up answered directly from memory, no new search performed.")

    except Exception as e:
        agent_logger.error(f"answer_followup_node failed: {str(e)}", exc_info=True)
        state["final_recommendation"] = (
            "I had trouble answering that follow-up — could you rephrase or ask a new search instead?"
        )

    return state
