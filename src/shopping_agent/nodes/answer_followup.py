from langchain_core.prompts import ChatPromptTemplate
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import GroqClient
from src.shopping_agent.utils.logger import agent_logger

def answer_followup_node(state: ShoppingState) -> ShoppingState:
    """
    Answers a follow-up question directly using products already in memory —
    skips search/validate/price-check entirely, saving API calls and time.

    NOTE ON STREAMING: uses chain.stream(...) instead of chain.invoke(...)
    so the underlying LLM call runs in streaming mode. The live typewriter
    effect itself is produced at the app.py level via LangGraph's
    stream_mode=["updates", "messages"], which intercepts these same LLM
    callbacks node-by-node — not by anything pushed manually from here.
    """
    agent_logger.info("Entering answer_followup_node.")

    # IMPORTANT: graph_state is a single dict that persists across the whole
    # conversation. The last real search left `structured_recommendation`
    # populated with the previous card's data, and nothing clears it
    # automatically. app.py decides what to render with:
    #   if structured and not clarification_needed: render the card
    #   elif plain_recommendation: render this node's text answer
    # Since `structured` was still truthy from the earlier turn, every
    # follow-up answer generated here was being silently discarded in favor
    # of re-rendering the stale card. Clearing it here is what lets app.py
    # actually show the fresh answer below instead.
    state["structured_recommendation"] = None

    try:
        user_query = state.get("user_query", "")
        last_shown_deals = state.get("last_shown_deals", [])
        referenced = (state.get("search_params") or {}).get("referenced_product")

        deals_text = "\n".join(
            f"- {d.get('title')} | ₹{d.get('price')} | Score: {d.get('confidence_score')}/100 "
            f"| Reasoning: {d.get('reasoning')} | Source: {d.get('source')}"
            for d in last_shown_deals
        )

        llm = GroqClient().get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "The user is asking a follow-up question about products already shown to them. "
             "Answer directly and conversationally using ONLY the data provided below. "
             "Do not search for new products or invent details not present in the data."),
            ("human",
             "Previously shown products:\n{deals}\n\n"
             "User is likely referring to: {referenced}\n\n"
             "User's follow-up question: {query}")
        ])

        chain = prompt | llm

        agent_logger.info("Streaming follow-up answer from memory, no new search performed.")

        full_response = ""
        for chunk in chain.stream({
            "deals": deals_text,
            "referenced": referenced or "unclear — use best judgement from the question",
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