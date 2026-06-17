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
        raise ValueError("No JSON found in message-type classification response")
    return json.loads(match.group(0))


def classify_message_type_node(state: ShoppingState) -> ShoppingState:
    """
    Runs as the VERY FIRST node, before intent/follow-up classification.
    Uses the LLM to decide whether the message is:
      - CHITCHAT: greetings, thanks, small talk, meta questions ("what can you do")
      - SHOPPING: contains any genuine product-buying intent

    If CHITCHAT, the LLM also generates the direct reply right here, so the
    graph can short-circuit straight to END without ever touching search/
    validation/pricing nodes.
    """
    agent_logger.info("Entering classify_message_type_node.")

    user_query = state.get("user_query", "")

    try:
        llm = GroqClient().get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the first-stage router for a shopping assistant chatbot. "
             "Classify the user's message into exactly one of two types:\n\n"
             "1. CHITCHAT — greetings (hi, hello), thanks, small talk, goodbyes, "
             "or meta questions about the assistant itself (e.g. 'what can you do', "
             "'who are you'). The message has NO genuine product-buying intent.\n\n"
             "2. SHOPPING — the message contains ANY genuine intent to find, buy, "
             "compare, or ask about a product or follow up on previously shown "
             "products — even if mixed with a greeting (e.g. 'hi, i want a laptop' "
             "is SHOPPING, not CHITCHAT, because real intent is present).\n\n"
             "If CHITCHAT, also write a short, warm, conversational reply (1-2 "
             "sentences) that gently steers the user toward telling you what "
             "they want to shop for.\n\n"
             "Respond with ONLY raw JSON in this exact shape: "
             "{{\"type\": \"CHITCHAT\" or \"SHOPPING\", \"reply\": \"your reply text, "
             "or null if type is SHOPPING\"}}"
            ),
            ("human", "User message: {query}")
        ])

        chain = prompt | llm
        result = chain.invoke({"query": user_query})

        agent_logger.info(f"RAW message-type classification output: {result.content!r}")
        parsed = _extract_json(result.content)

        message_type = parsed.get("type", "SHOPPING")
        state["message_type"] = message_type

        if message_type == "CHITCHAT":
            reply = parsed.get("reply") or "Hey there! What are you looking to shop for today?"
            state["final_recommendation"] = reply
            agent_logger.info(f"Classified as CHITCHAT. Reply: {reply}")
        else:
            agent_logger.info("Classified as SHOPPING. Proceeding into normal pipeline.")

    except Exception as e:
        agent_logger.error(f"classify_message_type_node failed: {str(e)}", exc_info=True)
        # Fail safe: assume SHOPPING so we never accidentally block a real request
        state["message_type"] = "SHOPPING"

    return state