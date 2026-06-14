import json
import re
from langchain_core.prompts import ChatPromptTemplate
from src.shopping_agent.services.groq_client import GroqClient
from src.shopping_agent.utils.logger import agent_logger

SKIP_LABEL = "Skip / Don't Know"
OTHER_LABEL = "Other (type your own)"

FALLBACK_SPEC_OPTIONS = {
    "What matters most to you?": [SKIP_LABEL, "Lowest price", "Best quality", "Specific brand"],
}

def _extract_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
    raw_text = re.sub(r"```$", "", raw_text).strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw_text[:300]}")
    return json.loads(match.group(0))

def generate_spec_options(user_query: str) -> dict:
    agent_logger.info(f"Generating dynamic clarifying questions for: '{user_query}'")

    try:
        llm = GroqClient().get_llm()

        # NOTE: every literal { and } in the system message below is escaped
        # as {{ and }} — otherwise ChatPromptTemplate tries to treat
        # "Question text here?" as a template variable, which caused the
        # KeyError you hit.
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "A user in India wants to buy something but hasn't given enough detail. List the "
            "3 to 5 most important questions a shopping expert would ask before searching for "
            "THIS specific product, with 3-6 realistic answer options per question. Base every "
            "question and option on what this exact product is — a laptop needs processor/RAM "
            "questions, an AC needs room-size/tonnage questions, a shoe needs brand/type/size "
            "questions.\n\n"
            "IMPORTANT: This is for the Indian market. ALL prices, budgets, and currency "
            "values in your answer options MUST be in Indian Rupees using the ₹ symbol "
            "(e.g. ₹30,000-₹40,000), never in dollars ($) or any other currency.\n\n"
             "Respond with ONLY a JSON object, nothing else before or after it. "
             "Format: {{\"Question text here?\": [\"option1\", \"option2\", \"option3\"]}}"
            ),
            ("human", "Product: {query}")
        ])

        chain = prompt | llm
        result = chain.invoke({"query": user_query})

        agent_logger.info(f"RAW GROQ OUTPUT for '{user_query}': {result.content!r}")

        parsed = _extract_json(result.content)

        spec_options = {
            q: [SKIP_LABEL] + [str(v) for v in values] + [OTHER_LABEL]
            for q, values in parsed.items()
            if isinstance(values, list) and values
        }

        if not spec_options:
            agent_logger.warning("Parsed JSON was empty after filtering. Using fallback.")
            return FALLBACK_SPEC_OPTIONS

        return spec_options

    except Exception as e:
        agent_logger.error(f"Dynamic question generation FAILED for '{user_query}': {type(e).__name__}: {str(e)}", exc_info=True)
        return FALLBACK_SPEC_OPTIONS