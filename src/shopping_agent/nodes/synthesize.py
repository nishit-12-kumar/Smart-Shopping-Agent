from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import GroqClient
from src.shopping_agent.utils.logger import agent_logger

MAX_ALTERNATIVES = 2

# --- Output schema: LLM only fills in REASONING, never prices/titles/IDs ---
# This guarantees the card always shows real data — the LLM can't hallucinate
# a price or product name since those are taken directly from validated_deals.

class TopPickAnalysis(BaseModel):
    why_it_wins: str = Field(description="One punchy sentence on why this is the best match for the user's request.")
    specs_matched: List[str] = Field(description="2-4 short tags of specs/preferences this product satisfies, e.g. '5G', '128GB storage'.")
    specs_warning: Optional[str] = Field(default=None, description="One short tag for any spec NOT confirmed or slightly off, e.g. 'Brand not specified'. Null if nothing to flag.")


class SynthesisOutput(BaseModel):
    top_pick_analysis: TopPickAnalysis
    alternative_trade_offs: List[str] = Field(description="One short trade-off sentence per alternative product, in the SAME ORDER the alternatives were given.")
    filtered_out_note: Optional[str] = Field(default=None, description="One short sentence noting how many other options were filtered out and why, e.g. '2 more options filtered out — low review count or price mismatch.' Null if there were no other options.")
    red_flags: List[str] = Field(default_factory=list, description="One short sentence per suspicious-pricing or low-trust product, naming the product. Empty list if none.")
    bottom_line: str = Field(description="One concluding sentence telling the user what to do next.")
    follow_up_suggestions: List[str] = Field(description="Exactly 2 short tappable follow-up questions the user might ask next, phrased as the user would type them, e.g. 'Show me cheaper options'.")


def _build_synthesis_chain():
    groq_client = GroqClient().get_llm()
    structured_llm = groq_client.with_structured_output(SynthesisOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert shopping assistant. You will be given a TOP PICK product, "
         "a list of ALTERNATIVE products, and a count of FILTERED OUT products. "
         "Generate only the reasoning fields requested in the schema — do NOT restate "
         "prices, titles, or sources, those are handled separately. Be concise: every "
         "field should be a short phrase or single sentence, never a paragraph."),
        ("human",
         "User request: {query}\n\n"
         "Top pick:\n{top_pick}\n\n"
         "Alternatives:\n{alternatives}\n\n"
         "Filtered out count: {filtered_count}\n\n"
         "Any suspicious pricing flags:\n{red_flag_data}")
    ])

    return prompt | structured_llm


def _format_product_line(p: dict) -> str:
    line = f"{p.get('title')} | ₹{p.get('price')} | Confidence: {p.get('confidence_score')}/100"
    if p.get("rating"):
        line += f" | Rating: {p.get('rating')} ({p.get('reviews')} reviews)"
    return line


def synthesize_node(state: ShoppingState) -> ShoppingState:
    """
    Selects the top pick and alternatives DETERMINISTICALLY in code (sorted
    by confidence_score), then asks the LLM only for the qualitative
    reasoning fields (why it wins, trade-offs, red flags, bottom line,
    follow-up chips). This guarantees the UI never shows a hallucinated
    price or product name — those always come straight from validated_deals.

    NOTE: structured output (Pydantic schema via with_structured_output)
    does not support token-by-token streaming — it returns the full object
    at once via tool-calling under the hood. This node therefore uses
    .invoke() instead of .stream(). The "Agent is reasoning..." status box
    in app.py still shows live node-progress, so the UI doesn't feel frozen,
    it just won't have a typewriter effect for this specific node anymore.
    """
    agent_logger.info("Entering synthesize_node.")

    try:
        user_query = state.get("user_query", "")
        deals = state.get("validated_deals", [])

        if not deals:
            serpapi_error_message = state.get("serpapi_error_message")
            if serpapi_error_message:
                agent_logger.warning("No deals to synthesize — SerpAPI was unavailable.")
                state["final_recommendation"] = serpapi_error_message
            else:
                agent_logger.warning("No deals to synthesize. Returning fallback message.")
                state["final_recommendation"] = (
                    "I couldn't find any products that meet your criteria with high confidence. "
                    "Could we try adjusting your budget or brand preferences?"
                )
            state["structured_recommendation"] = None
            return state

        sorted_deals = sorted(deals, key=lambda d: d.get("confidence_score", 0), reverse=True)
        top_pick = sorted_deals[0]
        alternatives = sorted_deals[1:1 + MAX_ALTERNATIVES]
        filtered_count = max(0, len(sorted_deals) - 1 - len(alternatives))

        red_flag_products = [d for d in sorted_deals if d.get("is_suspicious_pricing")]
        red_flag_data = "\n".join(
            f"- {d.get('title')}: {d.get('pricing_analysis', '')}" for d in red_flag_products
        ) or "None"

        top_pick_text = _format_product_line(top_pick)
        alternatives_text = "\n".join(_format_product_line(a) for a in alternatives) or "None"

        chain = _build_synthesis_chain()

        agent_logger.info("Sending top pick + alternatives to Groq for structured synthesis generation.")
        result: SynthesisOutput = chain.invoke({
            "query": user_query,
            "top_pick": top_pick_text,
            "alternatives": alternatives_text,
            "filtered_count": filtered_count,
            "red_flag_data": red_flag_data
        })

        # Assemble the final structure: REAL product data + LLM reasoning
        structured = {

            "top_pick": {
                "title": top_pick.get("title"),
                "price": top_pick.get("price"),
                "source": top_pick.get("source"),
                "link": top_pick.get("link"),
                "thumbnail": top_pick.get("thumbnail"),
                "confidence_score": top_pick.get("confidence_score"),
                "rating": top_pick.get("rating"),
                "reviews": top_pick.get("reviews"),
                "why_it_wins": result.top_pick_analysis.why_it_wins,
                "specs_matched": result.top_pick_analysis.specs_matched,
                "specs_warning": result.top_pick_analysis.specs_warning,
            },

            "alternatives": [
                {
                    "title": alt.get("title"),
                    "price": alt.get("price"),
                    "thumbnail": alt.get("thumbnail"),
                    "link": alt.get("link"),
                    "is_suspicious_pricing": alt.get("is_suspicious_pricing", False),
                    "trade_off": (
                        result.alternative_trade_offs[i]
                        if i < len(result.alternative_trade_offs) else ""
                    ),
                }
                for i, alt in enumerate(alternatives)
            ],
            
            "filtered_out_note": result.filtered_out_note,
            "red_flags": result.red_flags,
            "bottom_line": result.bottom_line,
            "follow_up_suggestions": result.follow_up_suggestions,
        }

        state["structured_recommendation"] = structured

        agent_logger.info(f"DEBUG top_pick link: {structured['top_pick'].get('link')!r}")
        agent_logger.info(f"DEBUG alternatives links: {[a.get('link') for a in structured['alternatives']]!r}")

        # Keep a flat text version too, for logging/follow-up context — not
        # rendered directly in the UI anymore, but useful for answer_followup_node
        state["final_recommendation"] = (
            f"Top pick: {top_pick.get('title')} at ₹{top_pick.get('price')}. {result.bottom_line}"
        )

        agent_logger.info("Successfully generated structured recommendation.")

        state["last_shown_deals"] = deals
        state.setdefault("conversation_history", []).append({
            "user_query": user_query,
            "products_shown": [d.get("title") for d in deals]
        })

    except Exception as e:
        agent_logger.error(f"Error in synthesize_node: {str(e)}", exc_info=True)
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Synthesis failed: {str(e)}")
        state["structured_recommendation"] = None
        state["final_recommendation"] = (
            "I ran into an issue while putting together your final recommendations. "
            "Please check the logs or try again."
        )

    return state
