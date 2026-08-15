from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


# 1. Define the Strict Output Schema using Pydantic
class ProductEvaluation(BaseModel):

    # Structured LLM output for a single product's deal evaluation.
    confidence_score: int = Field(description="A score from 0 to 100 indicating how well this product matches the user's request.")
    reasoning: str = Field(description="A one-line explanation of why this score was given, considering relevance to the request, specs, and review reliability.")


class DealValidationOutput(BaseModel):

    # Structured LLM output wrapping a batch of per-product deal evaluations.
    evaluations: List[ProductEvaluation] = Field(description="List of evaluations matching the order of the input products.")


def _flag_low_review_count(product: dict) -> Optional[str]:
    rating = product.get("rating")
    reviews = product.get("reviews")

    if rating and reviews and rating >= 4.5 and reviews < 50:
        return f"⚠️ High rating ({rating}★) but only {reviews} reviews — insufficient data to fully trust this rating."
    return None


def _format_product_line(p: dict) -> str:
    """
    Formats a single product for the prompt using .get() with sensible
    fallbacks instead of direct dict indexing (p['title']). SerpAPI
    results can have partially-missing fields (serpapi_client.py's own
    docstring notes this), so p['title'] etc. would raise a raw
    KeyError and crash the whole node the first time a listing was
    missing a field, instead of degrading gracefully like the rest of
    the pipeline does.
    """
    title = p.get("title") or "Unknown product"
    price = p.get("price")
    price_text = f"₹{price}" if price is not None else "price unavailable"
    rating = p.get("rating")
    rating_text = f"{rating}" if rating is not None else "no rating"
    reviews = p.get("reviews")
    reviews_text = f"{reviews} reviews" if reviews is not None else "no review count"

    return f"- {title} | Price: {price_text} | Rating: {rating_text} ({reviews_text})"


def validate_deals_node(state: ShoppingState) -> ShoppingState:

    # 1. Evaluates the raw products using Groq LLM and assigns a confidence score.
    agent_logger.info("Entering validate_deals_node.")

    try:
        raw_products = state.get("raw_products", [])
        user_query = state.get("user_query", "")

        if not raw_products:
            agent_logger.warning("No raw products found to validate. Skipping.")
            return state

        # Initialize the Groq LLM and bind the Pydantic schema
        groq_client = get_groq_llm()
        structured_llm = groq_client.with_structured_output(DealValidationOutput)

        # 2. Create the Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert shopping assistant validating search results. "
                       "Evaluate the provided products against the user's original request. "
                       "Assign a confidence score (0-100) based ONLY on: "
                       "(1) how well the product matches what the user asked for, and "
                       "(2) how reliable the review signal looks (rating and review count together). "
                       "Do NOT factor price into the score at all — pricing risk is assessed separately "
                       "by a deterministic system, not by you. A very cheap or very expensive price is "
                       "NOT itself a reason to raise or lower this score."),
            ("human", "User Request: {query}\n\nProducts to Evaluate:\n{products}")
        ])

        # 3. Create the Chain and Execute
        chain = prompt | structured_llm

        agent_logger.info(f"Sending {len(raw_products)} products to Groq for evaluation.")

        # Format products into a clean string to save tokens
        products_text = "\n".join(
            _format_product_line(p) for p in raw_products
        )

        result = chain.invoke({"query": user_query, "products": products_text})

        # 4. Merge evaluations back into the product dictionaries
        evaluations = result.evaluations

        if len(evaluations) != len(raw_products):
            agent_logger.error(
                f"LLM returned {len(evaluations)} evaluations for "
                f"{len(raw_products)} products — counts don't match, "
                "so positional alignment can't be trusted. Falling back "
                "to unvalidated products for this batch."
            )
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(
                "Deal validation returned a mismatched number of "
                "evaluations; showing unvalidated products instead."
            )
            state["validated_deals"] = raw_products
            return state

        validated_deals = []
        for product, eval_data in zip(raw_products, evaluations):
            validated_item = product.copy()
            validated_item["confidence_score"] = eval_data.confidence_score
            validated_item["reasoning"] = eval_data.reasoning
            validated_item["review_flag"] = _flag_low_review_count(product)
            validated_deals.append(validated_item)

        state["validated_deals"] = validated_deals
        agent_logger.info("Deal validation completed successfully.")

    except Exception as e:
        agent_logger.error(f"Error in validate_deals_node: {str(e)}", exc_info=True)
        if "errors" not in state: state["errors"] = []
        state["errors"].append(f"Deal validation failed: {str(e)}")
        state["validated_deals"] = state.get("raw_products", []) # Fallback to unvalidated products

    return state
