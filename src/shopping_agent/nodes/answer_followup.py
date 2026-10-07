from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


# ============================================================================
# PRODUCT MEMORY RESOLUTION
# ============================================================================


def _resolve_referenced_product(
    product_memory: Dict[str, Dict[str, Any]],
    product_id: str | None,
) -> Dict[str, Any] | None:
    """
    Resolve a product reference using the session-wide product memory.

    Product IDs such as:
        s1p1
        s1p2
        s2p3

    are assigned by synthesize_node().

    Unlike the old implementation, this function does NOT search only
    last_shown_deals. Therefore products from older searches can also be
    resolved.
    """
    if not product_id:
        return None

    return product_memory.get(str(product_id))


# ============================================================================
# HISTORY FORMATTING
# ============================================================================


def _format_history_for_prompt(
    search_history: List[Dict[str, Any]],
) -> str:
    """
    Format recent search history for the follow-up LLM.

    History contains lightweight product summaries, not complete product
    dictionaries. Full details for a referenced product come from
    product_memory.
    """
    if not search_history:
        return "None"

    blocks = []

    for turn in search_history:
        query = turn.get("query", "")

        products = turn.get("products", [])

        lines = []

        for product in products:
            product_id = product.get("id")
            title = product.get("title")
            price = product.get("price")

            price_text = (
                f"₹{price}"
                if price is not None
                else "price unavailable"
            )

            lines.append(
                f"  [{product_id}] "
                f"{title} — {price_text}"
            )

        if lines:
            blocks.append(
                f'Search: "{query}"\n'
                + "\n".join(lines)
            )

    return (
        "\n\n".join(blocks)
        if blocks
        else "None"
    )


# ============================================================================
# FULL PRODUCT FORMATTING
# ============================================================================


def _format_product_for_prompt(
    product: Dict[str, Any],
) -> str:
    """
    Convert a full product record into controlled, factual text for the LLM.

    We intentionally do not dump the entire dictionary into the prompt.
    Only fields useful for answering shopping follow-ups are included.
    """

    if not product:
        return "No product data available."

    lines = []

    product_id = product.get("id")
    title = product.get("title")
    price = product.get("price")
    rating = product.get("rating")
    reviews = product.get("reviews")

    lines.append(
        f"Product ID: {product_id}"
    )

    lines.append(
        f"Title: {title}"
    )

    if price is not None:
        lines.append(
            f"Price: ₹{price}"
        )

    if rating is not None:
        rating_line = f"Rating: {rating}/5"

        if reviews is not None:
            rating_line += f" ({reviews} reviews)"

        lines.append(rating_line)

    # Common product specification fields.
    spec_fields = (
        "brand",
        "category",
        "ram",
        "storage",
        "processor",
        "gpu",
        "display",
        "network",
        "battery",
        "operating_system",
    )

    specifications = []

    for field in spec_fields:
        value = product.get(field)

        if value is not None and str(value).strip():
            specifications.append(
                f"{field}: {value}"
            )

    if specifications:
        lines.append(
            "Specifications: "
            + ", ".join(specifications)
        )

    confidence = product.get(
        "confidence_score"
    )

    if confidence is not None:
        lines.append(
            f"Relevance Confidence: "
            f"{confidence}/100"
        )

    recommendation_score = product.get(
        "recommendation_score"
    )

    if recommendation_score is not None:
        lines.append(
            f"Recommendation Score: "
            f"{recommendation_score}/100"
        )

    pricing_risk = product.get(
        "pricing_risk_score"
    )

    pricing_risk_level = product.get(
        "pricing_risk_level"
    )

    if pricing_risk is not None:
        risk_text = (
            f"Pricing Risk: {pricing_risk}/100"
        )

        if pricing_risk_level:
            risk_text += (
                f" ({pricing_risk_level})"
            )

        lines.append(risk_text)

    pricing_reasons = product.get(
        "pricing_risk_reasons"
    )

    if isinstance(pricing_reasons, list):
        clean_reasons = [
            str(reason).strip()
            for reason in pricing_reasons
            if str(reason).strip()
        ]

        if clean_reasons:
            lines.append(
                "Pricing Risk Reasons: "
                + "; ".join(clean_reasons)
            )

    reasoning = product.get(
        "reasoning"
    )

    if reasoning:
        lines.append(
            f"Validation Reasoning: {reasoning}"
        )

    source = product.get("source")

    if source:
        lines.append(
            f"Source: {source}"
        )

    return "\n".join(lines)


# ============================================================================
# LATEST PRODUCTS FORMATTING
# ============================================================================


def _format_latest_products_for_prompt(
    last_shown_deals: List[Dict[str, Any]],
) -> str:
    """
    Format the latest search results.

    These products already contain their full information because
    last_shown_deals intentionally stores the most recent search in detail.
    """

    if not last_shown_deals:
        return "None"

    blocks = []

    for product in last_shown_deals:

        product_id = product.get("id")
        title = product.get("title")
        price = product.get("price")
        confidence = product.get("confidence_score")
        recommendation_score = product.get(
            "recommendation_score"
        )
        source = product.get("source")

        price_text = (
            f"₹{price}"
            if price is not None
            else "price unavailable"
        )

        block = [
            f"[{product_id}] {title}",
            f"Price: {price_text}",
        ]

        if confidence is not None:
            block.append(
                f"Confidence: {confidence}/100"
            )

        if recommendation_score is not None:
            block.append(
                f"Recommendation Score: "
                f"{recommendation_score}/100"
            )

        if source:
            block.append(
                f"Source: {source}"
            )

        blocks.append(
            " | ".join(block)
        )

    return "\n".join(blocks)


# ============================================================================
# FOLLOW-UP NODE
# ============================================================================


def answer_followup_node(
    state: ShoppingState,
) -> ShoppingState:
    """
    Answer a follow-up question using products already present in memory.

    Important behavior:

    1. No new SerpAPI search is performed.
    2. classify_intent provides referenced_product_id.
    3. product_memory performs the exact deterministic lookup.
    4. Full referenced-product information is supplied to the LLM.
    5. Recent search history is also supplied for conversational context.
    """

    agent_logger.info(
        "Entering answer_followup_node."
    )

    # Follow-up answers are plain text.
    state["structured_recommendation"] = None

    try:
        # ---------------------------------------------------------------------
        # STEP 1: Read state
        # ---------------------------------------------------------------------

        user_query = state.get(
            "user_query",
            "",
        )

        last_shown_deals = state.get(
            "last_shown_deals",
            [],
        )

        search_history = state.get(
            "search_history",
            [],
        )

        product_memory = state.get(
            "product_memory",
            {},
        )

        referenced_id = (
            state.get("search_params") or {}
        ).get(
            "referenced_product_id"
        )

        # ---------------------------------------------------------------------
        # STEP 2: Resolve referenced product
        # ---------------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # We now use product_memory instead of last_shown_deals.
        #
        # Example:
        #
        # referenced_id = "s1p1"
        #
        # product_memory:
        #
        #   s1p1 -> HP Victus
        #   s2p1 -> MacBook
        #
        # Even if s1p1 is from an older search, it can still be found.
        # ---------------------------------------------------------------------

        referenced_product = (
            _resolve_referenced_product(
                product_memory,
                referenced_id,
            )
        )

        # ---------------------------------------------------------------------
        # STEP 3: Format latest products
        # ---------------------------------------------------------------------

        deals_text = (
            _format_latest_products_for_prompt(
                last_shown_deals
            )
        )

        # ---------------------------------------------------------------------
        # STEP 4: Format recent history
        # ---------------------------------------------------------------------

        earlier_history_text = (
            _format_history_for_prompt(
                search_history
            )
        )

        # ---------------------------------------------------------------------
        # STEP 5: Build referenced-product context
        # ---------------------------------------------------------------------

        if referenced_product:

            referenced_text = (
                _format_product_for_prompt(
                    referenced_product
                )
            )

            agent_logger.info(
                "Resolved follow-up product "
                "using product_memory: id=%s title=%r",
                referenced_id,
                referenced_product.get("title"),
            )

        elif referenced_id:

            # The classifier identified a product ID, but the ID is not
            # currently available in product_memory.
            #
            # IMPORTANT:
            # Do not guess the product from a title or position.
            # We explicitly tell the LLM that the exact product data
            # is unavailable.
            referenced_text = (
                f"Product ID: {referenced_id}\n"
                "Full product details for this product "
                "are not currently available in session memory. "
                "Do not invent them."
            )

            agent_logger.warning(
                "Referenced product ID %s was not found "
                "in product_memory. memory_size=%d",
                referenced_id,
                len(product_memory),
            )

        else:

            referenced_text = (
                "No single product was identified. "
                "Use the user's question together with "
                "the available recent product context."
            )

        # ---------------------------------------------------------------------
        # STEP 6: Build LLM prompt
        # ---------------------------------------------------------------------

        llm = get_groq_llm()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
The user is asking a follow-up question about products
already shown during this shopping session.

Answer directly and conversationally.

IMPORTANT RULES:

1. Do NOT search for new products.
2. Do NOT invent product details.
3. Use ONLY the product information supplied below.
4. If a referenced product is supplied with full details,
   answer using those details.
5. If a product ID is identified but its full details are
   unavailable, explicitly say that the available memory does
   not contain enough information instead of guessing.
6. The product ID is an internal identifier and does not need
   to be shown to the user unless useful.
7. Pricing risk is a warning, NOT proof that a product is fake.
8. Do not change prices, specifications, ratings, or scores.
9. If the question asks for a comparison, compare only products
   whose supplied information supports that comparison.
10. If the user asks for something like "show me cheaper options",
    explain based on the products already available in memory.
""".strip(),
                ),
                (
                    "human",
                    """
MOST RECENTLY SHOWN PRODUCTS:
{deals}

RECENT SEARCH HISTORY:
{earlier_history}

SPECIFIC REFERENCED PRODUCT:
{referenced}

USER'S FOLLOW-UP QUESTION:
{query}

Answer the follow-up question using only the supplied data.
""".strip(),
                ),
            ]
        )

        chain = prompt | llm

        # ---------------------------------------------------------------------
        # STEP 7: Stream answer
        # ---------------------------------------------------------------------

        agent_logger.info(
            "Streaming follow-up answer from memory; "
            "no new product search will be performed."
        )

        full_response = ""

        for chunk in chain.stream(
            {
                "deals": deals_text,
                "earlier_history": earlier_history_text,
                "referenced": referenced_text,
                "query": user_query,
            }
        ):

            content = getattr(
                chunk,
                "content",
                "",
            )

            if content:
                full_response += content

        # ---------------------------------------------------------------------
        # STEP 8: Store final answer
        # ---------------------------------------------------------------------

        state["final_recommendation"] = (
            full_response.strip()
        )

        # If the model somehow returned an empty response,
        # use a safe fallback rather than leaving the state blank.
        if not state["final_recommendation"]:

            if referenced_product:
                state["final_recommendation"] = (
                    "I found the referenced product, "
                    "but I couldn't generate a useful answer "
                    "from the available information. "
                    "Could you rephrase your question?"
                )
            else:
                state["final_recommendation"] = (
                    "I couldn't find enough information in "
                    "the current session memory to answer "
                    "that follow-up. Could you rephrase it?"
                )

        agent_logger.info(
            "Follow-up answered successfully. "
            "referenced_product_id=%s",
            referenced_id,
        )

    except Exception as e:

        agent_logger.error(
            f"answer_followup_node failed: {str(e)}",
            exc_info=True,
        )

        state.setdefault(
            "errors",
            [],
        ).append(
            f"Follow-up handling failed: {str(e)}"
        )

        state["final_recommendation"] = (
            "I had trouble answering that follow-up. "
            "Could you rephrase your question?"
        )

    return state
