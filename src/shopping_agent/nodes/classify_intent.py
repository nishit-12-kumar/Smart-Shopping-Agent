from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


# ============================================================================
# CONFIGURATION
# ============================================================================

# Maximum number of products from the long-term product memory that we expose
# to the LLM in one prompt.
#
# Recent products are already covered by search_history, so this is mainly
# useful for resolving references to older products.
MAX_LONG_TERM_PRODUCTS_FOR_PROMPT = 50


# ============================================================================
# STRUCTURED LLM OUTPUT
# ============================================================================


class IntentClassification(BaseModel):
    intent: str = Field(
        description=(
            "Either 'FOLLOW_UP' if the message is about a product already "
            "shown during this shopping session, or 'NEW' if it's a fresh "
            "shopping search request."
        )
    )

    referenced_product_id: Optional[str] = Field(
        default=None,
        description=(
            "The exact product id, such as 's1p1' or 's2p3', of the product "
            "the user is referring to. The id must come from the product "
            "history or long-term product memory shown in the prompt. "
            "Return null if intent is NEW, or if the FOLLOW_UP does not "
            "refer to one specific product."
        )
    )


# ============================================================================
# RECENT SEARCH HISTORY
# ============================================================================


def _format_history_for_prompt(
    search_history: List[Dict[str, Any]],
) -> str:
    """
    Format recent search history for the intent-classification LLM.

    This history is compact and contains:
        [product_id] title — price

    It is mainly used for references such as:
        "the first one"
        "option 2"
        "compare the Lenovo and ASUS"
    """

    if not search_history:
        return "None"

    blocks = []

    # Search history is stored oldest -> newest.
    # We preserve that ordering because the product IDs already identify
    # the exact search turn.
    for turn_number, turn in enumerate(
        search_history,
        start=1,
    ):

        query = turn.get("query", "")
        products = turn.get("products", [])

        lines = []

        for product in products:
            product_id = product.get("id")
            title = product.get("title")
            price = product.get("price")

            if price is None:
                price_text = "price unavailable"
            else:
                price_text = f"₹{price}"

            lines.append(
                f"  [{product_id}] "
                f"{title} — {price_text}"
            )

        if not lines:
            continue

        blocks.append(
            f'Search turn {turn_number}: "{query}"\n'
            + "\n".join(lines)
        )

    return (
        "\n\n".join(blocks)
        if blocks
        else "None"
    )


# ============================================================================
# LONG-TERM PRODUCT MEMORY
# ============================================================================


def _format_product_memory_for_prompt(
    product_memory: Dict[str, Dict[str, Any]],
    recent_history: List[Dict[str, Any]],
) -> str:
    """
    Format a compact index of products stored in long-term session memory.

    IMPORTANT:
    We do NOT send full product dictionaries here.

    We only expose:
        product id
        title
        price

    Full details are retrieved later by answer_followup.py using the
    selected product ID.

    Products already visible in recent search_history are skipped to
    avoid unnecessary duplication.
    """

    if not product_memory:
        return "None"

    # ---------------------------------------------------------------------
    # Collect IDs already visible in recent search history.
    # ---------------------------------------------------------------------

    recent_ids = set()

    for turn in recent_history:
        for product in turn.get("products", []):
            product_id = product.get("id")

            if product_id:
                recent_ids.add(str(product_id))

    # ---------------------------------------------------------------------
    # Build compact long-term memory entries.
    # ---------------------------------------------------------------------

    lines = []

    for product_id, product in product_memory.items():

        if not isinstance(product, dict):
            continue

        # Recent products are already in the recent-history section.
        if str(product_id) in recent_ids:
            continue

        title = product.get(
            "title",
            "Unknown product",
        )

        price = product.get("price")

        if price is None:
            price_text = "price unavailable"
        else:
            price_text = f"₹{price}"

        lines.append(
            f"  [{product_id}] "
            f"{title} — {price_text}"
        )

        if (
            len(lines)
            >= MAX_LONG_TERM_PRODUCTS_FOR_PROMPT
        ):
            break

    return (
        "\n".join(lines)
        if lines
        else "None"
    )


# ============================================================================
# INTENT CLASSIFICATION NODE
# ============================================================================


def classify_intent_node(
    state: ShoppingState,
) -> ShoppingState:
    """
    Determine whether the current user message is:

        NEW
        or
        FOLLOW_UP

    If it is a follow-up about one specific product, identify the exact
    product ID.

    The classifier uses two levels of memory:

    1. Recent search_history
       - useful for positional references such as "the first one"

    2. product_memory
       - useful for references to products from older searches
         such as "what about that HP Victus from earlier?"
    """

    agent_logger.info(
        "Entering classify_intent_node."
    )

    # -------------------------------------------------------------------------
    # Read memory and current message
    # -------------------------------------------------------------------------

    search_history = state.get(
        "search_history",
        [],
    )

    product_memory = state.get(
        "product_memory",
        {},
    )

    user_query = state.get(
        "user_query",
        "",
    )

    # -------------------------------------------------------------------------
    # No previous memory -> definitely a NEW search.
    # -------------------------------------------------------------------------

    if not search_history and not product_memory:
        state["intent"] = "NEW"

        state["search_params"] = (
            state.get("search_params") or {}
        )

        state["search_params"][
            "referenced_product_id"
        ] = None

        agent_logger.info(
            "No previous search or product memory. "
            "Classified as NEW."
        )

        return state

    try:

        # ---------------------------------------------------------------------
        # Format the two memory layers.
        # ---------------------------------------------------------------------

        recent_history_text = (
            _format_history_for_prompt(
                search_history
            )
        )

        long_term_memory_text = (
            _format_product_memory_for_prompt(
                product_memory,
                search_history,
            )
        )

        # ---------------------------------------------------------------------
        # Build classifier prompt.
        # ---------------------------------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are classifying a user's shopping message.

The user may either:

1. Start a NEW shopping search, or
2. Ask a FOLLOW_UP question about products already shown
   during this shopping session.

Your job is to determine the intent and, when appropriate,
identify the exact product ID being referenced.

IMPORTANT RULES:

1. Return "NEW" for a genuinely new shopping request.

2. Return "FOLLOW_UP" when the user is asking about products
   already shown during this session.

3. If the user refers to one specific product, return its EXACT
   product ID in referenced_product_id.

4. NEVER return the product title in referenced_product_id.
   Return only the ID shown in square brackets.

5. If the user says:
      "the first one"
      "option 1"
      "the second one"
   prefer the products from the MOST RECENT search when determining
   the positional reference.

6. If the user explicitly names a product, brand, or model from an
   older search, you may use the LONG-TERM PRODUCT MEMORY section
   to identify its exact product ID.

7. If the user asks for a general follow-up such as:
      "show me something cheaper"
      "give me more options"
      "compare these"
   return FOLLOW_UP but return null for referenced_product_id
   unless one specific product is clearly identified.

8. Never invent a product ID.
   The returned ID must appear in the supplied memory.

9. If you are uncertain about a product reference, return null
   rather than guessing.

10. A follow-up can refer to a product from several searches ago.
""".strip(),
                ),
                (
                    "human",
                    """
RECENT SEARCH HISTORY:
{recent_history}

LONG-TERM PRODUCT MEMORY:
{long_term_memory}

NEW USER MESSAGE:
{query}

Return the structured intent classification now.
""".strip(),
                ),
            ]
        )

        # ---------------------------------------------------------------------
        # Structured LLM
        # ---------------------------------------------------------------------

        llm = get_groq_llm()

        structured_llm = (
            llm.with_structured_output(
                IntentClassification
            )
        )

        chain = prompt | structured_llm

        result: IntentClassification = (
            chain.invoke(
                {
                    "recent_history": recent_history_text,
                    "long_term_memory": long_term_memory_text,
                    "query": user_query,
                }
            )
        )

        # ---------------------------------------------------------------------
        # Validate intent
        # ---------------------------------------------------------------------

        intent = (
            result.intent
            if result.intent in (
                "FOLLOW_UP",
                "NEW",
            )
            else "NEW"
        )

        referenced_product_id = (
            result.referenced_product_id
        )

        # ---------------------------------------------------------------------
        # Validate the returned product ID.
        #
        # We do not blindly trust the LLM.
        # The ID must actually exist in our session memory.
        # ---------------------------------------------------------------------

        if referenced_product_id:

            referenced_product_id = str(
                referenced_product_id
            ).strip()

            product_exists = (
                referenced_product_id
                in product_memory
            )

            if not product_exists:

                # It might still be present in search_history
                # even if product_memory is somehow incomplete.
                history_product_ids = set()

                for turn in search_history:
                    for product in turn.get(
                        "products",
                        [],
                    ):
                        product_id = product.get(
                            "id"
                        )

                        if product_id:
                            history_product_ids.add(
                                str(product_id)
                            )

                product_exists = (
                    referenced_product_id
                    in history_product_ids
                )

            if not product_exists:

                agent_logger.warning(
                    "LLM returned unknown product ID: %s. "
                    "Clearing reference instead of guessing.",
                    referenced_product_id,
                )

                referenced_product_id = None

        # If the classifier says NEW, there should be no product reference.
        if intent == "NEW":
            referenced_product_id = None

        # ---------------------------------------------------------------------
        # Save classification result into state.
        # ---------------------------------------------------------------------

        state["intent"] = intent

        state["search_params"] = (
            state.get("search_params")
            or {}
        )

        state["search_params"][
            "referenced_product_id"
        ] = referenced_product_id

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------

        agent_logger.info(
            "Intent classification result: "
            "intent=%s referenced_product_id=%s "
            "recent_searches=%d memory_products=%d",
            intent,
            referenced_product_id,
            len(search_history),
            len(product_memory),
        )

    except Exception as e:

        agent_logger.error(
            f"Intent classification failed: {str(e)}",
            exc_info=True,
        )

        # ---------------------------------------------------------------------
        # Fail safe
        # ---------------------------------------------------------------------
        #
        # If classification fails, treating the request as NEW is safer than
        # accidentally answering the wrong historical product.
        # ---------------------------------------------------------------------

        state["intent"] = "NEW"

        state["search_params"] = (
            state.get("search_params")
            or {}
        )

        state["search_params"][
            "referenced_product_id"
        ] = None

    return state
