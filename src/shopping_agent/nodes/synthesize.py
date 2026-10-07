"""
Recommendation synthesis node for the Smart Shopping Negotiator.

This module is deliberately split into two responsibilities:

1. Deterministic scoring/assembly in Python.
2. Concise qualitative explanations from the Groq LLM.

The LLM never controls product identity, price, links, IDs,
recommendation scores, or pricing-risk decisions.

The recommendation match score uses a hybrid approach:

    70% deterministic requirement matching
    30% LLM semantic/relevance validation

When no structured requirement can be extracted from the user's
query, the hybrid match falls back to the LLM relevance score so that
the score is not artificially reduced.

All user-facing LLM text is normalized to plain text before storage.
"""

from __future__ import annotations

import html
import math
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.nodes.price_validity import build_peer_stats_by_index
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_ALTERNATIVES = 3

# Number of recent search turns retained in compact search history.
# This is used for follow-up / intent context.
MAX_HISTORY_TURNS = 8

MAX_FOLLOW_UP_SUGGESTIONS = 2


# ============================================================================
# RECOMMENDATION WEIGHTS
# ============================================================================

# Hybrid match weight inside the FINAL recommendation score.
#
# This remains 50%, but that 50% is now itself composed of:
#
#     70% deterministic requirement matching
#     30% LLM semantic relevance
#
# Therefore the effective contribution to the final score is:
#
#     deterministic requirement match = 0.50 * 0.70 = 35%
#     semantic LLM relevance           = 0.50 * 0.30 = 15%
#
# Other final components:
#
#     rating                        = 15%
#     reviews                       = 10%
#     price value                   = 10%
#     pricing trust                 = 15%
#
WEIGHT_MATCH = 0.50
WEIGHT_RATING = 0.15
WEIGHT_REVIEWS = 0.10
WEIGHT_PRICE_VALUE = 0.10
WEIGHT_PRICING_TRUST = 0.15


# Internal composition of the hybrid match score.
HYBRID_DETERMINISTIC_WEIGHT = 0.70
HYBRID_LLM_WEIGHT = 0.30


# HTML/XML-like tags occasionally appear in otherwise valid LLM strings.
_HTML_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:text|plain|html|markdown)?\s*|\s*```\s*$",
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")

_NUMBER_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?"
)


# ============================================================================
# PLAIN-TEXT / DATA NORMALIZATION HELPERS
# ============================================================================


def _clean_plain_text(
    value: Any,
    default: str = "",
) -> str:
    """Return safe, compact plain text from an arbitrary value."""
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    # Remove Markdown code fences.
    text = _CODE_FENCE_RE.sub(
        "",
        text,
    ).strip()

    # Decode HTML entities.
    for _ in range(2):
        decoded = html.unescape(text)

        if decoded == text:
            break

        text = decoded

    # Remove HTML/XML tags.
    text = _HTML_TAG_RE.sub(
        "",
        text,
    )

    text = (
        text
        .replace("<br/>", " ")
        .replace("<br>", " ")
        .replace("<br />", " ")
    )

    # Remove remaining code-fence artifacts and normalize whitespace.
    text = text.replace(
        "```",
        " ",
    )

    text = _WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()

    return text or default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert common numeric representations to float without raising."""
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        try:
            number = float(value)
            return (
                number
                if math.isfinite(number)
                else default
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return default

    text = str(value).strip().replace(
        ",",
        "",
    )

    match = _NUMBER_RE.search(text)

    if not match:
        return default

    try:
        number = float(
            match.group(0)
        )

        return (
            number
            if math.isfinite(number)
            else default
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Convert review-count-like values such as '1,234 reviews' to int."""
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return (
            int(value)
            if math.isfinite(value)
            else default
        )

    text = str(value).strip().replace(
        ",",
        "",
    )

    match = re.search(
        r"\d+",
        text,
    )

    if not match:
        return default

    try:
        return int(
            match.group(0)
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default


def _clamp_score(
    value: Any,
    default: float = 0.0,
) -> float:
    """Normalize a score to the inclusive 0-100 range."""
    return max(
        0.0,
        min(
            100.0,
            _safe_float(
                value,
                default,
            ),
        ),
    )


def _clean_list(
    values: Any,
    max_items: Optional[int] = None,
) -> List[str]:
    """Normalize a list of strings and remove empty/duplicate entries."""
    if not isinstance(
        values,
        list,
    ):
        return []

    cleaned: List[str] = []
    seen = set()

    for value in values:

        text = _clean_plain_text(
            value
        )

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(text)

        if (
            max_items is not None
            and len(cleaned) >= max_items
        ):
            break

    return cleaned


def _product_title(
    product: Dict[str, Any],
) -> str:
    return _clean_plain_text(
        product.get("title"),
        "Unknown product",
    )


def _product_price_text(
    product: Dict[str, Any],
) -> str:

    price = product.get(
        "price"
    )

    numeric = _safe_float(
        price,
        -1.0,
    )

    if numeric > 0:
        return f"₹{numeric:,.0f}"

    return "price unavailable"


# ============================================================================
# STRUCTURED LLM OUTPUT
# ============================================================================


class TopPickAnalysis(BaseModel):
    why_it_wins: str = Field(
        description=(
            "One short sentence explaining why the supplied top pick best "
            "matches the user's request. Plain text only."
        )
    )

    specs_matched: List[str] = Field(
        description=(
            "2-4 short plain-text specification tags confirmed by the supplied "
            "product data. Never invent a specification."
        )
    )

    specs_warning: Optional[str] = Field(
        default=None,
        description=(
            "One short plain-text warning for an unconfirmed or mismatched "
            "important specification, otherwise null."
        ),
    )


class SynthesisOutput(BaseModel):
    top_pick_analysis: TopPickAnalysis

    alternative_trade_offs: List[str] = Field(
        description=(
            "Exactly one short plain-text trade-off sentence for each supplied "
            "alternative, in exactly the same order. Never return HTML/XML "
            "tags, Markdown, code fences, or UI markup."
        )
    )

    filtered_out_note: Optional[str] = Field(
        default=None,
        description=(
            "One short plain-text sentence about the number of filtered-out "
            "options. Do not invent filtering reasons."
        ),
    )

    bottom_line: str = Field(
        description=(
            "One short plain-text concluding recommendation sentence."
        )
    )

    follow_up_suggestions: List[str] = Field(
        description=(
            "Exactly two short plain-text follow-up questions the user could "
            "ask next. No HTML/XML/Markdown."
        )
    )


# ============================================================================
# BASIC RECOMMENDATION COMPONENTS
# ============================================================================


def _rating_score(
    product: Dict[str, Any],
) -> float:
    """
    Convert a 0-5 rating into a 0-100 score.
    """
    rating = _safe_float(
        product.get("rating")
    )

    if rating <= 0:
        return 0.0

    return max(
        0.0,
        min(
            100.0,
            rating / 5.0 * 100.0,
        ),
    )


def _review_score(
    product: Dict[str, Any],
) -> float:
    """
    Convert review count to a logarithmically scaled 0-100 score.

    10,000+ reviews approximately saturate at 100.
    """
    reviews = _safe_int(
        product.get("reviews")
    )

    if reviews <= 0:
        return 0.0

    return min(
        100.0,
        math.log10(
            reviews + 1
        ) / 4.0 * 100.0,
    )


def _price_value_score(
    product: Dict[str, Any],
    peer_stats: Optional[Dict[str, Any]],
) -> float:
    """
    Score price relative to the trusted peer-group median.

    Intended mapping:
        0.5x median -> 100
        1.0x median -> 50
        1.5x median -> 0

    Values outside that range are clamped.
    """
    current_price = _safe_float(
        product.get("price")
    )

    if (
        current_price <= 0
        or not peer_stats
    ):
        return 50.0

    median_price = _safe_float(
        peer_stats.get("median")
    )

    if median_price <= 0:
        return 50.0

    ratio = (
        current_price
        / median_price
    )

    score = 150.0 - (
        ratio * 100.0
    )

    return max(
        0.0,
        min(
            100.0,
            score,
        ),
    )


def _pricing_trust_score(
    product: Dict[str, Any],
) -> float:
    """
    Convert deterministic pricing risk (0-100)
    to pricing trust (100-0).
    """
    risk = product.get(
        "pricing_risk_score"
    )

    if risk is None:
        return 50.0

    return (
        100.0
        - _clamp_score(risk)
    )


# ============================================================================
# HYBRID MATCH SCORE
# ============================================================================


def calculate_hybrid_match_score(
    product: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the hybrid requirement/relevance match score.

    Hybrid formula:

        Hybrid Match
        =
        70% Deterministic Requirement Match
        +
        30% LLM Semantic Relevance

    The deterministic score is created in validate_deals.py.

    The LLM score is stored as confidence_score.

    IMPORTANT FALLBACK:
    If no deterministic requirement score exists, we use the LLM score
    directly rather than multiplying it by 0.30. This prevents a vague
    query such as "best laptop" from being unfairly penalized just because
    no structured requirement could be extracted.
    """

    llm_score = _clamp_score(
        product.get(
            "confidence_score"
        )
    )

    deterministic_value = product.get(
        "deterministic_match_score"
    )

    # No structured requirements were available.
    #
    # Use the LLM semantic relevance score as the complete match score.
    if deterministic_value is None:

        return (
            round(
                llm_score,
                2,
            ),
            {
                "deterministic": None,
                "llm": round(
                    llm_score,
                    2,
                ),
                "hybrid": round(
                    llm_score,
                    2,
                ),
            },
        )

    deterministic_score = _clamp_score(
        deterministic_value
    )

    hybrid_score = (
        HYBRID_DETERMINISTIC_WEIGHT
        * deterministic_score
        + HYBRID_LLM_WEIGHT
        * llm_score
    )

    return (
        round(
            max(
                0.0,
                min(
                    100.0,
                    hybrid_score,
                ),
            ),
            2,
        ),
        {
            "deterministic": round(
                deterministic_score,
                2,
            ),
            "llm": round(
                llm_score,
                2,
            ),
            "hybrid": round(
                hybrid_score,
                2,
            ),
        },
    )


# ============================================================================
# FINAL RECOMMENDATION SCORE
# ============================================================================


def calculate_recommendation_score(
    product: Dict[str, Any],
    peer_stats: Optional[Dict[str, Any]],
) -> float:
    """
    Calculate the final deterministic recommendation score.

    Final formula:

        50% Hybrid Match
        15% Rating
        10% Reviews
        10% Price Value
        15% Pricing Trust
    """

    hybrid_match_score, _ = (
        calculate_hybrid_match_score(
            product
        )
    )

    rating_score = _rating_score(
        product
    )

    review_score = _review_score(
        product
    )

    price_value_score = (
        _price_value_score(
            product,
            peer_stats,
        )
    )

    trust_score = (
        _pricing_trust_score(
            product
        )
    )

    total = (
        WEIGHT_MATCH
        * hybrid_match_score

        + WEIGHT_RATING
        * rating_score

        + WEIGHT_REVIEWS
        * review_score

        + WEIGHT_PRICE_VALUE
        * price_value_score

        + WEIGHT_PRICING_TRUST
        * trust_score
    )

    return round(
        max(
            0.0,
            min(
                100.0,
                total,
            ),
        ),
        2,
    )


# ============================================================================
# SCORE BREAKDOWN
# ============================================================================


def get_score_breakdown(
    product_id: str,
    all_deals: list,
) -> Optional[dict]:
    """
    Return the same scoring components used to calculate the final score.

    The returned structure now also exposes the hybrid-match breakdown:

        deterministic requirement match
        LLM semantic relevance
        final hybrid match
    """
    if not isinstance(
        all_deals,
        list,
    ):
        return None

    product_index = next(
        (
            i
            for i, product in enumerate(
                all_deals
            )
            if (
                isinstance(
                    product,
                    dict,
                )
                and product.get("id")
                == product_id
            )
        ),
        None,
    )

    if product_index is None:
        return None

    product = all_deals[
        product_index
    ]

    try:
        peer_stats_by_index = (
            build_peer_stats_by_index(
                all_deals
            )
        )

        peer_stats = (
            peer_stats_by_index.get(
                product_index
            )
        )

    except Exception as exc:

        agent_logger.warning(
            "Unable to rebuild peer stats for score breakdown: %s",
            exc,
        )

        peer_stats = None

    hybrid_match_score, hybrid_details = (
        calculate_hybrid_match_score(
            product
        )
    )

    components = [
        (
            "match",
            "Match (Hybrid)",
            hybrid_match_score,
            WEIGHT_MATCH,
        ),
        (
            "rating",
            "Rating",
            _rating_score(product),
            WEIGHT_RATING,
        ),
        (
            "reviews",
            "Reviews",
            _review_score(product),
            WEIGHT_REVIEWS,
        ),
        (
            "price_value",
            "Price Value",
            _price_value_score(
                product,
                peer_stats,
            ),
            WEIGHT_PRICE_VALUE,
        ),
        (
            "pricing_trust",
            "Pricing Trust",
            _pricing_trust_score(
                product
            ),
            WEIGHT_PRICING_TRUST,
        ),
    ]

    breakdown: Dict[str, Any] = {}

    formula_rows: List[
        Dict[str, Any]
    ] = []

    computed_total = 0.0

    for (
        key,
        label,
        score,
        weight,
    ) in components:

        contribution = (
            score * weight
        )

        computed_total += contribution

        breakdown[key] = round(
            score
        )

        formula_rows.append(
            {
                "key": key,
                "label": label,
                "score": round(score),
                "weight_pct": round(
                    weight * 100
                ),
                "contribution": round(
                    contribution,
                    2,
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Additional hybrid-match detail.
    #
    # These are NOT additional final-score components.
    # They explain how the single 50% "Match" component was produced.
    # ------------------------------------------------------------------------

    breakdown[
        "deterministic_match"
    ] = (
        None
        if hybrid_details[
            "deterministic"
        ] is None
        else round(
            hybrid_details[
                "deterministic"
            ]
        )
    )

    breakdown[
        "semantic_match"
    ] = round(
        hybrid_details["llm"]
    )

    breakdown[
        "hybrid_match"
    ] = round(
        hybrid_match_score
    )

    breakdown[
        "hybrid_match_weights"
    ] = {
        "deterministic_pct": round(
            HYBRID_DETERMINISTIC_WEIGHT
            * 100
        ),
        "llm_pct": round(
            HYBRID_LLM_WEIGHT
            * 100
        ),
    }

    breakdown[
        "overall"
    ] = product.get(
        "recommendation_score"
    )

    breakdown[
        "computed_total"
    ] = round(
        max(
            0.0,
            min(
                100.0,
                computed_total,
            ),
        ),
        2,
    )

    breakdown[
        "formula_rows"
    ] = formula_rows

    return breakdown


# ============================================================================
# LLM CHAIN
# ============================================================================


def _build_synthesis_chain():
    """
    Build the structured-output Groq chain used for qualitative text.
    """
    llm = get_groq_llm()

    structured_llm = (
        llm.with_structured_output(
            SynthesisOutput
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the qualitative explanation layer of a shopping recommendation system.

Python has ALREADY:
1. Validated the products.
2. Calculated deterministic requirement matching.
3. Calculated the hybrid recommendation score.
4. Selected the top pick and alternatives.

Your job is ONLY to write short explanations using the supplied facts.

NON-NEGOTIABLE RULES:
- Do not select a different top pick.
- Do not change, calculate, or invent scores.
- Do not invent prices, product names, ratings, review counts, or specifications.
- Mention a specification only if it is explicitly present in the supplied data.
- Treat pricing risk as a warning, never as proof that a listing is fraudulent.
- Do not invent reasons for filtered-out products.
- Keep all generated text concise and natural.
- Return exactly one trade-off for each alternative, in the same order.
- Return exactly two follow-up suggestions.

PLAIN-TEXT OUTPUT REQUIREMENT:
- Every string field MUST contain plain human-readable text only.
- NEVER output HTML, XML, JSX, CSS, Markdown, Markdown code fences, or UI markup.
- NEVER output tags such as <p>, </p>, <div>, <span>, <strong>, <br>, etc.
- Do not prefix or suffix a sentence with markup.
""".strip(),
            ),
            (
                "human",
                """
USER REQUEST:
{query}

TOP PICK:
{top_pick}

ALTERNATIVES:
{alternatives}

FILTERED OUT COUNT:
{filtered_count}

PRICING FLAGS:
{red_flag_data}

Generate the structured explanation now.
Keep every string field as plain text.
""".strip(),
            ),
        ]
    )

    return prompt | structured_llm


# ============================================================================
# LLM INPUT FORMATTING
# ============================================================================


def _format_product_line(
    product: Dict[str, Any],
) -> str:
    """
    Format only verified product data for the LLM.
    """
    title = _product_title(
        product
    )

    price = _product_price_text(
        product
    )

    score = _clamp_score(
        product.get(
            "recommendation_score"
        )
    )

    confidence = _clamp_score(
        product.get(
            "confidence_score"
        )
    )

    hybrid_match = _clamp_score(
        product.get(
            "hybrid_match_score",
            confidence,
        )
    )

    deterministic_match = product.get(
        "deterministic_match_score"
    )

    parts = [
        f"Title: {title}",
        f"Price: {price}",
        f"Recommendation Score: {score:.2f}/100",
        f"Hybrid Match: {hybrid_match:.2f}/100",
        f"LLM Relevance: {confidence:.2f}/100",
    ]

    if deterministic_match is not None:
        parts.append(
            "Deterministic Requirement Match: "
            f"{_clamp_score(deterministic_match):.2f}/100"
        )

    rating = product.get(
        "rating"
    )

    if (
        rating is not None
        and _safe_float(rating) > 0
    ):
        review_count = _safe_int(
            product.get(
                "reviews"
            )
        )

        parts.append(
            f"Rating: {_safe_float(rating):.1f}/5 "
            f"({review_count} reviews)"
        )

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

    specs: List[str] = []

    for field in spec_fields:

        value = _clean_plain_text(
            product.get(field)
        )

        if value:
            specs.append(
                f"{field}: {value}"
            )

    if specs:
        parts.append(
            "Specifications: "
            + ", ".join(specs)
        )

    if product.get(
        "pricing_risk_score"
    ) is not None:

        risk = _clamp_score(
            product.get(
                "pricing_risk_score"
            )
        )

        level = _clean_plain_text(
            product.get(
                "pricing_risk_level"
            ),
            "UNKNOWN",
        )

        parts.append(
            f"Pricing Risk: {risk:.0f}/100 ({level})"
        )

    return " | ".join(
        parts
    )


# ============================================================================
# PRODUCT / HISTORY HELPERS
# ============================================================================


def _assign_product_ids(
    deals: list,
    turn_number: int,
) -> None:
    """
    Assign stable per-session IDs without overwriting existing IDs.
    """
    for index, product in enumerate(
        deals,
        start=1,
    ):

        if not isinstance(
            product,
            dict,
        ):
            continue

        product.setdefault(
            "id",
            f"s{turn_number}p{index}",
        )


def _build_product_summary(
    product: Dict[str, Any],
) -> dict:
    """
    Create the compact product representation stored in search history.

    Hybrid scoring fields are also stored because they are useful for
    historical context and debugging.
    """
    return {
        "id": product.get("id"),
        "title": product.get("title"),
        "price": product.get("price"),
        "rating": product.get("rating"),
        "reviews": product.get("reviews"),
        "confidence_score": product.get(
            "confidence_score"
        ),
        "deterministic_match_score": product.get(
            "deterministic_match_score"
        ),
        "hybrid_match_score": product.get(
            "hybrid_match_score"
        ),
        "recommendation_score": product.get(
            "recommendation_score"
        ),
        "pricing_risk_score": product.get(
            "pricing_risk_score"
        ),
        "pricing_risk_level": product.get(
            "pricing_risk_level"
        ),
        "is_suspicious_pricing": product.get(
            "is_suspicious_pricing",
            False,
        ),
    }


def _fallback_trade_off(
    alt: Dict[str, Any],
    top_pick: Dict[str, Any],
) -> str:
    """
    Generate a deterministic trade-off when the LLM omits one.
    """
    alt_price = _safe_float(
        alt.get("price")
    )

    top_price = _safe_float(
        top_pick.get("price")
    )

    alt_score = _clamp_score(
        alt.get(
            "recommendation_score"
        )
    )

    top_score = _clamp_score(
        top_pick.get(
            "recommendation_score"
        )
    )

    if (
        alt_price > 0
        and top_price > 0
    ):

        if (
            alt_price < top_price
            and alt_score < top_score
        ):
            return (
                "Lower price, but a lower overall "
                "recommendation score."
            )

        if (
            alt_price > top_price
            and alt_score < top_score
        ):
            return (
                "Higher price with a lower overall "
                "recommendation score."
            )

        if alt_price < top_price:
            return (
                "Lower price, with a different "
                "overall trade-off."
            )

        if alt_price > top_price:
            return (
                "Higher price, with a different "
                "overall trade-off."
            )

    if alt_score < top_score:
        return (
            "A lower overall recommendation score "
            "than the top pick."
        )

    if alt_score > top_score:
        return (
            "A higher overall recommendation score, "
            "but it was not selected as the top pick."
        )

    return (
        "A different option with a similar "
        "overall recommendation score."
    )


def _fallback_synthesis(
    top_pick: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
    filtered_count: int,
) -> SynthesisOutput:
    """
    Build valid structured output without an LLM if the API fails.
    """
    matched_specs: List[str] = []

    for field, label in (
        ("processor", "processor"),
        ("ram", "RAM"),
        ("storage", "storage"),
        ("gpu", "GPU"),
        ("display", "display"),
        ("network", "network"),
    ):

        value = _clean_plain_text(
            top_pick.get(field)
        )

        if value:
            matched_specs.append(
                f"{label}: {value}"
            )

        if len(matched_specs) >= 4:
            break

    if not matched_specs:
        matched_specs = [
            "Matches the supplied product requirements"
        ]

    why = (
        f"{_product_title(top_pick)} has the highest "
        f"deterministic recommendation score among "
        f"the validated results."
    )

    filtered_note = None

    if filtered_count > 0:
        filtered_note = (
            f"{filtered_count} other options were "
            f"filtered out of this result set."
        )

    followups = [
        "Show me a cheaper option",
        "Compare the top pick with the alternatives",
    ]

    return SynthesisOutput(
        top_pick_analysis=TopPickAnalysis(
            why_it_wins=why,
            specs_matched=matched_specs,
            specs_warning=None,
        ),
        alternative_trade_offs=[
            _fallback_trade_off(
                alt,
                top_pick,
            )
            for alt in alternatives
        ],
        filtered_out_note=filtered_note,
        bottom_line=(
            f"The top pick is "
            f"{_product_title(top_pick)} at "
            f"{_product_price_text(top_pick)}."
        ),
        follow_up_suggestions=followups,
    )


# ============================================================================
# MAIN SYNTHESIS NODE
# ============================================================================


def synthesize_node(
    state: ShoppingState,
) -> ShoppingState:
    """
    Score validated deals, generate explanations,
    and update session state.

    Memory responsibilities:
    1. Keep full latest products in last_shown_deals.
    2. Keep compact recent summaries in search_history.
    3. Keep full session-level product records in product_memory.
    """
    agent_logger.info(
        "Entering synthesize_node."
    )

    try:
        user_query = _clean_plain_text(
            state.get(
                "user_query"
            )
        )

        raw_deals = (
            state.get(
                "validated_deals"
            )
            or []
        )

        # Keep only dictionary-like product records.
        deals = [
            deal
            for deal in raw_deals
            if isinstance(
                deal,
                dict,
            )
        ]

        if not deals:

            serpapi_message = _clean_plain_text(
                state.get(
                    "serpapi_error_message"
                )
            )

            state[
                "structured_recommendation"
            ] = None

            state[
                "final_recommendation"
            ] = (
                serpapi_message
                or (
                    "I couldn't find any products that "
                    "meet your criteria with high confidence. "
                    "Could we try adjusting your budget "
                    "or preferences?"
                )
            )

            agent_logger.warning(
                "No validated deals available for synthesis."
            )

            return state

        # ---------------------------------------------------------------------
        # STEP 1: Create unique search turn number.
        # ---------------------------------------------------------------------

        turn_number = (
            _safe_int(
                state.get(
                    "search_turn_count"
                ),
                0,
            )
            + 1
        )

        state[
            "search_turn_count"
        ] = turn_number

        _assign_product_ids(
            deals,
            turn_number,
        )

        # ---------------------------------------------------------------------
        # STEP 2: Calculate peer statistics.
        # ---------------------------------------------------------------------

        try:
            peer_stats_by_index = (
                build_peer_stats_by_index(
                    deals
                )
            )

        except Exception as exc:

            agent_logger.warning(
                "Peer-stat calculation failed; "
                "using neutral price-value scores: %s",
                exc,
            )

            peer_stats_by_index = {}

        # ---------------------------------------------------------------------
        # STEP 3: Calculate HYBRID MATCH score.
        # ---------------------------------------------------------------------
        #
        # The validated deals should already contain:
        #
        #     confidence_score
        #     deterministic_match_score
        #
        # from validate_deals.py.
        #
        # We combine those into:
        #
        #     hybrid_match_score
        #
        # before calculating the final recommendation score.
        # ---------------------------------------------------------------------

        for index, product in enumerate(
            deals
        ):

            hybrid_match_score, hybrid_details = (
                calculate_hybrid_match_score(
                    product
                )
            )

            product[
                "hybrid_match_score"
            ] = hybrid_match_score

            product[
                "hybrid_match_details"
            ] = hybrid_details

            # -------------------------------------------------------------
            # IMPORTANT:
            #
            # We use hybrid_match_score as the 50% match component.
            # -------------------------------------------------------------

            product[
                "recommendation_score"
            ] = calculate_recommendation_score(
                product,
                peer_stats_by_index.get(
                    index
                ),
            )

        # ---------------------------------------------------------------------
        # STEP 4: Sort by deterministic recommendation score.
        # ---------------------------------------------------------------------

        sorted_deals = sorted(
            deals,
            key=lambda product: (
                _clamp_score(
                    product.get(
                        "recommendation_score"
                    )
                ),
                _safe_float(
                    product.get(
                        "confidence_score"
                    )
                ),
            ),
            reverse=True,
        )

        top_pick = sorted_deals[0]

        alternatives = sorted_deals[
            1 : 1 + MAX_ALTERNATIVES
        ]

        filtered_count = max(
            0,
            len(sorted_deals)
            - 1
            - len(alternatives),
        )

        # ---------------------------------------------------------------------
        # STEP 5: Store full products in long-term product memory.
        # ---------------------------------------------------------------------

        product_memory = state.setdefault(
            "product_memory",
            {},
        )

        for product in sorted_deals:

            if not isinstance(
                product,
                dict,
            ):
                continue

            product_id = product.get(
                "id"
            )

            if not product_id:
                continue

            # Deep-copy so later modifications to the current turn cannot
            # accidentally alter historical product memory.
            product_memory[
                product_id
            ] = deepcopy(
                product
            )

        state[
            "product_memory"
        ] = product_memory

        agent_logger.info(
            "Stored %d products in product_memory. "
            "Current memory size=%d.",
            len(sorted_deals),
            len(product_memory),
        )

        # ---------------------------------------------------------------------
        # STEP 6: Pricing-risk context.
        # ---------------------------------------------------------------------

        red_flag_products = [
            product
            for product in sorted_deals
            if bool(
                product.get(
                    "is_suspicious_pricing",
                    False,
                )
            )
        ]

        red_flag_lines: List[str] = []

        for product in red_flag_products:

            reasons = (
                product.get(
                    "pricing_risk_reasons"
                )
                or []
            )

            clean_reasons = _clean_list(
                reasons
            )

            reason_text = (
                "; ".join(
                    clean_reasons
                )
                or "pricing signals require verification"
            )

            red_flag_lines.append(
                f"- {_product_title(product)}: "
                f"Risk "
                f"{_clamp_score(product.get('pricing_risk_score')):.0f}/100 "
                f"({_clean_plain_text(product.get('pricing_risk_level'), 'UNKNOWN')}) "
                f"| {reason_text}"
            )

        red_flag_data = (
            "\n".join(
                red_flag_lines
            )
            or "None"
        )

        # ---------------------------------------------------------------------
        # STEP 7: Prepare LLM synthesis input.
        # ---------------------------------------------------------------------

        top_pick_text = _format_product_line(
            top_pick
        )

        alternatives_text = "\n".join(
            f"Alternative {index + 1}: "
            f"{_format_product_line(product)}"
            for index, product in enumerate(
                alternatives
            )
        ) or "None"

        # ---------------------------------------------------------------------
        # STEP 8: LLM qualitative explanation.
        # ---------------------------------------------------------------------

        try:

            chain = _build_synthesis_chain()

            result = chain.invoke(
                {
                    "query": user_query,
                    "top_pick": top_pick_text,
                    "alternatives": alternatives_text,
                    "filtered_count": filtered_count,
                    "red_flag_data": red_flag_data,
                }
            )

            if not isinstance(
                result,
                SynthesisOutput,
            ):
                raise TypeError(
                    "Unexpected synthesis output type: "
                    f"{type(result).__name__}"
                )

        except Exception as exc:

            agent_logger.error(
                "LLM synthesis failed; "
                "using deterministic fallback: %s",
                exc,
                exc_info=True,
            )

            result = _fallback_synthesis(
                top_pick,
                alternatives,
                filtered_count,
            )

            state.setdefault(
                "errors",
                [],
            ).append(
                "LLM synthesis failed; "
                "deterministic recommendation text was used."
            )

        # ---------------------------------------------------------------------
        # STEP 9: Normalize LLM output.
        # ---------------------------------------------------------------------

        why_it_wins = _clean_plain_text(
            result.top_pick_analysis.why_it_wins,
            (
                f"{_product_title(top_pick)} has the "
                f"highest recommendation score among "
                f"the validated results."
            ),
        )

        specs_matched = _clean_list(
            result.top_pick_analysis.specs_matched,
            max_items=4,
        )

        if not specs_matched:
            specs_matched = [
                "Matches the supplied product requirements"
            ]

        specs_warning = (
            _clean_plain_text(
                result.top_pick_analysis.specs_warning
            )
            or None
        )

        filtered_out_note = (
            _clean_plain_text(
                result.filtered_out_note
            )
            or None
        )

        bottom_line = _clean_plain_text(
            result.bottom_line,
            (
                f"The top pick is "
                f"{_product_title(top_pick)} at "
                f"{_product_price_text(top_pick)}."
            ),
        )

        raw_trade_offs = (
            result.alternative_trade_offs
        )

        trade_offs = (
            [
                _clean_plain_text(
                    value
                )
                for value in raw_trade_offs
            ]
            if isinstance(
                raw_trade_offs,
                list,
            )
            else []
        )

        normalized_trade_offs: List[str] = []

        for index, alternative in enumerate(
            alternatives
        ):

            candidate = (
                trade_offs[index]
                if index < len(trade_offs)
                else ""
            )

            normalized_trade_offs.append(
                candidate
                or _fallback_trade_off(
                    alternative,
                    top_pick,
                )
            )

        followups = _clean_list(
            result.follow_up_suggestions,
            max_items=MAX_FOLLOW_UP_SUGGESTIONS,
        )

        fallback_followups = [
            "Show me a cheaper option",
            "Compare the top pick with the alternatives",
        ]

        for suggestion in fallback_followups:

            if (
                len(followups)
                >= MAX_FOLLOW_UP_SUGGESTIONS
            ):
                break

            if suggestion.casefold() not in {
                item.casefold()
                for item in followups
            }:

                followups.append(
                    suggestion
                )

        followups = followups[
            :MAX_FOLLOW_UP_SUGGESTIONS
        ]

        # ---------------------------------------------------------------------
        # STEP 10: Structured recommendation.
        # ---------------------------------------------------------------------

        structured: Dict[str, Any] = {
            "top_pick": {
                "id": top_pick.get("id"),
                "title": top_pick.get("title"),
                "price": top_pick.get("price"),
                "source": top_pick.get("source"),
                "link": top_pick.get("link"),
                "thumbnail": top_pick.get("thumbnail"),
                "confidence_score": top_pick.get(
                    "confidence_score"
                ),
                "deterministic_match_score": top_pick.get(
                    "deterministic_match_score"
                ),
                "hybrid_match_score": top_pick.get(
                    "hybrid_match_score"
                ),
                "recommendation_score": top_pick.get(
                    "recommendation_score"
                ),
                "rating": top_pick.get(
                    "rating"
                ),
                "reviews": top_pick.get(
                    "reviews"
                ),
                "pricing_risk_score": top_pick.get(
                    "pricing_risk_score"
                ),
                "pricing_risk_level": top_pick.get(
                    "pricing_risk_level"
                ),
                "is_suspicious_pricing": top_pick.get(
                    "is_suspicious_pricing",
                    False,
                ),
                "why_it_wins": why_it_wins,
                "specs_matched": specs_matched,
                "specs_warning": specs_warning,
            },

            "alternatives": [],

            "filtered_out_note": filtered_out_note,

            "red_flags": [
                {
                    "id": product.get("id"),
                    "title": product.get("title"),
                    "risk_score": product.get(
                        "pricing_risk_score"
                    ),
                    "risk_level": product.get(
                        "pricing_risk_level"
                    ),
                    "reasons": product.get(
                        "pricing_risk_reasons",
                        [],
                    ),
                }
                for product in red_flag_products
            ],

            "bottom_line": bottom_line,

            "follow_up_suggestions": followups,
        }

        for alternative, trade_off in zip(
            alternatives,
            normalized_trade_offs,
        ):

            structured[
                "alternatives"
            ].append(
                {
                    "id": alternative.get(
                        "id"
                    ),
                    "title": alternative.get(
                        "title"
                    ),
                    "price": alternative.get(
                        "price"
                    ),
                    "thumbnail": alternative.get(
                        "thumbnail"
                    ),
                    "link": alternative.get(
                        "link"
                    ),
                    "confidence_score": alternative.get(
                        "confidence_score"
                    ),
                    "deterministic_match_score": alternative.get(
                        "deterministic_match_score"
                    ),
                    "hybrid_match_score": alternative.get(
                        "hybrid_match_score"
                    ),
                    "recommendation_score": alternative.get(
                        "recommendation_score"
                    ),
                    "rating": alternative.get(
                        "rating"
                    ),
                    "reviews": alternative.get(
                        "reviews"
                    ),
                    "is_suspicious_pricing": alternative.get(
                        "is_suspicious_pricing",
                        False,
                    ),
                    "trade_off": trade_off,
                }
            )

        state[
            "structured_recommendation"
        ] = structured

        state[
            "final_recommendation"
        ] = (
            f"Top pick: "
            f"{_product_title(top_pick)} at "
            f"{_product_price_text(top_pick)}. "
            f"{bottom_line}"
        )

        # ---------------------------------------------------------------------
        # STEP 11: Latest product memory for UI.
        # ---------------------------------------------------------------------

        state[
            "last_shown_deals"
        ] = deals

        # ---------------------------------------------------------------------
        # STEP 12: Recent compact search history.
        # ---------------------------------------------------------------------

        history = state.setdefault(
            "search_history",
            [],
        )

        history.append(
            {
                "query": user_query,
                "products": [
                    _build_product_summary(
                        product
                    )
                    for product in sorted_deals
                ],
            }
        )

        state[
            "search_history"
        ] = history[
            -MAX_HISTORY_TURNS:
        ]

        # ---------------------------------------------------------------------
        # STEP 13: Existing conversation history.
        # ---------------------------------------------------------------------

        conversation_history = (
            state.setdefault(
                "conversation_history",
                [],
            )
        )

        conversation_history.append(
            {
                "user_query": user_query,
                "products_shown": [
                    _clean_plain_text(
                        product.get(
                            "title"
                        ),
                        "Unknown product",
                    )
                    for product in sorted_deals
                ],
            }
        )

        # ---------------------------------------------------------------------
        # Logging.
        # ---------------------------------------------------------------------

        agent_logger.info(
            "Synthesis completed successfully: "
            "top_pick=%r "
            "alternatives=%d "
            "product_memory_size=%d "
            "search_history_turns=%d "
            "hybrid_match=%.2f",
            top_pick.get(
                "title"
            ),
            len(alternatives),
            len(product_memory),
            len(
                state.get(
                    "search_history",
                    [],
                )
            ),
            _safe_float(
                top_pick.get(
                    "hybrid_match_score"
                )
            ),
        )

    except Exception as exc:

        agent_logger.error(
            "Error in synthesize_node: %s",
            exc,
            exc_info=True,
        )

        state.setdefault(
            "errors",
            [],
        ).append(
            f"Synthesis failed: {exc}"
        )

        state[
            "structured_recommendation"
        ] = None

        state[
            "final_recommendation"
        ] = (
            "I ran into an issue while putting together "
            "your final recommendations. Please try "
            "the search again."
        )

    return state