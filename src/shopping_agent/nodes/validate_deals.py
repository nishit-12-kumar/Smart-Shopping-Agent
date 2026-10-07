from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


# ============================================================================
# CONFIGURATION
# ============================================================================

# Hybrid match score will eventually be calculated in synthesize.py as:
#
#     70% deterministic requirement match
#     30% LLM validation / semantic relevance
#
# These are declared here for clarity/documentation.
DETERMINISTIC_MATCH_WEIGHT = 0.70
LLM_MATCH_WEIGHT = 0.30


# Internal weights for the deterministic requirement matcher.
#
# These weights are used ONLY among the requirements that are actually
# detected in the user's query. They are normalized automatically.
#
# Example:
#
# Query:
#     "laptop under 60000 with 16GB RAM"
#
# Only:
#     budget + RAM + category
#
# are considered.
REQUIREMENT_WEIGHTS = {
    "category": 20.0,
    "budget": 30.0,
    "ram": 20.0,
    "storage": 15.0,
    "processor": 10.0,
    "gpu": 5.0,
}


# Common shopping/product categories.
# This is intentionally small and extensible.
CATEGORY_KEYWORDS = {
    "laptop": {
        "laptop",
        "notebook",
        "macbook",
    },
    "phone": {
        "phone",
        "mobile",
        "smartphone",
        "iphone",
        "android",
    },
    "tablet": {
        "tablet",
        "ipad",
    },
    "tv": {
        "tv",
        "television",
    },
    "camera": {
        "camera",
        "dslr",
        "mirrorless",
    },
    "headphones": {
        "headphone",
        "headphones",
        "earphone",
        "earphones",
        "earbud",
        "earbuds",
        "tws",
    },
    "shoes": {
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
    },
    "watch": {
        "watch",
        "smartwatch",
    },
}


# ============================================================================
# STRUCTURED LLM OUTPUT
# ============================================================================


class ProductEvaluation(BaseModel):
    """
    LLM-generated qualitative validation.

    This remains separate from deterministic requirement matching.

    The LLM is mainly responsible for:
    - semantic relevance
    - general product suitability
    - review-signal reliability
    """

    confidence_score: int = Field(
        ge=0,
        le=100,
        description=(
            "A score from 0 to 100 indicating semantic relevance and "
            "overall suitability of this product for the user's request, "
            "including the reliability of its review signal. Do not use "
            "price or budget fit as the basis for this score; those are "
            "handled deterministically."
        ),
    )

    reasoning: str = Field(
        description=(
            "A one-line explanation of the semantic relevance and review "
            "reliability of the product."
        )
    )


class DealValidationOutput(BaseModel):
    """
    Structured LLM output for the batch of products.
    """

    evaluations: List[ProductEvaluation] = Field(
        description=(
            "List of evaluations matching the exact order of the input products."
        )
    )


# ============================================================================
# BASIC HELPERS
# ============================================================================


def _safe_float(value: Any) -> Optional[float]:
    """
    Safely convert values such as:
        60000
        "60000"
        "₹60,000"
        "60k"
    into float.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    text = text.replace(",", "")
    text = text.replace("₹", "")
    text = text.replace("rs.", "")
    text = text.replace("rs", "")

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac)?",
        text,
    )

    if not match:
        return None

    try:
        number = float(match.group(1))
    except ValueError:
        return None

    multiplier = match.group(2)

    if multiplier in ("k", "thousand"):
        number *= 1_000

    elif multiplier in ("lakh", "lac"):
        number *= 100_000

    return number


def _normalize_text(value: Any) -> str:
    """
    Normalize text for matching.
    """
    if value is None:
        return ""

    text = str(value).lower()

    # Keep useful alphanumeric characters.
    text = re.sub(r"[^a-z0-9.+#\s]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _product_text(product: Dict[str, Any]) -> str:
    """
    Combine product title + available structured fields.

    SerpAPI currently gives us mostly title/price/rating/review information,
    but this function also uses richer fields if another data source or
    future search implementation provides them.
    """
    fields = [
        product.get("title"),
        product.get("category"),
        product.get("brand"),
        product.get("ram"),
        product.get("storage"),
        product.get("processor"),
        product.get("gpu"),
        product.get("display"),
        product.get("network"),
        product.get("battery"),
        product.get("operating_system"),
    ]

    return _normalize_text(
        " ".join(
            str(value)
            for value in fields
            if value is not None
        )
    )


# ============================================================================
# LOW REVIEW SIGNAL
# ============================================================================


def _flag_low_review_count(
    product: dict,
) -> Optional[str]:

    rating = product.get("rating")
    reviews = product.get("reviews")

    if (
        rating
        and reviews
        and rating >= 4.5
        and reviews < 50
    ):
        return (
            f"⚠️ High rating ({rating}★) but only "
            f"{reviews} reviews — insufficient data to "
            f"fully trust this rating."
        )

    return None


# ============================================================================
# PRODUCT PROMPT FORMATTING
# ============================================================================


def _format_product_line(
    product: dict,
) -> str:
    """
    Formats a product safely for the LLM.
    """

    title = (
        product.get("title")
        or "Unknown product"
    )

    price = product.get("price")

    price_text = (
        f"₹{price}"
        if price is not None
        else "price unavailable"
    )

    rating = product.get("rating")

    rating_text = (
        f"{rating}"
        if rating is not None
        else "no rating"
    )

    reviews = product.get("reviews")

    reviews_text = (
        f"{reviews} reviews"
        if reviews is not None
        else "no review count"
    )

    return (
        f"- {title} | "
        f"Price: {price_text} | "
        f"Rating: {rating_text} ({reviews_text})"
    )


# ============================================================================
# QUERY REQUIREMENT EXTRACTION
# ============================================================================


def _extract_budget(
    query: str,
) -> Optional[float]:
    """
    Extract maximum budget from natural language.

    Examples:
        under 60000
        under ₹60,000
        below 60k
        budget of 70000
        within 80k
        upto ₹1 lakh
    """

    text = _normalize_text(query)

    patterns = [
        # under / below / less than / upto / up to / within
        r"(?:under|below|less than|upto|up to|within|max(?:imum)?(?:\s+budget)?|budget(?:\s+of|\s+is)?)"
        r"\s*(?:₹|rs|inr)?\s*"
        r"(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac)?",

        # "₹60000 budget"
        r"(?:₹|rs|inr)?\s*"
        r"(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac)?"
        r"\s*budget",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
        )

        if not match:
            continue

        number = float(
            match.group(1)
        )

        multiplier = match.group(2)

        if multiplier in (
            "k",
            "thousand",
        ):
            number *= 1_000

        elif multiplier in (
            "lakh",
            "lac",
        ):
            number *= 100_000

        return number

    return None


def _extract_ram_requirement(
    query: str,
) -> Optional[int]:
    """
    Extract RAM requirement.

    Examples:
        16GB RAM
        16 GB RAM
        RAM 16GB
        RAM of 16GB
    """

    text = _normalize_text(query)

    patterns = [
        r"(\d+)\s*gb\s*(?:of\s*)?ram",
        r"ram\s*(?:of|is|:|=)?\s*(\d+)\s*gb",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
        )

        if match:
            return int(
                match.group(1)
            )

    return None


def _extract_storage_requirement(
    query: str,
) -> Optional[Tuple[float, str]]:
    """
    Extract minimum storage requirement.

    Examples:
        512GB SSD
        1TB storage
        1 TB SSD
        storage 512GB
    """

    text = _normalize_text(query)

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:ssd|nvme|hdd|storage)",
        r"(?:storage|ssd|nvme|hdd)\s*(?:of|is|:|=)?\s*"
        r"(\d+(?:\.\d+)?)\s*(gb|tb)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
        )

        if not match:
            continue

        number = float(
            match.group(1)
        )

        unit = match.group(2)

        return number, unit

    return None


def _storage_to_gb(
    value: float,
    unit: str,
) -> float:

    unit = unit.lower()

    if unit == "tb":
        return value * 1024

    return value


def _extract_category(
    query: str,
) -> Optional[str]:
    """
    Detect a broad product category from the query.
    """

    text = _normalize_text(query)

    # Check longer/specific phrases first.
    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            if re.search(
                rf"\b{re.escape(keyword)}\b",
                text,
            ):
                return category

    return None


def _extract_processor_requirement(
    query: str,
) -> Optional[List[str]]:
    """
    Extract common processor families/models.

    This is intentionally conservative.
    If we cannot confidently extract a processor requirement,
    we simply don't include processor in the deterministic score.
    """

    text = _normalize_text(query)

    patterns = [
        r"\b(?:intel\s+)?i[3579]\b",
        r"\bryzen\s+[3579]\b",
        r"\b(?:apple\s+)?m[1-4]\b",
        r"\bsnapdragon\s+\w+\b",
        r"\bintel\s+ultra\s+[3579]\b",
    ]

    matches = []

    for pattern in patterns:
        found = re.findall(
            pattern,
            text,
        )

        matches.extend(found)

    cleaned = []

    for match in matches:
        value = _normalize_text(match)

        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned or None


def _extract_gpu_requirement(
    query: str,
) -> Optional[List[str]]:
    """
    Extract common GPU/model requirements.
    """

    text = _normalize_text(query)

    patterns = [
        r"\brtx\s+\d{3,4}(?:\s+ti)?\b",
        r"\bgtx\s+\d{3,4}(?:\s+ti)?\b",
        r"\bradeon\s+[a-z0-9]+\b",
        r"\brx\s+\d{3,4}\b",
        r"\biris\s+xe\b",
        r"\bmx\s+\d{3,4}\b",
    ]

    matches = []

    for pattern in patterns:
        found = re.findall(
            pattern,
            text,
        )

        matches.extend(found)

    cleaned = []

    for match in matches:
        value = _normalize_text(match)

        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned or None


def _extract_requirements(
    query: str,
) -> Dict[str, Any]:
    """
    Extract explicit, machine-checkable requirements from the query.

    This is NOT intended to replace the LLM.

    Its purpose is to identify hard/structured constraints that should be
    scored deterministically rather than delegated to the model.
    """

    requirements: Dict[str, Any] = {}

    budget = _extract_budget(query)

    if budget is not None:
        requirements["budget"] = budget

    ram = _extract_ram_requirement(query)

    if ram is not None:
        requirements["ram"] = ram

    storage = _extract_storage_requirement(query)

    if storage is not None:
        requirements["storage_gb"] = _storage_to_gb(
            storage[0],
            storage[1],
        )

    category = _extract_category(query)

    if category:
        requirements["category"] = category

    processor = _extract_processor_requirement(query)

    if processor:
        requirements["processor"] = processor

    gpu = _extract_gpu_requirement(query)

    if gpu:
        requirements["gpu"] = gpu

    return requirements


# ============================================================================
# PRODUCT ATTRIBUTE EXTRACTION
# ============================================================================


def _extract_product_ram(
    product: Dict[str, Any],
) -> Optional[int]:
    """
    Extract RAM from structured field first, then title.
    """

    structured_ram = product.get(
        "ram"
    )

    if structured_ram is not None:

        match = re.search(
            r"(\d+)\s*gb",
            str(structured_ram).lower(),
        )

        if match:
            return int(
                match.group(1)
            )

    title = _normalize_text(
        product.get("title")
    )

    patterns = [
        r"(\d+)\s*gb\s*(?:of\s*)?ram",
        r"ram\s*(?:of|is|:|=)?\s*(\d+)\s*gb",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            title,
        )

        if match:
            return int(
                match.group(1)
            )

    return None


def _extract_product_storage_gb(
    product: Dict[str, Any],
) -> Optional[float]:
    """
    Extract storage from structured field first, then title.

    We intentionally look for storage-related context so that
    "16GB RAM" isn't incorrectly interpreted as 16GB storage.
    """

    structured_storage = product.get(
        "storage"
    )

    if structured_storage is not None:

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(gb|tb)",
            str(structured_storage).lower(),
        )

        if match:

            return _storage_to_gb(
                float(match.group(1)),
                match.group(2),
            )

    title = _normalize_text(
        product.get("title")
    )

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:ssd|nvme|hdd|storage)",
        r"(\d+(?:\.\d+)?)\s*(tb)\s*(?:rom|storage)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            title,
        )

        if match:

            return _storage_to_gb(
                float(match.group(1)),
                match.group(2),
            )

    return None


def _product_matches_processor(
    product: Dict[str, Any],
    required_processors: List[str],
) -> Optional[float]:
    """
    Return:
        100 -> explicit processor requirement matched
        0   -> explicit processor info exists but doesn't match
        50  -> insufficient product information
    """

    product_text = _product_text(
        product
    )

    if not product_text:
        return 50.0

    for processor in required_processors:

        if processor in product_text:
            return 100.0

    # We know the query had a processor requirement but could not find
    # any matching processor in the product data.
    #
    # Since SerpAPI can provide incomplete fields, use a neutral score
    # rather than automatically assuming the product fails.
    has_processor_signal = any(
        token in product_text
        for token in [
            "intel",
            "ryzen",
            "snapdragon",
            "apple m",
            "core i",
            "ultra",
        ]
    )

    if has_processor_signal:
        return 0.0

    return 50.0


def _product_matches_gpu(
    product: Dict[str, Any],
    required_gpus: List[str],
) -> Optional[float]:

    product_text = _product_text(
        product
    )

    if not product_text:
        return 50.0

    for gpu in required_gpus:

        if gpu in product_text:
            return 100.0

    has_gpu_signal = any(
        token in product_text
        for token in [
            "rtx",
            "gtx",
            "radeon",
            "rx ",
            "iris xe",
            "mx ",
        ]
    )

    if has_gpu_signal:
        return 0.0

    return 50.0


# ============================================================================
# DETERMINISTIC REQUIREMENT MATCHING
# ============================================================================


def _category_match_score(
    product: Dict[str, Any],
    required_category: str,
) -> float:
    """
    Compare the required category with available product information.
    """

    product_text = _product_text(
        product
    )

    if not product_text:
        return 50.0

    category_keywords = CATEGORY_KEYWORDS.get(
        required_category,
        {required_category},
    )

    for keyword in category_keywords:

        if re.search(
            rf"\b{re.escape(keyword)}\b",
            product_text,
        ):
            return 100.0

    # We have a product title/data, but category doesn't match.
    return 0.0


def _budget_match_score(
    product: Dict[str, Any],
    budget: float,
) -> float:
    """
    Budget is a hard explicit requirement.

    <= budget -> 100
    > budget  -> 0
    missing price -> 50
    """

    price = _safe_float(
        product.get("price")
    )

    if price is None or price <= 0:
        return 50.0

    if price <= budget:
        return 100.0

    return 0.0


def _ram_match_score(
    product: Dict[str, Any],
    required_ram: int,
) -> float:
    """
    Treat requested RAM as a minimum requirement.

    Example:
        user asks 16GB
        product 32GB -> 100
        product 16GB -> 100
        product 8GB  -> 0
        RAM unknown   -> 50
    """

    product_ram = _extract_product_ram(
        product
    )

    if product_ram is None:
        return 50.0

    if product_ram >= required_ram:
        return 100.0

    return 0.0


def _storage_match_score(
    product: Dict[str, Any],
    required_storage_gb: float,
) -> float:

    product_storage = _extract_product_storage_gb(
        product
    )

    if product_storage is None:
        return 50.0

    if product_storage >= required_storage_gb:
        return 100.0

    return 0.0


def calculate_deterministic_match_score(
    product: Dict[str, Any],
    requirements: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, float], List[str]]:
    """
    Calculate a deterministic requirement-match score.

    IMPORTANT:
    Only requirements actually extracted from the user's query participate.

    Returns:
        score:
            0-100, or None if no structured requirements could be extracted.

        components:
            Individual requirement scores.

        reasons:
            Human-readable explanation of the deterministic score.
    """

    if not requirements:
        return None, {}, []

    components: Dict[str, float] = {}
    reasons: List[str] = []

    # -------------------------------------------------------------------------
    # CATEGORY
    # -------------------------------------------------------------------------

    if requirements.get("category"):

        score = _category_match_score(
            product,
            requirements["category"],
        )

        components["category"] = score

        if score == 100:
            reasons.append(
                "Product category matches the requested category."
            )
        elif score == 0:
            reasons.append(
                "Product category does not match the requested category."
            )
        else:
            reasons.append(
                "Product category could not be verified from available data."
            )

    # -------------------------------------------------------------------------
    # BUDGET
    # -------------------------------------------------------------------------

    if requirements.get("budget") is not None:

        budget = requirements["budget"]

        score = _budget_match_score(
            product,
            budget,
        )

        components["budget"] = score

        price = _safe_float(
            product.get("price")
        )

        if score == 100:
            reasons.append(
                f"Price is within the ₹{budget:,.0f} budget."
            )
        elif score == 0:
            reasons.append(
                f"Price exceeds the ₹{budget:,.0f} budget."
            )
        else:
            reasons.append(
                "Price is unavailable, so budget fit cannot be verified."
            )

    # -------------------------------------------------------------------------
    # RAM
    # -------------------------------------------------------------------------

    if requirements.get("ram") is not None:

        required_ram = requirements["ram"]

        score = _ram_match_score(
            product,
            required_ram,
        )

        components["ram"] = score

        product_ram = _extract_product_ram(
            product
        )

        if score == 100:
            reasons.append(
                f"RAM meets the requested minimum of "
                f"{required_ram}GB."
            )
        elif score == 0:
            reasons.append(
                f"RAM is below the requested "
                f"{required_ram}GB."
            )
        else:
            reasons.append(
                "RAM could not be verified from available product data."
            )

    # -------------------------------------------------------------------------
    # STORAGE
    # -------------------------------------------------------------------------

    if requirements.get("storage_gb") is not None:

        required_storage_gb = requirements[
            "storage_gb"
        ]

        score = _storage_match_score(
            product,
            required_storage_gb,
        )

        components["storage"] = score

        product_storage = _extract_product_storage_gb(
            product
        )

        if score == 100:
            reasons.append(
                f"Storage meets the requested minimum "
                f"of {required_storage_gb:g}GB."
            )
        elif score == 0:
            reasons.append(
                f"Storage is below the requested minimum "
                f"of {required_storage_gb:g}GB."
            )
        else:
            reasons.append(
                "Storage could not be verified from available product data."
            )

    # -------------------------------------------------------------------------
    # PROCESSOR
    # -------------------------------------------------------------------------

    if requirements.get("processor"):

        score = _product_matches_processor(
            product,
            requirements["processor"],
        )

        components["processor"] = score

        if score == 100:
            reasons.append(
                "Processor matches the requested processor family/model."
            )
        elif score == 0:
            reasons.append(
                "Processor does not match the requested processor."
            )
        else:
            reasons.append(
                "Processor could not be verified from available product data."
            )

    # -------------------------------------------------------------------------
    # GPU
    # -------------------------------------------------------------------------

    if requirements.get("gpu"):

        score = _product_matches_gpu(
            product,
            requirements["gpu"],
        )

        components["gpu"] = score

        if score == 100:
            reasons.append(
                "GPU matches the requested graphics requirement."
            )
        elif score == 0:
            reasons.append(
                "GPU does not match the requested graphics requirement."
            )
        else:
            reasons.append(
                "GPU could not be verified from available product data."
            )

    # -------------------------------------------------------------------------
    # WEIGHTED NORMALIZATION
    # -------------------------------------------------------------------------

    applicable_weight = 0.0
    weighted_total = 0.0

    for component_name, score in components.items():

        weight = REQUIREMENT_WEIGHTS.get(
            component_name,
            0.0,
        )

        applicable_weight += weight

        weighted_total += (
            score * weight
        )

    if applicable_weight <= 0:
        return None, components, reasons

    deterministic_score = (
        weighted_total
        / applicable_weight
    )

    return (
        round(
            max(
                0.0,
                min(
                    100.0,
                    deterministic_score,
                ),
            ),
            2,
        ),
        components,
        reasons,
    )


# ============================================================================
# VALIDATION NODE
# ============================================================================


def validate_deals_node(
    state: ShoppingState,
) -> ShoppingState:

    agent_logger.info(
        "Entering validate_deals_node."
    )

    try:

        raw_products = state.get(
            "raw_products",
            [],
        )

        user_query = state.get(
            "user_query",
            "",
        )

        if not raw_products:

            agent_logger.warning(
                "No raw products found to validate. Skipping."
            )

            return state

        # ---------------------------------------------------------------------
        # Extract deterministic requirements ONCE for this search.
        # ---------------------------------------------------------------------

        requirements = _extract_requirements(
            user_query
        )

        agent_logger.info(
            "Extracted deterministic requirements: %s",
            requirements,
        )

        # ---------------------------------------------------------------------
        # LLM setup
        # ---------------------------------------------------------------------

        groq_client = get_groq_llm()

        structured_llm = (
            groq_client.with_structured_output(
                DealValidationOutput
            )
        )

        # ---------------------------------------------------------------------
        # LLM prompt
        # ---------------------------------------------------------------------
        #
        # Important architecture change:
        #
        # Python will explicitly evaluate hard/structured requirements.
        #
        # The LLM should focus more on semantic relevance and review
        # reliability, not duplicate deterministic budget/spec checks.
        # ---------------------------------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert shopping assistant validating search results.

Evaluate each product against the user's original request.

Your score should primarily reflect:
1. Semantic relevance to what the user is asking for.
2. Overall suitability of the product.
3. Reliability of the review signal using rating and review count.

IMPORTANT:
- Hard structured requirements such as budget, RAM, storage, processor,
  and GPU are evaluated separately by deterministic Python logic.
- Do NOT factor price into your score.
- Do NOT invent product specifications.
- Do NOT assume that a cheap product is better.
- Do NOT assume that a high price is worse.
- Your score is a semantic/review validation signal that will later be
  combined with a deterministic requirement-match score.

Return a confidence_score between 0 and 100.
""".strip(),
                ),
                (
                    "human",
                    """
User Request:
{query}

Products to Evaluate:
{products}

Evaluate every product in the exact order provided.
""".strip(),
                ),
            ]
        )

        chain = prompt | structured_llm

        agent_logger.info(
            "Sending %d products to Groq for evaluation.",
            len(raw_products),
        )

        products_text = "\n".join(
            _format_product_line(product)
            for product in raw_products
        )

        result = chain.invoke(
            {
                "query": user_query,
                "products": products_text,
            }
        )

        # ---------------------------------------------------------------------
        # Validate LLM output count
        # ---------------------------------------------------------------------

        evaluations = result.evaluations

        if len(evaluations) != len(raw_products):

            agent_logger.error(
                "LLM returned %d evaluations for %d products — "
                "counts don't match, so positional alignment can't "
                "be trusted.",
                len(evaluations),
                len(raw_products),
            )

            if "errors" not in state:
                state["errors"] = []

            state["errors"].append(
                "Deal validation returned a mismatched number of "
                "evaluations; showing products with deterministic "
                "requirements only."
            )

            # Even if the LLM fails, deterministic matching can still
            # operate. This is a major benefit of separating the two.
            validated_deals = []

            for product in raw_products:

                validated_item = product.copy()

                (
                    deterministic_score,
                    deterministic_components,
                    deterministic_reasons,
                ) = calculate_deterministic_match_score(
                    validated_item,
                    requirements,
                )

                validated_item[
                    "deterministic_match_score"
                ] = deterministic_score

                validated_item[
                    "deterministic_match_components"
                ] = deterministic_components

                validated_item[
                    "deterministic_match_reasons"
                ] = deterministic_reasons

                validated_item[
                    "confidence_score"
                ] = None

                validated_item[
                    "reasoning"
                ] = (
                    "LLM validation was unavailable; "
                    "deterministic requirement matching was used."
                )

                validated_item[
                    "review_flag"
                ] = _flag_low_review_count(
                    product
                )

                validated_deals.append(
                    validated_item
                )

            state[
                "validated_deals"
            ] = validated_deals

            return state

        # ---------------------------------------------------------------------
        # Merge deterministic + LLM validation
        # ---------------------------------------------------------------------

        validated_deals = []

        for product, eval_data in zip(
            raw_products,
            evaluations,
        ):

            validated_item = product.copy()

            # LLM signal
            validated_item[
                "confidence_score"
            ] = max(
                0,
                min(
                    100,
                    int(
                        eval_data.confidence_score
                    ),
                ),
            )

            validated_item[
                "reasoning"
            ] = eval_data.reasoning

            # Deterministic signal
            (
                deterministic_score,
                deterministic_components,
                deterministic_reasons,
            ) = calculate_deterministic_match_score(
                validated_item,
                requirements,
            )

            validated_item[
                "deterministic_match_score"
            ] = deterministic_score

            validated_item[
                "deterministic_match_components"
            ] = deterministic_components

            validated_item[
                "deterministic_match_reasons"
            ] = deterministic_reasons

            # Existing review warning remains available.
            validated_item[
                "review_flag"
            ] = _flag_low_review_count(
                product
            )

            validated_deals.append(
                validated_item
            )

        state[
            "validated_deals"
        ] = validated_deals

        agent_logger.info(
            "Deal validation completed successfully. "
            "Deterministic requirements=%s",
            requirements,
        )

    except Exception as e:

        agent_logger.error(
            f"Error in validate_deals_node: {str(e)}",
            exc_info=True,
        )

        if "errors" not in state:
            state["errors"] = []

        state["errors"].append(
            f"Deal validation failed: {str(e)}"
        )

        # Existing safe fallback.
        state[
            "validated_deals"
        ] = state.get(
            "raw_products",
            [],
        )

    return state