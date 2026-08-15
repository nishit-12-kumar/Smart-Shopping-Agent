import re
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import get_groq_llm
from src.shopping_agent.utils.logger import agent_logger


# ============================================================================
# CONFIGURATION
# ============================================================================

# Maximum discount that is normally expected for a category.
# IMPORTANT:
# These are NOT proof that a discount is fake.
# They are only one signal contributing to the pricing-risk score.
CATEGORY_MAX_DISCOUNT = {
    "laptop": 0.40,
    "phone": 0.40,
    "mobile": 0.40,
    "smartphone": 0.40,
    "electronics": 0.45,
    "tv": 0.45,
    "television": 0.45,
    "camera": 0.45,
    "shoes": 0.65,
    "clothing": 0.70,
    "fashion": 0.70,
    "furniture": 0.55,
    "appliance": 0.45,
}

DEFAULT_MAX_DISCOUNT = 0.55


# Original price / current price ratios that deserve extra attention.
#
# Example:
# ₹80,000 / ₹40,000 = 2.0
#
# This does NOT mean the price is fake.
# It only adds risk points.
SUSPICIOUS_ROUND_RATIOS = [2.0, 2.5, 3.0, 4.0, 5.0]
ROUND_RATIO_TOLERANCE = 0.05


# Minimum number of products required in a peer group (including the
# product itself) before we trust a peer-based price comparison at all.
MIN_PEERS_FOR_COMPARISON = 3

# Jaccard similarity (on significant title tokens, storage/color
# stripped out) required for two products to be treated as peers.
#
# This is a lightweight, deterministic heuristic — not a perfect
# classifier. At larger scale you'd likely swap this for embedding-based
# similarity, but it's enough to stop something like "iPhone case"
# (₹1,000) from being averaged into the same price distribution as the
# "iPhone 16" phones it's an accessory for.
PEER_SIMILARITY_THRESHOLD = 0.5

# Modified z-score (MAD-based, robust to outliers unlike a mean/stdev
# z-score) beyond which a peer-group price is flagged. 3.5 is the
# commonly used threshold for this statistic (Iglewicz & Hoaglin).
MAD_Z_HIGH_THRESHOLD = 3.5


# Risk score contribution of each rule.
RISK_EXCESSIVE_DISCOUNT = 25
RISK_ROUND_MRP = 20
RISK_PEER_OUTLIER = 30
RISK_LOW_REVIEWS = 10
RISK_LOW_RATING = 10


# Risk-level thresholds.
LOW_RISK_THRESHOLD = 25
MEDIUM_RISK_THRESHOLD = 50
HIGH_RISK_THRESHOLD = 75


# ============================================================================
# CATEGORY HELPERS
# ============================================================================

def _category_threshold(product: Dict[str, Any], query: str = "") -> float:
    """
    Returns the maximum normal discount threshold.

    Priority:
    1. Product category
    2. User query
    3. Default threshold

    Product category is preferred because using only the query is fragile.
    """

    category = str(product.get("category", "")).lower().strip()

    if category:
        for keyword, threshold in CATEGORY_MAX_DISCOUNT.items():
            if keyword in category:
                return threshold

    # Fallback for products where category is not available.
    query_lower = query.lower()

    for keyword, threshold in CATEGORY_MAX_DISCOUNT.items():
        if keyword in query_lower:
            return threshold

    return DEFAULT_MAX_DISCOUNT


# ============================================================================
# PEER GROUPING
# ============================================================================
#
# Search Results -> Product Matching -> Peer Group -> Median + MAD -> Outlier Detection -> Risk Score
#
# Everything in this section exists to answer one question correctly:
# "which OTHER products in this result set are actually comparable to
# this one?" — an iPhone case is not a peer of an iPhone, and comparing
# its ₹1,000 price against a ₹65,000 phone would make both the median
# and the outlier flags meaningless.

ACCESSORY_KEYWORDS = {
    "case", "cover", "screen protector", "tempered glass", "screen guard",
    "charger", "charging cable", "cable", "adapter", "stand", "strap",
    "pouch", "skin", "sleeve", "mount", "holder", "protector", "stylus",
    "cleaning kit", "charging dock", "power bank",
}

# Storage sizes and colors are the kind of thing that differs between
# variants of the *same* product line (e.g. "128GB" vs "256GB") — they
# should be stripped out before comparing titles, so variants still
# match each other.
_STORAGE_COLOR_RE = re.compile(
    r"\b\d+\s?(gb|tb|mb)\b|"
    r"\b(black|white|blue|red|green|gold|silver|gray|grey|"
    r"space\s?gray|space\s?grey|purple|pink|yellow|titanium|"
    r"graphite|midnight|starlight|natural)\b",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def _significant_tokens(title: str) -> set:
    """
    Reduces a product title down to the tokens that identify *what
    product this is* (brand + model), stripping variant-only details
    (storage size, color) so "iPhone 16 128GB" and "iPhone 16 256GB"
    are still recognised as the same underlying product line.
    """
    title = title.lower()
    title = _STORAGE_COLOR_RE.sub(" ", title)
    title = _NON_ALNUM_RE.sub(" ", title)
    return {token for token in title.split() if len(token) > 1}


def _is_accessory(title: str) -> bool:
    """True if the title looks like an accessory rather than a standalone product (e.g. 'iPhone case')."""
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in ACCESSORY_KEYWORDS)


def _title_similarity(tokens_a: set, tokens_b: set) -> float:
    """Jaccard similarity between two significant-token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def build_peer_groups(products: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Clusters `products` (by index into the list) into peer groups for
    price comparison, using union-find over pairwise title similarity.

    Accessories never merge into a peer group with anything else — they
    always end up as their own singleton group, since averaging a
    ₹1,000 case into a ₹65,000 phone's peer stats would be meaningless
    in both directions.
    """
    n = len(products)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    titles = [str(p.get("title", "")) for p in products]
    token_sets = [_significant_tokens(t) for t in titles]
    accessory_flags = [_is_accessory(t) for t in titles]

    for i in range(n):
        if accessory_flags[i]:
            continue
        for j in range(i + 1, n):
            if accessory_flags[j]:
                continue
            if _title_similarity(token_sets[i], token_sets[j]) >= PEER_SIMILARITY_THRESHOLD:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    return list(groups.values())


def _group_price_stats(
    products: List[Dict[str, Any]],
    group_indices: List[int],
) -> Optional[Dict[str, float]]:
    """
    Computes peer median + MAD ONCE for an entire peer group.

    This is the fix for the original performance bug: every product in
    the group now does an O(1) lookup against this precomputed dict
    instead of each product separately re-scanning the whole product
    list and recomputing statistics.median() from scratch (which is an
    O(n log n) sort, done n times => effectively O(n^2 log n) for a
    batch of n products).

    It is also the fix for the correctness bug: `group_indices` is
    already a peer group from build_peer_groups(), so this median is
    computed only across genuinely comparable products — never the
    full, unrelated batch of search results.
    """
    prices = []
    for idx in group_indices:
        price = products[idx].get("price")
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if price > 0:
            prices.append(price)

    if len(prices) < MIN_PEERS_FOR_COMPARISON:
        return None

    med = statistics.median(prices)
    mad = statistics.median([abs(p - med) for p in prices])

    return {"median": med, "mad": mad, "n": len(prices)}


def build_peer_stats_by_index(
    products: List[Dict[str, Any]]
) -> Dict[int, Optional[Dict[str, float]]]:
    """
    Runs product matching + peer grouping + stats calculation for the
    whole batch ONCE, returning a {product_index: peer_stats} lookup
    that calculate_price_risk can use for every product in O(1).
    """
    groups = build_peer_groups(products)
    stats_by_index: Dict[int, Optional[Dict[str, float]]] = {}

    for group in groups:
        group_stats = _group_price_stats(products, group)
        for idx in group:
            stats_by_index[idx] = group_stats

    return stats_by_index


# ============================================================================
# PRICE VALIDITY RULES
# ============================================================================

def check_excessive_discount(
    product: Dict[str, Any],
    query: str = "",
) -> Tuple[bool, Optional[str]]:
    """
    Checks whether the claimed discount is unusually large
    for the product category.

    This is only a risk signal, not proof of fake pricing.
    """

    original = product.get("original_price")
    current = product.get("price")

    if original is None or current is None:
        return False, None

    try:
        original = float(original)
        current = float(current)
    except (TypeError, ValueError):
        return False, None

    if original <= 0 or current <= 0 or original <= current:
        return False, None

    discount_pct = (original - current) / original

    threshold = _category_threshold(product, query)

    if discount_pct > threshold:
        return True, (
            f"Claimed discount of {discount_pct:.0%} is above the "
            f"typical ~{threshold:.0%} ceiling for this category"
        )

    return False, None


def check_round_multiple_mrp(
    product: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Checks whether the original price is an unusually clean
    multiple of the current price.

    Example:
        Original = ₹80,000
        Current  = ₹40,000
        Ratio    = 2.0

    This is NOT proof of fake MRP.
    It is only a supporting risk signal.
    """

    original = product.get("original_price")
    current = product.get("price")

    if original is None or current is None:
        return False, None

    try:
        original = float(original)
        current = float(current)
    except (TypeError, ValueError):
        return False, None

    if original <= 0 or current <= 0 or original <= current:
        return False, None

    ratio = original / current

    for suspicious_ratio in SUSPICIOUS_ROUND_RATIOS:
        if abs(ratio - suspicious_ratio) <= ROUND_RATIO_TOLERANCE:
            return True, (
                f"Original price is almost exactly "
                f"{suspicious_ratio:g}x the current price"
            )

    return False, None


def check_price_outlier(
    product: Dict[str, Any],
    peer_stats: Optional[Dict[str, float]],
) -> Tuple[bool, Optional[str]]:
    """
    Flags a product whose price is a statistical outlier *within its
    own peer group* (same product line — storage/color variants aside)
    rather than against the entire, unrelated batch of search results.

    peer_stats is precomputed once per group by build_peer_stats_by_index
    / _group_price_stats — this function does an O(1) lookup + a single
    comparison, it never rescans the product list.

    Uses a MAD-based modified z-score, which — unlike a mean/stdev
    z-score — isn't itself thrown off by the same extreme values it's
    trying to detect. Flags both directions:
      - suspiciously CHEAP      -> possible scam / bait listing
      - suspiciously EXPENSIVE  -> price inflated relative to true peers
    """
    if not peer_stats:
        return False, None

    current = product.get("price")

    try:
        current = float(current)
    except (TypeError, ValueError):
        return False, None

    if current <= 0:
        return False, None

    median_price = peer_stats["median"]
    mad = peer_stats["mad"]

    if median_price <= 0:
        return False, None

    deviation_pct = (current - median_price) / median_price

    if mad > 0:
        # 0.6745 scales MAD so it's comparable to a normal-distribution
        # standard deviation (the standard "modified z-score" formula).
        modified_z = 0.6745 * (current - median_price) / mad
    else:
        # Every peer is priced identically -> MAD is 0 and division is
        # undefined. Any real difference from that shared price is
        # inherently notable, so fall back to a straight percentage.
        if deviation_pct == 0:
            return False, None
        modified_z = math.copysign(MAD_Z_HIGH_THRESHOLD, deviation_pct)

    if modified_z <= -MAD_Z_HIGH_THRESHOLD:
        return True, (
            f"Priced at ₹{current:,.0f}, {deviation_pct:.0%} below the peer "
            f"median of ₹{median_price:,.0f} across {peer_stats['n']} similar "
            f"listings — unusually cheap for this product line"
        )

    if modified_z >= MAD_Z_HIGH_THRESHOLD:
        return True, (
            f"Priced at ₹{current:,.0f}, {deviation_pct:+.0%} above the peer "
            f"median of ₹{median_price:,.0f} across {peer_stats['n']} similar "
            f"listings — unusually expensive for this product line"
        )

    return False, None


def check_low_reviews(
    product: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Low review count does not mean the price is fake.
    It simply reduces confidence in the pricing signal.
    """

    reviews = product.get("reviews")

    if reviews is None:
        return False, None

    try:
        reviews = int(reviews)
    except (TypeError, ValueError):
        return False, None

    if reviews < 10:
        return True, (
            f"Very low review count ({reviews}), so the listing "
            "has limited social proof"
        )

    return False, None


def check_low_rating(
    product: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Very low ratings can indicate a less trustworthy listing.
    """

    rating = product.get("rating")

    if rating is None:
        return False, None

    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return False, None

    if 0 < rating < 3.0:
        return True, (
            f"Low product rating ({rating:.1f}/5), which reduces "
            "confidence in the listing"
        )

    return False, None


# ============================================================================
# PRICE RISK CALCULATION
# ============================================================================

def calculate_price_risk(
    product: Dict[str, Any],
    query: str,
    peer_stats: Optional[Dict[str, float]],
) -> Tuple[int, List[str]]:
    """
    Calculates a deterministic pricing-risk score.

    IMPORTANT:
    The LLM is NOT involved in this calculation.

    `peer_stats` is this product's peer group's precomputed median/MAD
    (see build_peer_stats_by_index), not the raw product list — the
    caller computes it once per group up front, so this function is now
    a fixed amount of work per product regardless of how many products
    were returned by the search.

    Score:
        0-24   -> LOW
        25-49  -> MEDIUM
        50-74  -> HIGH
        75-100 -> VERY HIGH
    """

    score = 0
    reasons: List[str] = []

    # ------------------------------------------------------------------
    # Rule 1: Excessive claimed discount
    # ------------------------------------------------------------------

    hit, reason = check_excessive_discount(product, query)

    if hit:
        score += RISK_EXCESSIVE_DISCOUNT

        if reason:
            reasons.append(reason)

    # ------------------------------------------------------------------
    # Rule 2: Suspicious round-number MRP
    # ------------------------------------------------------------------

    hit, reason = check_round_multiple_mrp(product)

    if hit:
        score += RISK_ROUND_MRP

        if reason:
            reasons.append(reason)

    # ------------------------------------------------------------------
    # Rule 3: Price outlier within this product's peer group
    # ------------------------------------------------------------------

    hit, reason = check_price_outlier(product, peer_stats)

    if hit:
        score += RISK_PEER_OUTLIER

        if reason:
            reasons.append(reason)

    # ------------------------------------------------------------------
    # Rule 4: Very low review count
    # ------------------------------------------------------------------

    hit, reason = check_low_reviews(product)

    if hit:
        score += RISK_LOW_REVIEWS

        if reason:
            reasons.append(reason)

    # ------------------------------------------------------------------
    # Rule 5: Very low rating
    # ------------------------------------------------------------------

    hit, reason = check_low_rating(product)

    if hit:
        score += RISK_LOW_RATING

        if reason:
            reasons.append(reason)

    # Never allow the score to exceed 100.
    score = min(score, 100)

    return score, reasons


def get_risk_level(score: int) -> str:
    """
    Converts numeric pricing risk into a human-readable level.
    """

    if score >= HIGH_RISK_THRESHOLD:
        return "VERY HIGH"

    if score >= MEDIUM_RISK_THRESHOLD:
        return "HIGH"

    if score >= LOW_RISK_THRESHOLD:
        return "MEDIUM"

    return "LOW"


# ============================================================================
# LLM EXPLANATION
# ============================================================================

class PricingExplanation(BaseModel):
    """
    LLM is ONLY responsible for converting deterministic findings
    into a short shopper-friendly explanation.
    """

    explanation: str = Field(
        description=(
            "One or two concise sentences explaining the pricing concern. "
            "Use ONLY the supplied rule findings and risk score. "
            "Do not invent additional facts."
        )
    )


class PricingExplanationOutput(BaseModel):
    explanations: List[PricingExplanation] = Field(
        description=(
            "One explanation for each flagged product, in exactly "
            "the same order as the input products."
        )
    )


def _explain_flagged_products(
    flagged: List[Dict[str, Any]],
    groq_client
) -> None:
    """
    Uses the LLM only for natural-language explanation.

    The LLM NEVER decides whether a product is suspicious.
    """

    structured_llm = groq_client.with_structured_output(
        PricingExplanationOutput
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a shopping assistant explaining pricing-risk signals.

The pricing risk score and rule findings have ALREADY been
calculated deterministically by Python.

Your job is ONLY to convert those findings into a short,
clear explanation for the shopper.

STRICT RULES:
- Do not change the risk level.
- Do not invent additional reasons.
- Do not claim that a product is definitely fraudulent or fake.
- Use cautious language such as "worth checking", "unusual",
  or "pricing may warrant verification".
- Do not invent prices, specifications, sellers, or product facts.
- Return exactly one explanation for each product.
- Keep each explanation to 1-2 sentences.
            """,
        ),
        (
            "human",
            """
Explain the pricing-risk findings for these products:

{products}
            """,
        ),
    ])

    products_text = "\n\n".join(
        (
            f"Product: {product.get('title')}\n"
            f"Price risk score: {product.get('pricing_risk_score')}/100\n"
            f"Risk level: {product.get('pricing_risk_level')}\n"
            f"Findings: "
            f"{'; '.join(product.get('pricing_risk_reasons', []))}"
        )
        for product in flagged
    )

    chain = prompt | structured_llm

    result = chain.invoke({
        "products": products_text
    })

    # Only update products for which the LLM actually returned an explanation.
    for product, explanation in zip(
        flagged,
        result.explanations
    ):
        if explanation.explanation:
            product["pricing_analysis"] = explanation.explanation


# ============================================================================
# MAIN NODE
# ============================================================================

def price_validity_node(state: ShoppingState) -> ShoppingState:
    """
    Detects potentially problematic pricing using deterministic
    rule-based checks.

    Pipeline:

        validated_deals
              |
        product matching -> peer groups          (build_peer_groups)
              |
        peer median + MAD, once per group          (build_peer_stats_by_index)
              |
        pricing-risk rules (per product, O(1) peer lookup)
              |
        pricing_risk_score / pricing_risk_level
              |
        LLM explanation
              |
        validated_deals

    The LLM NEVER determines the actual pricing-risk verdict.
    """

    agent_logger.info("Entering price_validity_node.")

    try:

        deals = state.get("validated_deals", [])

        if not deals:
            agent_logger.warning(
                "No validated deals found. Skipping price validity check."
            )
            return state

        query = state.get("user_query", "")

        flagged: List[Dict[str, Any]] = []

        # ================================================================
        # Product matching + peer groups + peer stats — computed ONCE
        # for the whole batch, not once per product.
        # ================================================================

        peer_stats_by_index = build_peer_stats_by_index(deals)

        agent_logger.info(
            f"Grouped {len(deals)} products into peer groups for price "
            f"comparison (peer stats available for "
            f"{sum(1 for v in peer_stats_by_index.values() if v)} of them)."
        )

        # ================================================================
        # Calculate pricing risk for every product
        # ================================================================

        for idx, product in enumerate(deals):

            risk_score, reasons = calculate_price_risk(
                product=product,
                query=query,
                peer_stats=peer_stats_by_index.get(idx),
            )

            risk_level = get_risk_level(risk_score)

            product["pricing_risk_score"] = risk_score
            product["pricing_risk_level"] = risk_level
            product["pricing_risk_reasons"] = reasons

            # Keep this field for compatibility with your existing
            # synthesize_node.
            product["is_suspicious_pricing"] = (
                risk_score >= MEDIUM_RISK_THRESHOLD
            )

            # Deterministic fallback explanation.
            if reasons:
                product["pricing_analysis"] = (
                    "; ".join(reasons)
                )

                flagged.append(product)

            else:
                product["pricing_analysis"] = (
                    "No significant pricing red flags detected."
                )

        agent_logger.info(
            f"Pricing risk calculated for {len(deals)} products. "
            f"{len(flagged)} products have pricing-risk signals."
        )

        # ================================================================
        # LLM explanation
        # ================================================================

        if flagged:

            try:

                groq_client = get_groq_llm()

                _explain_flagged_products(
                    flagged,
                    groq_client
                )

            except Exception as e:

                # IMPORTANT:
                # Pricing decisions have already been made by Python.
                # If the LLM fails, the pipeline still works.
                agent_logger.error(
                    "LLM pricing explanation failed. "
                    f"Using deterministic explanations instead: {str(e)}",
                    exc_info=True,
                )

        # ================================================================
        # Save results
        # ================================================================

        state["validated_deals"] = deals

        agent_logger.info(
            "Price validity check completed successfully."
        )

    except Exception as e:

        agent_logger.error(
            f"Error in price_validity_node: {str(e)}",
            exc_info=True,
        )

        if "errors" not in state:
            state["errors"] = []

        state["errors"].append(
            f"Price validity check failed: {str(e)}"
        )

        # Do NOT mark products as suspicious when the entire
        # pricing-analysis system itself crashes.
        #
        # Instead, explicitly indicate that analysis was unavailable.
        for deal in state.get("validated_deals", []):

            deal["pricing_risk_score"] = None
            deal["pricing_risk_level"] = "UNKNOWN"
            deal["pricing_risk_reasons"] = []
            deal["is_suspicious_pricing"] = False
            deal["pricing_analysis"] = (
                "Pricing analysis unavailable due to a system error."
            )

    return state