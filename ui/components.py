import html as _html
import re
from urllib.parse import quote_plus, urlparse

import streamlit as st

from src.shopping_agent.nodes.synthesize import get_score_breakdown

# Score-badge thresholds, named instead of repeated magic numbers (80/60)
SCORE_HIGH_THRESHOLD = 80
SCORE_MID_THRESHOLD = 60

RISK_LEVEL_STYLE = {
    "LOW": ("Low Risk", "var(--success)"),
    "MEDIUM": ("Moderate Risk", "var(--accent-gold)"),
    "HIGH": ("High Risk", "var(--danger)"),
    "VERY HIGH": ("Very High Risk", "var(--danger)"),
}


# ============================================================================
# Small shared helpers
# ============================================================================

def _esc(value) -> str:
    """Escape a value for safe HTML embedding. None becomes an empty string."""
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)


def _score_class(score, prefix: str) -> str:
    """Shared logic behind every colour-coded score pill."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return f"{prefix}-mid"

    value = max(0.0, min(100.0, value))

    if value >= SCORE_HIGH_THRESHOLD:
        return f"{prefix}-high"
    if value >= SCORE_MID_THRESHOLD:
        return f"{prefix}-mid"
    return f"{prefix}-low"


def _score_pill(score) -> str:
    """Small colour-coded '73/100' pill used throughout the card."""
    if score is None:
        return '<span class="score-pill score-pill-mid">N/A</span>'
    cls = _score_class(score, "score-pill")
    try:
        display = f"{round(float(score))}/100"
    except (TypeError, ValueError):
        display = "N/A"
    return f'<span class="{cls}">{display}</span>'


def _parse_reviews(value):
    """Parse review counts such as 1234, '1,234', or '1,234 reviews'."""
    if value is None:
        return None
    match = re.search(r"\d[\d,]*", str(value))
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def render_star_rating(rating, reviews=None) -> str:
    """Render a real buyer rating without inventing missing data."""
    if rating is None or str(rating).strip() == "":
        return ""

    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return ""

    if not (0.0 <= rating_value <= 5.0):
        return ""

    full_stars = int(rating_value)
    half_star = 1 if rating_value - full_stars >= 0.5 else 0
    empty_stars = max(0, 5 - full_stars - half_star)

    stars_html = "★" * full_stars
    if half_star:
        stars_html += "⯨"
    stars_html += "☆" * empty_stars

    review_count = _parse_reviews(reviews)
    review_text = f" ({review_count:,} reviews)" if review_count is not None else ""

    return (
        f'<span class="star-rating">{stars_html}</span>'
        f'<span class="star-rating-count">{rating_value:.1f}/5{_esc(review_text)}</span>'
    )


def get_product_link(product: dict) -> str:
    """Return a safe product URL, with a Google Shopping fallback."""
    link = product.get("link")

    if link:
        candidate = str(link).strip()
        try:
            parsed = urlparse(candidate)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return candidate
        except ValueError:
            pass

    title = str(product.get("title") or "product").strip()
    return f"https://www.google.com/search?q={quote_plus(title)}&tbm=shop"


def _find_full_product(product_id, last_shown_deals: list) -> dict:
    """
    structured_recommendation only carries a curated subset of fields.
    For a few sections (score breakdown, full pricing-risk reasons) we
    need the full product dict, which only lives in last_shown_deals
    (state's "full product dicts from the MOST RECENT search"). Falls
    back to {} gracefully if the id can't be found (e.g. an older chat
    message being re-rendered after a newer search replaced
    last_shown_deals).
    """
    if not product_id or not last_shown_deals:
        return {}
    return next((p for p in last_shown_deals if p.get("id") == product_id), {}) or {}


# ============================================================================
# TOP PICK
# ============================================================================

def _render_score_panel(top: dict, last_shown_deals: list):
    overall = top.get("recommendation_score")
    overall_cls = _score_class(overall, "scorepanel-overall")
    color_map = {
        "scorepanel-overall-high": "var(--success)",
        "scorepanel-overall-mid": "var(--accent-gold)",
        "scorepanel-overall-low": "var(--danger)",
    }
    overall_color = color_map.get(overall_cls, "var(--accent-gold)")
    overall_display = f"{round(float(overall))}" if overall is not None else "N/A"

    breakdown = get_score_breakdown(top.get("id"), last_shown_deals)

    rows_html = ""
    if breakdown:
        row_defs = [
            ("Match (50%)", breakdown.get("match")),
            ("Rating (15%)", breakdown.get("rating")),
            ("Reviews (10%)", breakdown.get("reviews")),
            ("Price Value (10%)", breakdown.get("price_value")),
            ("Pricing Trust (15%)", breakdown.get("pricing_trust")),
        ]
        for label, val in row_defs:
            rows_html += (
                f'<div class="score-row">'
                f'<span class="score-row-label">{_esc(label)}</span>'
                f'{_score_pill(val)}'
                f'</div>'
            )
    else:
        # Breakdown unavailable (e.g. an older turn whose last_shown_deals
        # was overwritten by a later search) — fall back to just the
        # confidence score, which structured_recommendation always has.
        rows_html = (
            f'<div class="score-row">'
            f'<span class="score-row-label">Match (50%)</span>'
            f'{_score_pill(top.get("confidence_score"))}'
            f'</div>'
        )

    st.markdown(f"""
    <div class="toppick-scorepanel">
        <div class="scorepanel-label">Overall Score</div>
        <div class="scorepanel-overall" style="color:{overall_color};">
            {overall_display}<span class="suffix">/100</span>
        </div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


def _render_top_pick(top: dict, last_shown_deals: list):
    thumb = top.get("thumbnail")
    img_block = (
        f'<img src="{_esc(thumb)}" alt="{_esc(top.get("title"))}" />'
        if thumb else '<span class="toppick-noimg">📦</span>'
    )

    tags_html = "".join(
        f'<span class="tag-good">✓ {_esc(t)}</span>'
        for t in (top.get("specs_matched") or [])
    )
    if top.get("specs_warning"):
        tags_html += f'<span class="tag-warn">⚠ {_esc(top["specs_warning"])}</span>'

    price = top.get("price")
    price_display = f"₹{price:,.2f}" if isinstance(price, (int, float)) else (f"₹{_esc(price)}" if price else "Price unavailable")

    rating_html = render_star_rating(top.get("rating"), top.get("reviews"))

    st.html(f"""
    <div class="toppick-card">
        <div class="toppick-badge-row">
            <span class="toppick-badge">⭐ TOP PICK</span>
            <span class="toppick-subtext">Best match for your request</span>
        </div>
        <div class="toppick-grid">
            <div class="toppick-imgwrap">{img_block}</div>
            <div class="toppick-body">
                <p class="toppick-title">{_esc(top.get('title', 'Untitled product'))}</p>
                <div class="tag-row">{tags_html}</div>
                {f'<div style="margin-bottom:8px;">{rating_html}</div>' if rating_html else ''}
                <p class="toppick-price">{price_display}</p>
                <p class="toppick-store">via {_esc(top.get('source') or 'Unknown store')}</p>
                <p class="toppick-desc">{_esc(top.get('why_it_wins', ''))}</p>
            </div>
        </div>
    </div>
    """)


def _render_top_pick_full(top: dict, last_shown_deals: list, render_key: str):
    """Top pick card + score panel side-by-side, then the View Deal CTA."""
    col_main, col_score = st.columns([2.4, 1], gap="medium")
    with col_main:
        _render_top_pick(top, last_shown_deals)
    with col_score:
        _render_score_panel(top, last_shown_deals)

    st.link_button("View Deal ↗", get_product_link(top), key=f"top_link_{render_key}", use_container_width=True)


# ============================================================================
# ALTERNATIVES
# ============================================================================

def _render_alternatives(alternatives: list, render_key: str):
    if not alternatives:
        return

    st.html(
        f'<p class="section-eyebrow">Alternatives ({len(alternatives)})</p>'
    )

    cols = st.columns(len(alternatives)) if len(alternatives) <= 3 else st.columns(3)

    for idx, alt in enumerate(alternatives):
        col = cols[idx % len(cols)]

        with col:
            thumb = alt.get("thumbnail")
            title = str(alt.get("title") or "Untitled product")

            img_block = (
                f'<img src="{_esc(thumb)}" alt="{_esc(title)}" />'
                if thumb
                else '<span class="alt-noimg">📦</span>'
            )

            card_class = (
                "alt-card alt-card-suspicious"
                if alt.get("is_suspicious_pricing")
                else "alt-card"
            )

            price = alt.get("price")

            if isinstance(price, (int, float)) and not isinstance(price, bool):
                price_display = (
                    f"₹{int(price):,}"
                    if float(price).is_integer()
                    else f"₹{float(price):,.2f}"
                )
            else:
                price_display = (
                    f"₹{_esc(price)}"
                    if price is not None and str(price).strip()
                    else "N/A"
                )

            trade_off = str(
                alt.get("trade_off")
                or "A different trade-off compared with the top pick."
            ).strip()

            suspicious = bool(alt.get("is_suspicious_pricing"))

            trade_off_cls = (
                "alt-tradeoff alt-tradeoff-flagged"
                if suspicious
                else "alt-tradeoff"
            )

            trade_off_prefix = "⚠ " if suspicious else ""

            rating_html = render_star_rating(
                alt.get("rating"),
                alt.get("reviews"),
            )

            # IMPORTANT:
            # Use st.html(), NOT st.markdown().
            card_html = f"""
<div class="{card_class}">
    <div class="alt-imgwrap">{img_block}</div>

    <div class="alt-top-row">
        <p class="alt-title">{_esc(title)}</p>
    </div>

    <div class="alt-top-row">
        {_score_pill(alt.get("recommendation_score"))}
        <span class="alt-price">{price_display}</span>
    </div>

    {f'<div>{rating_html}</div>' if rating_html else ''}

    <p class="{trade_off_cls}">
        {trade_off_prefix}{_esc(trade_off)}
    </p>
</div>
"""

            st.html(card_html)

            st.link_button(
                "View Deal ↗",
                get_product_link(alt),
                key=f"alt_link_{render_key}_{idx}",
                use_container_width=True,
            )

# ============================================================================
# WHY THIS IS THE BEST MATCH
# ============================================================================

def _render_why_best_match(top: dict):
    items_html = ""

    if top.get("why_it_wins"):
        items_html += (
            f'<div class="checklist-item"><span class="checklist-icon-good">✓</span>'
            f'<span>{_esc(top["why_it_wins"])}</span></div>'
        )

    for spec in (top.get("specs_matched") or []):
        items_html += (
            f'<div class="checklist-item"><span class="checklist-icon-good">✓</span>'
            f'<span>Matches: {_esc(spec)}</span></div>'
        )

    if top.get("specs_warning"):
        items_html += (
            f'<div class="checklist-item"><span class="checklist-icon-warn">⚠</span>'
            f'<span>{_esc(top["specs_warning"])}</span></div>'
        )

    reviews = _parse_reviews(top.get("reviews"))
    if reviews is not None and reviews < 50:
        items_html += (
            '<div class="checklist-item"><span class="checklist-icon-warn">⚠</span>'
            '<span>Limited review count — more data would increase confidence.</span></div>'
        )

    if not items_html:
        items_html = '<div class="checklist-item"><span>No additional match details available for this listing.</span></div>'

    st.markdown(f"""
    <div class="info-card">
        <p class="info-card-title">✅ Why This Is The Best Match</p>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PRICING TRUST
# ============================================================================

def _render_pricing_trust(top: dict, last_shown_deals: list):
    risk_score = top.get("pricing_risk_score")
    risk_level = top.get("pricing_risk_level")

    if risk_score is None:
        trust_score = None
        ring_color = "var(--text-secondary)"
        ring_pct = 0
        risk_label = "Not Assessed"
        label_color = "var(--text-secondary)"
    else:
        trust_score = max(0, min(100, round(100 - float(risk_score))))
        ring_pct = trust_score
        risk_key = str(risk_level or "").strip().upper()
        risk_label, label_color = RISK_LEVEL_STYLE.get(risk_key, ("Unknown", "var(--text-secondary)"))
        ring_color = label_color

    ring_value_display = trust_score if trust_score is not None else "—"

    full_product = _find_full_product(top.get("id"), last_shown_deals)
    reasons = full_product.get("pricing_risk_reasons") or []

    reasons_html = ""
    if reasons:
        for r in reasons:
            reasons_html += f'<div class="keyspec-row"><span>•</span><span>{_esc(r)}</span></div>'
    else:
        reasons_html = '<div class="keyspec-row"><span>•</span><span>No pricing risk signals detected for this listing.</span></div>'

    st.markdown(f"""
    <div class="info-card">
        <p class="info-card-title">🛡️ Pricing Trust (Deterministic)</p>
        <div class="trust-ring-wrap">
            <div class="trust-ring" style="background: conic-gradient({ring_color} calc({ring_pct} * 1%), var(--border-subtle) 0);">
                <div class="trust-ring-inner">
                    <span class="trust-ring-value">{ring_value_display}</span>
                    <span class="trust-ring-max">/100</span>
                </div>
            </div>
            <div>
                <div class="trust-risk-label" style="color:{label_color};">{_esc(risk_label)}</div>
            </div>
        </div>
        {reasons_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# KEY SPECIFICATIONS
# ============================================================================

def _render_key_specs(top: dict):
    rows_html = ""

    for spec in (top.get("specs_matched") or []):
        rows_html += (
            f'<div class="keyspec-row"><span class="checklist-icon-good">✓</span>'
            f'<span>{_esc(spec)}</span></div>'
        )

    if top.get("specs_warning"):
        rows_html += (
            f'<div class="keyspec-row"><span class="checklist-icon-warn">⚠</span>'
            f'<span>{_esc(top["specs_warning"])}</span></div>'
        )

    if not rows_html:
        rows_html = '<div class="keyspec-row"><span>No confirmed specifications available for this listing.</span></div>'

    st.markdown(f"""
    <div class="info-card">
        <p class="info-card-title">📋 Key Specifications</p>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RED FLAGS / FILTERED NOTE / BOTTOM LINE / FOLLOW-UPS
# ============================================================================

def _render_red_flags(red_flags: list):
    """
    red_flags is a list of DICTS: {id, title, risk_score, risk_level,
    reasons}. Each one is rendered as its own detail box — never dumped
    as a raw dict into the page.
    """
    if not red_flags:
        return

    st.markdown('<p class="section-eyebrow">⚠️ Pricing Risk Flags</p>', unsafe_allow_html=True)

    for flag in red_flags:
        title = flag.get("title", "Untitled product")
        risk_score = flag.get("risk_score")
        risk_level = flag.get("risk_level")
        reasons = flag.get("reasons") or []

        score_text = f"{risk_score}/100" if risk_score is not None else "N/A"
        risk_key = str(risk_level or "").strip().upper()
        level_label, level_color = RISK_LEVEL_STYLE.get(risk_key, ("Unknown", "var(--danger)"))

        reasons_html = "".join(
            f'<div class="keyspec-row"><span>•</span><span>{_esc(r)}</span></div>' for r in reasons
        )

        st.markdown(f"""
        <div class="info-card" style="border-color: var(--danger); margin-bottom:10px;">
            <p class="info-card-title" style="color:var(--danger);">⚠️ {_esc(title)}</p>
            <p style="font-size:12px; color:var(--text-secondary); margin:0 0 8px;">
                Risk score: <b style="color:{level_color};">{_esc(score_text)} ({_esc(level_label)})</b>
            </p>
            {reasons_html}
        </div>
        """, unsafe_allow_html=True)


def _render_filtered_note(note: str):
    if not note:
        return
    st.markdown(f'<div class="info-banner">ℹ️ {_esc(note)}</div>', unsafe_allow_html=True)


def _render_bottom_line(bottom_line: str):
    if not bottom_line:
        return
    st.markdown(f"""
    <div class="bottomline-box">
        <p class="bottomline-title">⭐ Bottom Line</p>
        <p class="bottomline-text">{_esc(bottom_line)}</p>
    </div>
    """, unsafe_allow_html=True)


def _render_follow_up_chips(suggestions: list, render_key: str):
    clean_suggestions = []
    for suggestion in suggestions or []:
        text = str(suggestion or "").strip()
        if text and text not in clean_suggestions:
            clean_suggestions.append(text)

    if not clean_suggestions:
        return

    clean_suggestions = clean_suggestions[:4]

    st.markdown(
        '<p class="ask-next-title">💬 You might want to ask</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(clean_suggestions))
    for i, suggestion in enumerate(clean_suggestions):
        with cols[i]:
            if st.button(
                f"💬 {suggestion}",
                key=f"chip_{render_key}_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_chip_prompt = suggestion
                st.rerun()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def render_recommendation_card(structured: dict, last_shown_deals: list = None, render_key: str = ""):
    if not structured:
        return

    last_shown_deals = last_shown_deals or []
    top = structured.get("top_pick") or {}

    if not top:
        st.info("No recommendation details are available for this turn.")
        return

    _render_top_pick_full(top, last_shown_deals, render_key)
    _render_alternatives(structured.get("alternatives") or [], render_key)
    _render_filtered_note(structured.get("filtered_out_note"))

    st.markdown('<p class="section-eyebrow">Deal Breakdown</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        _render_why_best_match(top)
    with col2:
        _render_pricing_trust(top, last_shown_deals)
    with col3:
        _render_key_specs(top)

    _render_red_flags(structured.get("red_flags") or [])
    _render_bottom_line(structured.get("bottom_line"))
    _render_follow_up_chips(structured.get("follow_up_suggestions") or [], render_key)