import streamlit as st
import time
from src.shopping_agent.graph.builder import build_graph
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.utils.logger import agent_logger
from src.shopping_agent.utils.spec_options import generate_spec_options, SKIP_LABEL, OTHER_LABEL
from src.shopping_agent.utils.logger import start_new_conversation_log

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Shopping Agent",
    page_icon="🛒",
    layout="centered"
)

st.markdown("""
<style>
    [data-testid="stChatMessage"] {
        background-color: #1a1d24;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        border: 1px solid #2a2e38;
    }
    .toppick-card {
        background-color: #1a1d24;
        border: 2px solid #f0c419;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 14px;
    }
    .toppick-img { width: 100%; height: 180px; object-fit: cover; display: block; }
    .toppick-body { padding: 14px 16px; }
    .badge-score-high { background: rgba(46,160,67,0.15); color: #3fb950; font-size: 11px; padding: 3px 9px; border-radius: 6px; }
    .badge-score-mid { background: rgba(240,196,25,0.15); color: #f0c419; font-size: 11px; padding: 3px 9px; border-radius: 6px; }
    .badge-score-low { background: rgba(231,76,60,0.15); color: #e74c3c; font-size: 11px; padding: 3px 9px; border-radius: 6px; }
    .spec-tag-good { background: rgba(46,160,67,0.12); color: #3fb950; font-size: 11px; padding: 3px 8px; border-radius: 6px; margin-right: 6px; }
    .spec-tag-warn { background: rgba(240,196,25,0.12); color: #f0c419; font-size: 11px; padding: 3px 8px; border-radius: 6px; }
    .alt-row { background-color: #1e222b; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; display: flex; gap: 12px; align-items: center; }
    .alt-img { width: 52px; height: 52px; object-fit: cover; border-radius: 6px; flex-shrink: 0; }
    .redflag-box { background: rgba(231,76,60,0.08); border-left: 4px solid #e74c3c; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
    .bottomline-box { border-top: 1px solid #2a2e38; padding-top: 18px; margin-top: 14px; }
</style>
""", unsafe_allow_html=True)


def render_score_badge(score):
    if score is None:
        return ""
    cls = "badge-score-high" if score >= 80 else "badge-score-mid" if score >= 60 else "badge-score-low"
    return f'<span class="{cls}">{score}/100</span>'


def render_star_rating(rating, reviews=None):
    """
    Renders a real buyer rating (out of 5) as filled/empty stars, with
    the review count alongside. Falls back gracefully if no rating exists
    for this product.
    """
    if rating is None:
        return ""

    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return ""

    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star

    stars_html = "★" * full_stars
    if half_star:
        stars_html += "⯨"  # half-star glyph; falls back gracefully if font lacks it
    stars_html += "☆" * empty_stars

    reviews_text = f" ({reviews:,} reviews)" if reviews else ""

    return f'''
    <span style="color:#f0c419; font-size:14px; letter-spacing:1px;">{stars_html}</span>
    <span style="color:#9a9a9a; font-size:12px; margin-left:6px;">{rating}/5{reviews_text}</span>
    '''

def get_product_link(product: dict) -> str:
    """
    Returns the real product link if SerpAPI provided one. If not (common
    for some aggregator/seller listings), falls back to a constructed
    Google Shopping search URL for the exact product title, so the link
    button is never missing entirely.
    """
    if product.get("link"):
        return product["link"]

    title = product.get("title", "")
    query = title.replace(" ", "+")
    return f"https://www.google.com/search?q={query}&tbm=shop"


# def render_recommendation_card(structured: dict):
def render_recommendation_card(structured: dict, render_key: str = ""):
    """
    Renders the full product recommendation card (top pick, alternatives,
    red flags, bottom line) from structured_recommendation data.
    Shared between the live turn and historical chat re-rendering so both
    look identical.
    """
    if not structured:
        return

    top = structured.get("top_pick", {})

    # --- Top Pick card ---
    st.markdown('<p style="font-size:12px; color:#9a9a9a; text-transform:uppercase; margin-bottom:6px;">Top pick</p>', unsafe_allow_html=True)

    if top.get("thumbnail"):

        st.markdown(f"""
        <div class="toppick-card">
            <img src="{top['thumbnail']}" class="toppick-img" style="width:100%; height:180px; object-fit:cover; display:block;" />
            <div class="toppick-body">
                <p style="font-weight:600; font-size:15px; margin:0 0 4px;">{top.get('title','')}</p>
                <div style="margin-bottom:6px;">{render_star_rating(top.get('rating'), top.get('reviews'))}</div>
                <p style="font-size:22px; font-weight:600; margin:8px 0 2px;">₹{top.get('price','N/A')}</p>
                <p style="font-size:12px; color:#9a9a9a; margin:0 0 10px;">via {top.get('source','')}</p>
                <p style="font-size:13px; color:#cfcfcf; margin:0 0 10px;">{top.get('why_it_wins','')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Graceful fallback if no thumbnail exists for this product
        st.markdown(f"""
        <div class="toppick-card">
            <div class="toppick-body">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
                    <p style="font-weight:600; font-size:15px; margin:0;">{top.get('title','')}</p>
                    {render_score_badge(top.get('confidence_score'))}
                </div>
                <p style="font-size:22px; font-weight:600; margin:8px 0 2px;">₹{top.get('price','N/A')}</p>
                <p style="font-size:12px; color:#9a9a9a; margin:0 0 10px;">via {top.get('source','')}</p>
                <p style="font-size:13px; color:#cfcfcf; margin:0 0 10px;">{top.get('why_it_wins','')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Spec match tags
    tags_html = "".join(f'<span class="spec-tag-good">✓ {t}</span>' for t in top.get("specs_matched", []))
    if top.get("specs_warning"):
        tags_html += f'<span class="spec-tag-warn">⚠ {top["specs_warning"]}</span>'
    if tags_html:
        st.markdown(f'<div style="margin-bottom:14px;">{tags_html}</div>', unsafe_allow_html=True)

    st.link_button("View deal ↗", get_product_link(top))

    # --- Alternatives ---
    alternatives = structured.get("alternatives", [])

    if alternatives:
        st.markdown('<p style="font-size:12px; color:#9a9a9a; text-transform:uppercase; margin:14px 0 6px;">Alternatives</p>', unsafe_allow_html=True)
        for idx, alt in enumerate(alternatives):
            img_html = f'<img src="{alt["thumbnail"]}" class="alt-img" style="width:52px; height:52px; object-fit:cover; border-radius:6px; flex-shrink:0;" />' if alt.get("thumbnail") else ""
            st.markdown(f"""
            <div class="alt-row">
                {img_html}
                <div style="flex:1; min-width:0;">
                    <p style="font-size:13px; font-weight:500; margin:0 0 2px;">{alt.get('title','')}</p>
                    <p style="font-size:12px; color:#9a9a9a; margin:0;">{alt.get('trade_off','')}</p>
                </div>
                <p style="font-size:15px; font-weight:600; margin:0;">₹{alt.get('price','N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        st.link_button(f"View {alt.get('title','')[:25]}... ↗", get_product_link(alt), key=f"alt_link_{render_key}_{idx}")

    if structured.get("filtered_out_note"):
        st.caption(structured["filtered_out_note"])

    # --- Red flags ---
    for flag in structured.get("red_flags", []):
        st.markdown(f'<div class="redflag-box"><span style="font-size:13px; color:#e74c3c;">⚠️ {flag}</span></div>', unsafe_allow_html=True)

    if structured.get("bottom_line"):
        st.markdown(f"""
        <div class="bottomline-box">
            <p style="font-size:13px; color:#cfcfcf; margin:0 0 16px 0; line-height:1.6;"><b style="color:#fff;">Bottom line —</b> {structured['bottom_line']}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Follow-up suggestion chips ---
    suggestions = structured.get("follow_up_suggestions", [])
    if suggestions:
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"chip_{render_key}_{i}"):
                    st.session_state.pending_chip_prompt = suggestion
                    st.rerun()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "log_session_started" not in st.session_state:
    start_new_conversation_log()
    st.session_state.log_session_started = True

if "agent" not in st.session_state:
    with st.spinner("Initializing Agentic Brain..."):
        st.session_state.agent = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your smart shopping negotiator. What are you looking to buy today? (e.g., 'I want to buy a coding laptop under 60k')"}
    ]

if "graph_state" not in st.session_state:
    st.session_state.graph_state = ShoppingState(
        user_query="",
        clarification_needed=False,
        search_params=None,
        raw_products=[],
        validated_deals=[],
        final_recommendation="",
        structured_recommendation=None,
        message_type="",
        errors=[],
        conversation_history=[],
        last_shown_deals=[],
        intent="NEW"
    )

if "pending_chip_prompt" not in st.session_state:
    st.session_state.pending_chip_prompt = None

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Architecture")
    st.markdown("""
    **An Agentic AI Project**
    * **Orchestration:** LangGraph
    * **Reasoning:** Groq (Llama 3.3 70B)
    * **Data:** SerpAPI (Google Shopping)
    """)

    if st.button("Reset Conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.graph_state = ShoppingState(
            user_query="",
            clarification_needed=False,
            search_params=None,
            raw_products=[],
            validated_deals=[],
            final_recommendation="",
            structured_recommendation=None,
            message_type="",
            errors=[],
            conversation_history=[],
            last_shown_deals=[],
            intent="NEW"
        )
        start_new_conversation_log()
        st.rerun()

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.title("🛒 Smart Shopping Negotiator")


# Render existing chat history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("structured"):
            render_recommendation_card(msg["structured"], render_key=f"history_{msg_idx}")
        else:
            st.markdown(msg["content"])


def run_agent_pipeline(user_input_query):
    st.session_state.graph_state["user_query"] = user_input_query
    st.session_state.graph_state["errors"] = []

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        full_response = ""

        with st.status("Agent is reasoning...", expanded=True) as status:
            try:
                final_state = None

                for mode, payload in st.session_state.agent.stream(
                    st.session_state.graph_state,
                    stream_mode=["updates", "messages"]
                ):
                    if mode == "updates":
                        for node_name, state in payload.items():
                            st.write(f"✅ Completed: **{node_name}**")
                            final_state = state

                    elif mode == "messages":
                        chunk, metadata = payload
                        # NOTE: "synthesize" no longer streams (structured
                        # output can't stream token-by-token) — only
                        # answer_followup still types out live now.
                        if metadata.get("langgraph_node") == "answer_followup":
                            full_response += chunk.content
                            text_placeholder.markdown(full_response + "▌")
                            time.sleep(0.02)

                if final_state:
                    st.session_state.graph_state = final_state

                status.update(label="Reasoning complete!", state="complete", expanded=False)

            except Exception as e:
                agent_logger.error(f"Streamlit UI encountered graph error: {str(e)}", exc_info=True)
                status.update(label="Execution failed", state="error", expanded=True)
                st.write(f"❌ Error: {str(e)}")
                return

        structured = st.session_state.graph_state.get("structured_recommendation")
        plain_recommendation = st.session_state.graph_state.get("final_recommendation", "")
        clarification_needed = st.session_state.graph_state.get("clarification_needed")

        if structured and not clarification_needed:
            # This was a fresh search -> render the full card
            text_placeholder.empty()
            render_recommendation_card(structured, render_key="live")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "structured": structured
            })
        elif plain_recommendation and not clarification_needed:
            # This was a follow-up answer -> plain streamed text, no card
            text_placeholder.markdown(plain_recommendation)
            st.session_state.messages.append({
                "role": "assistant",
                "content": plain_recommendation
            })
        else:
            text_placeholder.empty()

# ---------------------------------------------------------
# Dynamic Configuration Form
# ---------------------------------------------------------
if st.session_state.graph_state.get("clarification_needed"):
    last_query = st.session_state.graph_state.get("user_query", "")

    if "dynamic_spec_fields" not in st.session_state or st.session_state.get("spec_query_cache") != last_query:
        with st.spinner("Figuring out what specs matter for this..."):
            st.session_state.dynamic_spec_fields = generate_spec_options(last_query)
            st.session_state.spec_query_cache = last_query

    spec_fields = st.session_state.dynamic_spec_fields

    st.info("🔧 Customize your specifications below. Choose 'Skip / Don't Know' for any option you aren't sure about.")

    field_names = list(spec_fields.keys())
    selections = {}

    col1, col2 = st.columns(2)
    for i, field_name in enumerate(field_names):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            choice = st.selectbox(field_name, spec_fields[field_name], key=f"select_{i}_{field_name}")
            selections[field_name] = choice

            if choice == OTHER_LABEL:
                custom_value = st.text_input(
                    "Type your answer",
                    key=f"custom_{i}_{field_name}",
                    placeholder=f"e.g. your specific {field_name.lower()}"
                )
                if custom_value.strip():
                    selections[field_name] = custom_value.strip()

    free_text_value = st.text_input(
        "Anything else specific? (optional)",
        placeholder="e.g. lightweight, good battery life, budget-friendly...",
        key="free_text_extra"
    )

    if st.button("Apply Configuration & Search", key="apply_config_btn"):
        chosen_specs = [v for v in selections.values() if v not in (SKIP_LABEL, OTHER_LABEL)]
        if free_text_value:
            chosen_specs.append(free_text_value.strip())

        spec_string = ", ".join(chosen_specs)
        refined_query = (
            f"{st.session_state.graph_state['user_query']} with {spec_string}"
            if chosen_specs else st.session_state.graph_state['user_query']
        )

        st.session_state.messages.append({
            "role": "user",
            "content": f"Applied specs: {spec_string if chosen_specs else 'No preference (Skipped all)'}"
        })

        st.session_state.graph_state["clarification_needed"] = False
        st.session_state.pop("dynamic_spec_fields", None)
        st.session_state.pop("spec_query_cache", None)

        run_agent_pipeline(refined_query)
        st.rerun()

# ---------------------------------------------------------
# Handle a follow-up chip click (from render_recommendation_card)
# ---------------------------------------------------------
if st.session_state.pending_chip_prompt:
    chip_prompt = st.session_state.pending_chip_prompt
    st.session_state.pending_chip_prompt = None

    st.session_state.messages.append({"role": "user", "content": chip_prompt})
    with st.chat_message("user"):
        st.markdown(chip_prompt)

    run_agent_pipeline(chip_prompt)
    st.rerun()

# ---------------------------------------------------------
# Regular text chat box input
# ---------------------------------------------------------
if prompt := st.chat_input("Ask for a product or provide more details...", disabled=st.session_state.graph_state.get("clarification_needed", False)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    run_agent_pipeline(prompt)
    st.rerun()
