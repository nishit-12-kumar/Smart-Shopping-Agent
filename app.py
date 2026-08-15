# Streamlit chat UI for the Smart Shopping Negotiator.
import time

import streamlit as st

from src.shopping_agent.graph.builder import build_graph
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.utils.logger import agent_logger, start_new_conversation_log
from src.shopping_agent.utils.spec_options import generate_spec_options, SKIP_LABEL, OTHER_LABEL

from ui.styles import CHAT_CSS
from ui.components import render_recommendation_card

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Shopping Agent",
    page_icon="🛒",
    layout="wide"
)

st.markdown(CHAT_CSS, unsafe_allow_html=True)


def make_initial_state() -> ShoppingState:
    """
    Builds a fresh ShoppingState. Used both on first load and on
    'Reset Conversation' — previously this literal was duplicated in two
    places in app.py, which meant adding a new state field required
    remembering to update it in both spots.
    """
    return ShoppingState(
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
        search_history=[],
        search_turn_count=0,
        intent="NEW",
        serpapi_error_message=None
    )


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "log_session_started" not in st.session_state:
    start_new_conversation_log()
    st.session_state.log_session_started = True

if "agent" not in st.session_state:
    with st.spinner("Initializing Agentic Brain..."):
        # Stores the compiled LangGraph.
        st.session_state.agent = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your smart shopping agent. What are you looking to buy today? (e.g., 'I want to buy a coding laptop under 60k')"}
    ]

if "graph_state" not in st.session_state:
    st.session_state.graph_state = make_initial_state()

# Makes quick-action buttons work -> rerun the app with the chip prompt set, so the next turn uses it as the user query.
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

    if st.button("Reset Conversation", key="reset_conversation_btn_sidebar"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.graph_state = make_initial_state()
        start_new_conversation_log()
        st.rerun()

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.title("🛒 Smart Shopping Agent")


# Render existing chat history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("structured"):
            render_recommendation_card(
                msg["structured"],
                last_shown_deals=msg.get("last_shown_deals", []),
                render_key=f"history_{msg_idx}"
            )
        else:
            st.markdown(msg["content"])


def run_agent_pipeline(user_input_query):
    st.session_state.graph_state["user_query"] = user_input_query
    st.session_state.graph_state["errors"] = []
    st.session_state.graph_state["serpapi_error_message"] = None

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
            last_shown_deals = st.session_state.graph_state.get("last_shown_deals", [])
            text_placeholder.empty()
            render_recommendation_card(structured, last_shown_deals=last_shown_deals, render_key="live")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "structured": structured,
                # Snapshot (not a live reference) so a later search overwriting
                # state["last_shown_deals"] can't retroactively change what this
                # older turn's score breakdown / pricing reasons show.
                "last_shown_deals": list(last_shown_deals)
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




