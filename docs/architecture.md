# Smart Shopping Agent — System Architecture

## Overview

The Smart Shopping Agent is a multi-agent AI pipeline that processes natural language shopping requests into structured product recommendations. Instead of one large LLM call, the system is broken into specialized nodes — each doing one focused job — orchestrated by LangGraph as a stateful graph.

---

## High-Level Flow

```
User Message
     │
     ▼
┌─────────────────────────┐
│  classify_message_type  │  ← Is this chitchat or a real shopping request?
└─────────────────────────┘
         │              │
    CHITCHAT          SHOPPING
         │              │
         ▼              ▼
        END      ┌─────────────────┐
                 │ classify_intent │  ← Is this a follow-up or a new search?
                 └─────────────────┘
                      │         │
                 FOLLOW_UP      NEW
                      │         │
                      ▼         ▼
             ┌──────────────┐  ┌─────────────┐
             │answer_followup│  │ parse_query │  ← Does the query have enough specs?
             └──────────────┘  └─────────────┘
                      │              │         │
                     END    needs_clarification  has_specs
                                    │              │
                                   END        ┌──────────────────┐
                              (UI shows        │ search_products  │  ← SerpAPI call
                               spec form)      └──────────────────┘
                                                        │
                                               ┌────────────────┐
                                               │ validate_deals │  ← Confidence scoring
                                               └────────────────┘
                                                        │
                                               ┌────────────────┐
                                               │ price_validity │  ← Fake discount detection
                                               └────────────────┘
                                                        │
                                               ┌───────────────┐
                                               │   synthesize  │  ← Build final recommendation
                                               └───────────────┘
                                                        │
                                                       END
```

---

## Node Descriptions

### `classify_message_type`
**Entry point of the graph.**

Calls Groq with a simple classification prompt: is this message CHITCHAT (greeting, thanks, "what can you do") or SHOPPING (any genuine product-buying intent)? If CHITCHAT, the LLM also generates a short conversational reply and the graph terminates immediately — none of the expensive downstream nodes (search, validate, synthesize) are ever called.

- Input: `user_query`
- Output: `message_type` ("CHITCHAT" or "SHOPPING"), `final_recommendation` (if CHITCHAT)
- Routes to: `END` (CHITCHAT) or `classify_intent` (SHOPPING)

---

### `classify_intent`
**Memory gate — decides if the user is following up or starting fresh.**

Checks `last_shown_deals` in state. If products were shown in a previous turn, calls Groq to classify the new message as either a FOLLOW_UP (referring to something already shown) or a NEW search. If no previous deals exist, always classifies as NEW without an LLM call (saves quota).

- Input: `user_query`, `last_shown_deals`
- Output: `intent` ("NEW" or "FOLLOW_UP"), `search_params.referenced_product`
- Routes to: `answer_followup` (FOLLOW_UP) or `parse_query` (NEW)

---

### `answer_followup`
**Answers questions about previously shown products without re-searching.**

Uses the products stored in `last_shown_deals` to answer follow-up questions directly. Calls Groq with the stored product data and the user's question. Streams tokens for a live typewriter effect in the UI. Skips search, validate, price-check, and synthesize entirely — saving 3-4 API calls.

- Input: `user_query`, `last_shown_deals`, `search_params.referenced_product`
- Output: `final_recommendation`
- Routes to: `END`

---

### `parse_query`
**Decides if we have enough information to search.**

Checks the query for numeric specs, spec-related keywords (GB, ton, star, inch), and word count. If the query is too vague (e.g., "I want to buy a laptop"), sets `clarification_needed = True` and terminates — the Streamlit UI then renders a dynamic clarification form. If the query already has enough information, proceeds to search.

- Input: `user_query`
- Output: `clarification_needed` (bool), `search_params`
- Routes to: `END` (clarification needed) or `search_products` (has specs)

---

### `search_products`
**Fetches real, live product data from Google Shopping.**

Calls SerpAPI's `google_shopping` engine with the user's refined query (original query + chosen specs from the clarification form). Returns up to 5 products with title, price, source, rating, review count, thumbnail image URL, and product link (parsed from SerpAPI's `product_link` field).

If SerpAPI itself fails — quota exhausted, invalid key, or a connection error — `SerpApiClient` raises a `SerpAPIError` rather than silently returning empty or fake data. This node catches it, classifies the message (quota-related keywords vs. a generic failure), and stores a clear, user-facing explanation in `serpapi_error_message`. `raw_products` is set to `[]` either way, so the rest of the pipeline (`validate_deals`, `price_validity`) skips gracefully, and `synthesize` surfaces the explanation directly instead of a generic "no products found" message.

- Input: `search_params` (query string)
- Output: `raw_products` (list of product dicts), `serpapi_error_message` (set only on failure)
- Routes to: `validate_deals`

---

### `validate_deals`
**Scores every product for how well it matches the user's actual request.**

Uses Groq's `with_structured_output()` (Pydantic schema) to evaluate all fetched products against the original user query. Returns a confidence score (0-100) and one-line reasoning for each product. Also flags products with suspiciously high ratings but very few reviews (e.g., 4.8★ with only 8 reviews) using a plain Python heuristic — no LLM call needed for this.

- Input: `raw_products`, `user_query`
- Output: `validated_deals` (products + confidence_score + reasoning + review_flag)
- Routes to: `price_validity`

---

### `price_validity`
**Detects fake discounts and artificially inflated pricing.**

Sends all validated products to Groq with a focused prompt: "Does this price look artificially inflated compared to standard market rates for this category?" Returns a boolean `is_suspicious` and a short `analysis_reasoning` per product. Products flagged here are surfaced as red flag warnings in the final recommendation card.

- Input: `validated_deals`
- Output: `validated_deals` (with `is_suspicious_pricing` and `pricing_analysis` added per product)
- Routes to: `synthesize`

---

### `synthesize`
**Builds the final structured recommendation.**

This node deliberately does NOT ask the LLM to pick products or generate prices. Instead, it selects the top pick and alternatives deterministically in Python code (sorted by `confidence_score`), then asks Groq only for the qualitative reasoning fields via a Pydantic schema: `why_it_wins`, `specs_matched`, `specs_warning`, `alternative_trade_offs`, `red_flags`, `bottom_line`, and `follow_up_suggestions`. The final `structured_recommendation` dict combines real product data (prices, titles, images) with LLM-generated reasoning text — guaranteeing the UI can never show a hallucinated price.

If there are no deals to synthesize, this node checks `serpapi_error_message` first: if set (SerpAPI failed upstream), that explanation becomes `final_recommendation` directly, no LLM call made. Otherwise it falls back to a generic "try adjusting your budget or brand preferences" message — this is the genuinely-no-matching-products case, distinct from a SerpAPI outage.

Also saves `last_shown_deals` and `conversation_history` to state for multi-turn memory.

- Input: `validated_deals`, `user_query`, `serpapi_error_message`
- Output: `structured_recommendation`, `final_recommendation` (short text for follow-up context), `last_shown_deals`, `conversation_history`
- Routes to: `END`

---

## State Object

All nodes read from and write to a single shared `ShoppingState` TypedDict that flows through the entire graph:

```python
class ShoppingState(TypedDict):
    user_query: str                              # raw user input
    clarification_needed: bool                   # triggers spec form in UI
    search_params: Optional[Dict[str, Any]]      # refined query + referenced product
    raw_products: List[Dict[str, Any]]           # SerpAPI results
    serpapi_error_message: Optional[str]         # set when SerpAPI itself fails (quota/outage); lets synthesize show a clear reason instead of "no products found"
    validated_deals: List[Dict[str, Any]]        # products + scores + flags
    message_type: str                            # "CHITCHAT" or "SHOPPING"
    final_recommendation: str                    # plain text (for follow-up context)
    structured_recommendation: Optional[Dict]    # card data for UI rendering
    errors: List[str]                            # error log per turn
    conversation_history: List[Dict[str, Any]]   # past turn summaries
    last_shown_deals: List[Dict[str, Any]]       # products from most recent search
    intent: str                                  # "NEW" or "FOLLOW_UP"
```

---

## Error Handling: SerpAPI Failures

`SerpApiClient.search_google_shopping()` never silently returns fake or empty-looking data on failure — it raises a `SerpAPIError` (defined in `utils/exceptions.py`), which propagates up to `search_products_node`. That node classifies the failure into one of two user-facing messages:

- **Quota/usage related** (message contains phrases like "run out of searches", "exceeded", "quota", "usage limit") → a message telling the user the search quota is temporarily used up.
- **Anything else** (network failure, invalid key, etc.) → a generic "SerpAPI is currently unavailable, try again shortly" message.

This message is stored in `serpapi_error_message` and surfaces directly through `synthesize_node` as the final response, instead of being confused with a normal "no products matched your criteria" outcome.

**Known rough edge:** SerpAPI also uses the same `"error"` field for a completely different situation — a query that legitimately has zero shopping results (e.g. an overly specific search). Today this is indistinguishable from a real outage and shows the same "SerpAPI unavailable" message, even though retrying won't help — broadening the query would. This is a known limitation, not yet special-cased (see `docs/decisions.md`).

---

## Technology Stack

| Layer | Tool | Purpose |
|---|---|---|
| Orchestration | LangGraph | Defines the node graph, manages state, conditional routing |
| LLM | Groq (Llama 3.3 70B) | All reasoning — classification, scoring, synthesis |
| Structured output | LangChain + Pydantic | Ensures LLM responses are clean, typed, never freeform |
| Product data | SerpAPI (Google Shopping) | Live product listings, prices, ratings, images |
| UI | Streamlit | Chat interface, spec form, product card rendering |
| Secrets | python-dotenv | API key management via `.env` file |
| Logging | Python `logging` | Per-conversation log files in `logs/` |

---

## UI Architecture

The Streamlit UI (`app.py`) operates in two distinct rendering modes:

**1. Chat history loop (top of app.py)**
Runs on every rerun. Re-renders all past messages. For messages with `structured` data, calls `render_recommendation_card()` with a unique `render_key` based on message index to avoid Streamlit duplicate-key errors.

**2. `run_agent_pipeline()` (live turn)**
Called when the user sends a new message. Streams the graph via `stream_mode=["updates", "messages"]` — "updates" mode shows node-progress checkmarks in the status box; "messages" mode intercepts live LLM tokens from `answer_followup` for the typewriter effect. After the graph finishes, renders the product card directly.

**3. Clarification form**
Rendered outside the pipeline when `clarification_needed = True`. Calls `generate_spec_options()` which asks Groq to generate product-category-specific clarifying questions and dropdown options dynamically. On submit, rebuilds a refined query and re-invokes the pipeline.