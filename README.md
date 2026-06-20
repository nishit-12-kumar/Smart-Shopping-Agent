# 🛒 Smart Shopping Negotiator

An **agentic AI shopping assistant** that turns natural-language product requests into validated, structured buying recommendations for the Indian market (₹). Built as a multi-node **LangGraph** pipeline powered by **Groq (Llama 3.3 70B)** for reasoning and **SerpAPI (Google Shopping)** for live product data, with a **Streamlit** chat UI.

Instead of one big LLM call, the system is broken into focused, single-responsibility nodes — classification, search, deal validation, fake-discount detection, and synthesis — orchestrated as a stateful graph with conditional routing.

---

## 🔗 Live Demo

**[👉 Try it live](https://your-app-name.streamlit.app)**

---

## ✨ Features

- **Conversational shopping chat** — ask for a product in plain English (e.g. *"I want a coding laptop under 60k"*).
- **Chitchat vs. shopping detection** — greetings and small talk are answered directly without touching the expensive search/validation pipeline.
- **Multi-turn memory** — follow-up questions ("what about the second one?") are answered from previously shown products without a new search.
- **Dynamic clarification form** — if a query is too vague, the agent generates category-specific spec questions (RAM, storage, star rating, capacity, etc.) *on the fly* via the LLM, not from a hardcoded list — so it works for any product category.
- **Live product search** — real listings from Google Shopping via SerpAPI (title, price, rating, reviews, thumbnail, source, link), with a local mock fallback if the API fails or hits quota.
- **Confidence scoring** — every product is scored 0–100 against the user's actual request, with a one-line justification.
- **Fake-discount / suspicious-pricing detection** — flags listings with inflated pricing or a suspiciously high rating with very few reviews.
- **Deterministic, hallucination-safe recommendations** — the top pick and alternatives are selected in **Python** by confidence score, not by the LLM. The LLM only supplies reasoning text (why it wins, trade-offs, red flags, bottom line) — so prices and titles shown to the user always come from real data.
- **Token-by-token streaming** — live typewriter effect for follow-up answers via LangGraph's `stream_mode=["updates", "messages"]`.
- **Per-conversation logging** — each session gets its own timestamped log file for easy debugging.

---

## 🏗️ Architecture

```
User Message
     │
     ▼
classify_message_type   ← chitchat or shopping?
     │                   │
  CHITCHAT           SHOPPING
     │                   │
    END           classify_intent   ← follow-up or new search?
                     │         │
               FOLLOW_UP      NEW
                     │         │
             answer_followup  parse_query   ← enough specs to search?
                     │           │        │
                    END   needs_clarification   has_specs
                                  │              │
                                 END        search_products   ← SerpAPI call
                            (UI shows              │
                             spec form)      validate_deals   ← confidence scoring
                                                    │
                                             price_validity   ← fake discount detection
                                                    │
                                               synthesize      ← build final recommendation
                                                    │
                                                   END
```

| Node | Job |
|---|---|
| `classify_message_type` | Entry point — CHITCHAT vs SHOPPING; replies directly and ends the graph for chitchat |
| `classify_intent` | Follow-up vs new search, using `last_shown_deals` as memory |
| `answer_followup` | Answers questions about already-shown products without re-searching (streamed) |
| `parse_query` | Decides if the query has enough specs to search, or needs the clarification form |
| `search_products` | Calls SerpAPI (`google_shopping` engine), with mock-data fallback |
| `validate_deals` | Scores each product 0–100 against the user's request; flags rating/review mismatches |
| `price_validity` | Detects artificially inflated / suspicious pricing |
| `synthesize` | Deterministically (Python) picks top pick + alternatives; asks the LLM only for reasoning text; saves memory for follow-ups |

See [`docs/architecture.md`](docs/architecture.md) for the full node-by-node breakdown and state schema, and [`docs/decisions.md`](docs/decisions.md) for the engineering rationale behind every major design choice (why LangGraph, why Groq, why SerpAPI, why deterministic product selection, known limitations, and what would change in production).

---

## 🧰 Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Orchestration | LangGraph | State graph, conditional routing |
| LLM | Groq (Llama 3.3 70B) | Classification, scoring, synthesis |
| Structured output | LangChain + Pydantic | Typed, hallucination-resistant LLM output |
| Product data | SerpAPI (Google Shopping) | Live listings, prices, ratings, images |
| UI | Streamlit | Chat interface, spec form, recommendation cards |
| Secrets | python-dotenv | `.env`-based API key management |
| Logging | Python `logging` | Per-conversation log files (`logs/`) |

---

## 📁 Project Structure

```
Demo_Shopping/
├── app.py                          # Streamlit UI — chat, spec form, card rendering
├── requirements.txt
├── setup.py                        # Editable install (adds src/ to PYTHONPATH)
├── docs/
│   ├── architecture.md             # Full system architecture
│   └── decisions.md                # Engineering decisions & trade-offs
├── logs/                           # Per-conversation timestamped log files
└── src/shopping_agent/
    ├── config.py                   # Loads GROQ_API_KEY / SERPAPI_API_KEY from .env
    ├── graph/
    │   ├── builder.py              # Compiles the LangGraph state machine
    │   ├── edges.py                # Conditional routing logic
    │   └── state.py                # ShoppingState (shared TypedDict)
    ├── nodes/
    │   ├── classify_message_type.py
    │   ├── classify_intent.py
    │   ├── answer_followup.py
    │   ├── parse_query.py
    │   ├── search_products.py
    │   ├── validate_deals.py
    │   ├── price_validity.py
    │   └── synthesize.py
    ├── services/
    │   ├── groq_client.py          # Groq LLM client wrapper
    │   └── serpapi_client.py       # SerpAPI client wrapper
    └── utils/
        ├── logger.py                # Per-conversation logging setup
        ├── spec_options.py          # Dynamic clarification-form generation
        └── exceptions.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)
- A [SerpAPI key](https://serpapi.com) (free tier: 250 searches/month)

### Installation

```bash
git clone <your-repo-url>
cd Demo_Shopping

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -e .                # installs deps + adds src/ to PYTHONPATH
```

### Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

### Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and start chatting — e.g. *"I want to buy a coding laptop under 60k"*.

---

## 💬 Example Flow

1. **User:** "I need a good washing machine"
2. **Agent:** Not enough specs → shows a dynamically generated clarification form (capacity, type, star rating, brand preference…)
3. **User:** Fills in specs, submits
4. **Agent:** Searches live listings → validates deals → checks pricing → returns a **Top Pick** card with price, rating, matched specs, alternatives, red flags, and a bottom-line summary
5. **User:** "Is the second one worth it?"
6. **Agent:** Recognizes this as a follow-up → answers directly from the already-fetched products (streamed token-by-token), no new search needed

---

## 📸 Screenshots

> Add your own screenshots/GIFs here — this is what most people look at first on GitHub. Save images to a `docs/images/` (or `assets/`) folder in the repo, then reference them below.

| Chat interface | Recommendation card |
|---|---|
| ![Chat interface](docs/images/chat-interface.png) | ![Recommendation card](docs/images/recommendation-card.png) |

| Clarification form | Follow-up / streaming answer |
|---|---|
| ![Clarification form](docs/images/clarification-form.png) | ![Follow-up answer](docs/images/followup-streaming.png) |

**Suggested shots to capture:**
- Initial chat screen with the greeting message
- A vague query triggering the dynamic clarification form
- A completed Top Pick recommendation card (with alternatives, red flags, bottom line visible)
- A follow-up question being answered with the live typewriter/streaming effect
- The sidebar showing the system architecture panel

**How to add them:**
1. Create the folder: `mkdir -p docs/images`
2. Drop your `.png`/`.gif` files in there
3. Reference them in this README using `![alt text](docs/images/your-file.png)`
4. For a quick demo GIF, tools like [ScreenToGif](https://www.screentogif.com/) (Windows) or `Cmd+Shift+5` (macOS) work well — keep it under ~10MB so it renders smoothly on GitHub.

---

## ⚠️ Known Limitations

- Price history shown in the UI is an estimated trend, not real historical data (would require a paid service like Keepa).
- Review text isn't fetched — only rating and review count (SerpAPI's `google_shopping` engine doesn't return snippets).
- Follow-up vs. new-search classification can misfire on genuinely ambiguous phrasing.
- No user authentication — conversation history lives only in `st.session_state` and resets on page refresh.

Full list and reasoning in [`docs/decisions.md`](docs/decisions.md#10-known-limitations-and-honest-trade-offs).

---

## 🏭 What Would Change in Production

| Demo (current) | Production |
|---|---|
| In-memory Streamlit session state | Redis / PostgreSQL for persistent multi-user state |
| Flat per-conversation log files | Structured JSON logs → Datadog/CloudWatch |
| SerpAPI free tier (100 searches/mo) | Paid SerpAPI plan or direct retailer API |
| Local Streamlit server | Streamlit Cloud / Railway / GCP Cloud Run |
| `.env` file | AWS/GCP Secrets Manager |
| Groq free tier | Paid Groq or Anthropic/OpenAI based on cost/quality |
| No auth | Auth0 / Supabase with persistent search history |

---

## 👤 Author

**Nishit Kumar**