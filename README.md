# 🛒 Smart Shopping Negotiator

An **agentic AI-powered shopping assistant** that thinks before it searches. It classifies your intent, asks the right clarifying questions, fetches live product data, scores every deal, detects fake discounts, and delivers a polished product recommendation card — all inside a conversational chat interface.

Built with **LangGraph**, **Groq (Llama 3.3 70B)**, **SerpAPI**, and **Streamlit**.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Pipeline-orange)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![SerpAPI](https://img.shields.io/badge/Data-SerpAPI_Google_Shopping-green)

---

---

## 🔗 Live Demo

**[👉 Try it live](https://smart-shopping-agent-negotiator.streamlit.app/)**

---

## 📌 What This Project Does

Most shopping chatbots are just a single prompt → single answer. This system is a **7-node multi-agent pipeline** where each node has one focused job:

| What the user experiences | What's happening under the hood |
|---|---|
| Types "hi" or "thanks" | `classify_message_type` detects chitchat, replies directly, skips all search nodes |
| Types "I want to buy a laptop" | `parse_query` detects vague query, triggers dynamic clarification form |
| Fills in processor, RAM, budget | Refined query built from selections |
| Sees real products appear | `search_products` calls SerpAPI Google Shopping with live data |
| Each product has a confidence score | `validate_deals` evaluates product-request fit (0-100) via Groq |
| Suspicious product flagged | `price_validity` detects fake/inflated discounts |
| Gets a polished recommendation card | `synthesize` builds structured output — prices never hallucinated |
| Asks "tell me more about the first one" | `classify_intent` detects follow-up, `answer_followup` replies from memory |

---

## 🧠 Agent Architecture

```
User Message
     │
     ▼
classify_message_type ──CHITCHAT──► END (direct conversational reply)
     │
   SHOPPING
     │
     ▼
classify_intent ──FOLLOW_UP──► answer_followup ──► END
     │
    NEW
     │
     ▼
parse_query ──needs clarification──► END (Streamlit spec form renders)
     │
  has specs
     │
     ▼
search_products (SerpAPI Google Shopping)
     │
     ▼
validate_deals (confidence scoring 0-100 per product)
     │
     ▼
price_validity (fake discount detection)
     │
     ▼
synthesize (structured card — LLM reasons, never invents prices)
     │
     ▼
    END
```

---

## ✨ Features

### Agentic Intelligence
- **Intent classification** — distinguishes chitchat ("hi", "thanks") from genuine shopping requests using Groq, not keyword lists
- **Follow-up memory** — follow-up questions answered directly from previous results, no re-searching
- **Dynamic clarification form** — Groq generates product-specific questions (AC needs room size and tonnage, laptop needs RAM and processor, shoes need size and type) — works for any product, not a hardcoded form
- **Vague query detection** — "I want to buy a laptop" triggers clarification; "laptop with 16GB RAM under ₹60,000" proceeds directly to search

### Data & Validation
- **Live product data** — real products and prices from Google Shopping via SerpAPI
- **Confidence scoring** — every product rated 0-100 for fit against the user's actual request (not just price sorting)
- **Fake discount detection** — flags products with artificially inflated "original prices"
- **Review trust check** — warns about products with high ratings but suspiciously few reviews
- **Hallucination-proof output** — prices, titles, and images always come from real SerpAPI data; the LLM only generates reasoning text

### UI & Experience
- **Product image cards** — thumbnail photos from SerpAPI shown in the recommendation card
- **Star ratings** — real buyer ratings (e.g. ★★★★☆ 4.3/5 from 2,400 reviews) rendered visually
- **Color-coded confidence badges** — green (85+), yellow (60-84), red (below 60)
- **Clickable "View deal" links** — direct links to the seller's page; falls back to a Google Shopping search if SerpAPI doesn't return a direct URL
- **Spec match tags** — ✓ tags showing which of the user's preferences this product satisfies
- **Red flag banners** — suspicious pricing highlighted in a distinct warning block
- **Follow-up suggestion chips** — two tappable buttons after every recommendation for natural next questions
- **Live agent progress** — "✅ Completed: validate_deals" checkmarks appear as each node finishes
- **Per-conversation logging** — each chat session gets its own timestamped log file in `logs/`

---

## 🗂️ Project Structure

```
smart-shopping-agent/
│
├── app.py                              # Streamlit UI — chat interface, card rendering
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   └── shopping_agent/
│       ├── config.py                   # Loads .env, validates API keys present
│       │
│       ├── graph/
│       │   ├── state.py                # ShoppingState TypedDict — shared across all nodes
│       │   ├── builder.py              # Compiles and wires the LangGraph graph
│       │   └── edges.py                # Conditional routing functions
│       │
│       ├── nodes/
│       │   ├── classify_message_type.py  # Chitchat vs shopping intent
│       │   ├── classify_intent.py        # Follow-up vs new search detection
│       │   ├── answer_followup.py        # Answers from memory, skips search entirely
│       │   ├── parse_query.py            # Spec-sufficiency check, triggers clarification
│       │   ├── search_products.py        # SerpAPI call, quota-error handling
│       │   ├── validate_deals.py         # Confidence scoring via Groq + review-trust flag
│       │   ├── price_validity.py         # Fake discount detection via Groq
│       │   └── synthesize.py             # Structured card output — deterministic product selection
│       │
│       ├── services/
│       │   ├── groq_client.py            # Groq LLM client (Llama 3.3 70B, temp=0)
│       │   └── serpapi_client.py         # SerpAPI wrapper, product_link extraction
│       │
│       └── utils/
│           ├── logger.py                 # Per-conversation timestamped log files
│           ├── spec_options.py           # LLM-generated clarification questions
│           └── exceptions.py            # SerpAPIError, ShoppingAgentBaseException
│
├── logs/                               # Runtime conversation logs (gitignored)
│
└── docs/
    ├── architecture.md                 # Full node graph, state schema, routing diagram
    └── decisions.md                    # Engineering decisions and trade-offs
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) — free tier available
- A [SerpAPI key](https://serpapi.com) — free tier: 250 searches/month

### 1. Clone the repository
```bash
git clone https://github.com/your-username/smart-shopping-agent.git
cd smart-shopping-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

Open `.env` and add your keys:
```
GROQ_API_KEY=your_groq_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 💬 Example Conversations

**Fresh product search:**
```
You:    I want to buy a gaming laptop
Agent:  [Clarification form: Budget? / Primary use? / RAM? / Brand?]
You:    ₹80,000-₹1,00,000 / Gaming / 16GB / No preference
Agent:  [Product card: top pick with image, rating, price, spec tags, buy link + alternatives]
```

**Follow-up question:**
```
You:    Tell me more about the first option
Agent:  [Answers directly from previous results — no new API calls triggered]
```

**Chitchat:**
```
You:    Hey, what can you do?
Agent:  I can help you find the best deals on laptops, phones, ACs, and pretty 
        much anything you want to shop for. Just tell me what you're looking for!
```
---

## 🛠️ Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Core language |
| LangGraph | ≥0.0.20 | Multi-agent orchestration, conditional routing |
| LangChain-Groq | ≥0.1.0 | Groq LLM client integration |
| LangChain-Core | ≥0.1.50 | Prompts, Pydantic structured output |
| Groq / Llama 3.3 70B | — | All LLM reasoning (classification, scoring, synthesis) |
| SerpAPI | ≥2.4.2 | Live Google Shopping product data |
| Streamlit | ≥1.32.0 | Chat UI, card rendering, clarification form |
| Pydantic | ≥2.0.0 | Typed structured output schemas for LLM calls |
| python-dotenv | ≥1.0.1 | Environment variable / secret management |

---

## 🔑 Environment Variables

| Variable | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | ✅ | [console.groq.com](https://console.groq.com) |
| `SERPAPI_API_KEY` | ✅ | [serpapi.com](https://serpapi.com) |

---

## 🔐 Security Notes

- API keys are loaded via `.env` — never committed to git (`.gitignore` covers `.env`)
- `.env.example` is committed as a template showing required keys without real values
- Input queries are passed directly to Groq prompts — no SQL or shell execution, so injection risk is low; standard LLM prompt-injection awareness applies

---

## 📊 Production Considerations

This project is designed as a portfolio demo. For a production deployment, the following changes would apply:

| Area | Current | Production equivalent |
|---|---|---|
| State persistence | In-memory `ShoppingState` per session | Redis or PostgreSQL for multi-user persistence |
| Authentication | None | Auth0 / Supabase for user accounts |
| Search volume | SerpAPI free tier (100/month) | Paid SerpAPI plan or direct retailer API |
| Logging | Per-conversation flat files | Structured JSON → Datadog / CloudWatch |
| Deployment | Local Streamlit | Streamlit Cloud / Railway / GCP Cloud Run |
| Secrets | `.env` file | AWS Secrets Manager / GCP Secret Manager |

See [`docs/decisions.md`](docs/decisions.md) for the full reasoning behind every major design choice.

---

## 📁 Documentation

| File | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Full node graph, state object schema, routing diagram, UI architecture |
| [`docs/decisions.md`](docs/decisions.md) | 10 engineering decisions with reasoning, trade-offs, and known limitations |

---

## 🧪 Running Tests

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Dead code check
vulture app.py src/

# Lint
ruff check src/
```

---

## 👤 Author

Built by **Nishit Kumar**

- [LinkedIn](https://www.linkedin.com/in/nishit-12-kumar/)

---




















