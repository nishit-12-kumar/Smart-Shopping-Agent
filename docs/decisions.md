# Smart Shopping Agent — Engineering Decisions & Trade-offs

This document records the key engineering decisions made during development,
the reasoning behind each choice, and what would change in a production system.
This is the kind of thinking that separates a demo from a real product.

---

## 1. Why LangGraph instead of a simple chain?

**Decision:** Use LangGraph as the orchestration layer instead of a linear LangChain chain.

**Reasoning:**
A linear chain (A → B → C → always) can't handle the conditional flows this system needs: chitchat vs shopping, follow-up vs new search, clarification vs direct search. LangGraph's conditional edges let each node inspect the current state and route to different next nodes — this is what makes the system genuinely "agentic" rather than just a sophisticated prompt wrapper.

**Trade-off:** LangGraph adds complexity (state management, node definitions, edge routing) that a simple chain doesn't need. Worth it here because the branching logic is real and not trivial to fake with if/else in a single function.

**Production note:** In production, LangGraph's `interrupt()` mechanism would be used for the clarification step (proper human-in-the-loop pause) instead of routing to END and relying on the Streamlit UI to resume.

---

## 2. Why Groq instead of OpenAI?

**Decision:** Use Groq (Llama 3.3 70B) as the LLM provider.

**Reasoning:**
- Free tier with generous limits — important for a portfolio project that may be demoed repeatedly
- Inference speed is significantly faster than OpenAI's API (Groq uses custom hardware accelerators), which matters for live demos where waiting 10 seconds for a response looks bad
- Llama 3.3 70B is capable enough for all tasks in this pipeline — classification, structured output, reasoning

**Trade-off:** Groq's free tier has rate limits. For heavy usage or production, switching to OpenAI/Anthropic would be straightforward since LangChain abstracts the provider — just change the client import and model name.

**Production note:** The `GroqClient` class is already isolated in `services/groq_client.py`, so swapping providers requires changing only one file.

---

## 3. Why SerpAPI instead of scraping?

**Decision:** Use SerpAPI's `google_shopping` engine instead of scraping Amazon/Flipkart directly.

**Reasoning:**
- Amazon and Flipkart have aggressive anti-scraping measures (CAPTCHAs, IP blocks, legal ToS restrictions)
- Scraping breaks frequently when sites update their HTML structure
- SerpAPI returns clean, structured JSON — no parsing fragility, no browser automation needed
- Already handles geolocation (`gl=in` for India) and returns price, rating, review count, thumbnail, and source in one call

**Trade-off:** SerpAPI's free tier is limited to 100 searches/month. For a portfolio demo this is fine; for production, either a paid SerpAPI plan or a direct Flipkart/Amazon API integration would be needed.

**Production note:** The `SerpApiClient` class is isolated in `services/serpapi_client.py`. Switching data sources only requires modifying this one file — all downstream nodes consume the same product dict schema.

---

## 4. Why structured Pydantic output for synthesis — and why not for everything?

**Decision:** Use `with_structured_output()` (Pydantic schema) in `validate_deals`, `price_validity`, and `synthesize`, but not in `classify_message_type`, `classify_intent`, or `answer_followup`.

**Reasoning:**
Pydantic structured output guarantees the LLM returns a typed, predictable object instead of freeform text that needs regex parsing. This is critical for nodes that produce data used downstream in logic (confidence scores, suspicious flags, recommendation fields).

For classification nodes, a simple JSON string parsed with `re.search(r"\{.*\}", ...)` is sufficient and avoids the overhead of tool-calling (which structured output uses under the hood). For `answer_followup`, freeform prose is the right output format — it's meant to feel conversational, not structured.

**Trade-off:** Structured output is incompatible with token-by-token streaming (tool-calling returns the full object at once). This is why `synthesize` lost the typewriter effect — a deliberate trade-off: data reliability over visual polish for the most important node.

---

## 5. Why are top pick and alternatives selected in Python, not by the LLM?

**Decision:** `synthesize_node` sorts products by `confidence_score` in Python code to select the top pick and alternatives, then asks the LLM only for the reasoning text.

**Reasoning:**
LLMs can hallucinate. If we ask the LLM "which of these is best?", there's a non-zero chance it invents a price, misattributes a spec, or picks a product that doesn't match the data. By selecting products deterministically in code and asking the LLM only for qualitative reasoning (why it wins, trade-offs, bottom line), we guarantee:
- Prices shown to the user always come from real SerpAPI data
- Product titles always come from real SerpAPI data
- The LLM can only affect the text reasoning, not the factual claims

**Trade-off:** Purely confidence-score-based sorting doesn't capture every nuance a human expert might consider. But it's reliable, explainable, and safe.

---

## 6. Why does the clarification form use a separate `generate_spec_options()` call instead of a hardcoded form?

**Decision:** The clarification form fields and dropdown options are generated dynamically by Groq per product category, instead of a hardcoded dictionary per category (laptop, AC, phone, etc.).

**Reasoning:**
A hardcoded dictionary only covers categories we anticipated. If a user searches for "blender", "gaming chair", "running shoes", or "air purifier", a static dictionary would either show a generic fallback form or nothing useful. The LLM-generated form works for literally any product because it reasons about what specs matter for that specific category.

**Trade-off:** One extra Groq API call per clarification round. Mitigated by caching the generated fields in `st.session_state` so the same product category doesn't trigger a new call on every Streamlit rerun.

---

## 7. Why is there a multi-stage classification (chitchat → intent → parse) instead of one big classifier?

**Decision:** Three separate classification nodes each do one job, rather than a single node that handles all cases.

**Reasoning:**
Single Responsibility Principle applied to LLM nodes. Each node has a narrow, well-defined classification task:
- `classify_message_type`: "Is this even a shopping message?"
- `classify_intent`: "Is this a follow-up or new search?" (only runs if shopping intent confirmed)
- `parse_query`: "Do we have enough specs?" (only runs if new search confirmed)

Combining all three into one prompt would make the prompt complex, harder to debug (which classification failed?), and harder to improve independently. Keeping them separate means we can tune the chitchat classifier without touching the follow-up detection logic.

**Trade-off:** Three LLM calls instead of one for the routing layer. In practice, chitchat and follow-up paths terminate early, so most real shopping sessions only pay for one extra classification call (classify_message_type) before hitting the existing nodes.

---

## 8. Why use per-conversation log files instead of one global log?

**Decision:** Each conversation session gets its own timestamped log file (e.g., `conversation_2026-06-29_11-05-12_a91f3e2c.log`).

**Reasoning:**
A single growing log file makes debugging hard — finding the relevant lines for one conversation means scrolling through interleaved output from multiple sessions. Per-conversation files let you open the log for "the session where the AC search broke" directly, without filtering.

**Trade-off:** Many small files instead of one big one. For a demo/portfolio project, this is fine. For a multi-user production system, a proper logging service (Datadog, CloudWatch, structured JSON logs) would be used instead.

---

## 9. What would change in production?

| Current (demo) | Production equivalent |
|---|---|
| SQLite-style in-memory state | Redis or PostgreSQL for persistent multi-user state |
| Per-conversation flat log files | Structured JSON logs → Datadog/CloudWatch |
| SerpAPI free tier (100 searches/month) | Paid SerpAPI plan or direct retailer API |
| Streamlit single-user local server | Deployed on Streamlit Cloud, Railway, or GCP Cloud Run |
| `.env` file for secrets | AWS Secrets Manager or GCP Secret Manager |
| Groq free tier | Paid Groq or switch to Anthropic/OpenAI based on cost/quality needs |
| No authentication | Auth0 or Supabase for user accounts + search history persistence |
| SerpAPI failure shown as a plain in-chat message | Proper circuit breaker + retry with exponential backoff, plus alerting to the team when quota nears its limit |

---

## 10. Why surface SerpAPI failures explicitly instead of falling back to mock data?

**Decision:** When SerpAPI fails (quota exhausted, invalid key, network error), `search_products_node` shows the user a plain explanation ("SerpAPI search quota has been used up for now...") instead of quietly substituting a local mock product catalog.

**Reasoning:**
An earlier version of this project fell back to a static local JSON file of sample products whenever SerpAPI failed, so the demo would "still work." In practice this is worse than an honest error: the user has no way to tell real, live prices from fabricated demo data, which is exactly the kind of silent failure a shopping assistant shouldn't have — trust in the prices shown is the entire point of the product. An explicit message costs nothing and can't be mistaken for a real deal.

**Trade-off:** A live demo can go visibly blank if the SerpAPI quota runs out mid-demo, which a fallback would have papered over. Considered acceptable — an honest "quota's out, try again shortly" message is a better failure mode than a shopping tool that might be silently lying about prices.

**Production note:** `SerpAPIError` (in `utils/exceptions.py`) is the single point where this distinction is made — a retry/circuit-breaker layer would wrap around `SerpApiClient.search_google_shopping()` without needing to touch any node code.

---

## 11. Known limitations and honest trade-offs

- **Price analysis is not based on real historical price data.** `price_validity_node` asks the LLM to judge whether a listed price looks inflated using its general knowledge of market pricing for that product category — it does not track or graph actual price history over time. Real historical price tracking would require a paid service like Keepa (Amazon-only) or building a price database over time. This is clearly framed in the UI as an LLM judgment call, not verified history.

- **Review text is not fetched.** SerpAPI's `google_shopping` engine returns rating and review count, but not review text. Fetching actual review snippets requires a second SerpAPI call using the `google_product` engine with a product ID — planned but not implemented due to API quota constraints.

- **Follow-up classification can misfire on ambiguous queries.** "Show me something else" is genuinely ambiguous — is it a follow-up ("something else from what you showed me") or a new search? The classifier uses a prompt that biases toward FOLLOW_UP when products were recently shown, but edge cases exist.

- **SerpAPI's "no results" and "real failure" cases aren't distinguished yet.** SerpAPI returns a query that legitimately has zero shopping matches through the same `"error"` field used for actual outages/quota problems. Both currently surface the same "SerpAPI unavailable" message, even though the right next step is different (broaden the search vs. wait and retry). Flagged as a follow-up fix, not yet implemented.

- **No user authentication or persistent history.** Conversation history lives only in `st.session_state` and is lost on page refresh. A real product would persist this to a database tied to a user account.