# Smart Shopping Agent — Engineering Decisions & Trade-offs

This document records the main engineering decisions behind the Smart Shopping Agent, why each approach was chosen, the trade-offs involved, and the areas that would need to evolve for a production deployment.

The system is designed around a simple principle:

> **Use LLMs for language understanding and explanation, and deterministic code for data integrity and product decisions.**

---

## 1. Why LangGraph for orchestration?

### Decision

Use **LangGraph** as the orchestration layer instead of implementing the entire workflow as one large function or a simple linear chain.

### Reasoning

The shopping workflow is inherently conditional.

Different user messages require different paths:

```text
Chitchat
   ↓
Direct response

Shopping
   ↓
New search / Follow-up

New search
   ↓
Enough information / Clarification required
```

After product retrieval, additional stages also need to execute in a defined order:

```text
Search
  ↓
Validation
  ↓
Pricing analysis
  ↓
Recommendation synthesis
```

LangGraph provides:

- Explicit nodes
- Shared state
- Conditional routing
- Clear execution flow
- Easier debugging
- Natural support for multi-turn agent workflows

### Trade-off

LangGraph introduces additional concepts:

- State schemas
- Nodes
- Edges
- Routing functions
- Graph compilation

For a very small chatbot this would be unnecessary complexity. For this project, the branching workflow makes the complexity worthwhile.

### Production direction

A production version could use LangGraph's interruption/checkpointing capabilities more extensively for long-running human-in-the-loop workflows and persistent conversations.

---

# 2. Why Groq + Llama 3.3 70B?

### Decision

Use **Groq with Llama 3.3 70B** for the LLM layer.

### Reasoning

The project needs an LLM for several language-heavy tasks:

- Message classification
- Follow-up intent detection
- Product-fit reasoning
- Recommendation explanations
- Follow-up answers
- Dynamic specification generation

Groq is particularly useful for this application because response latency matters in an interactive shopping interface.

Fast inference improves the experience when the user is waiting for:

```text
classification
      ↓
validation
      ↓
reasoning
      ↓
final response
```

Llama 3.3 70B provides sufficient reasoning capability for these tasks while remaining practical for a portfolio project.

### Trade-off

The application remains dependent on:

- API availability
- Rate limits
- Model availability
- External inference costs at scale

### Architectural advantage

The LLM integration is isolated in the service layer, so the rest of the graph does not need to know the details of the provider.

---

# 3. Why SerpAPI instead of directly scraping retailers?

### Decision

Use **SerpAPI Google Shopping** for live product retrieval.

### Reasoning

Direct scraping of retailer websites introduces several problems:

- HTML structure changes
- Anti-bot systems
- CAPTCHAs
- IP restrictions
- Browser automation requirements
- Fragile parsing logic
- Potential terms-of-service concerns

SerpAPI provides structured shopping results that can be consumed by the application without maintaining retailer-specific scraping logic.

The system can receive information such as:

```text
Product title
Price
Source
Rating
Review count
Thumbnail
Product link
```

### Architectural benefit

The product-search integration is isolated behind a service client.

Conceptually:

```text
SerpAPI
   ↓
serpapi_client
   ↓
Normalized product data
   ↓
All downstream nodes
```

This means downstream recommendation logic does not need to know how the product was retrieved.

### Trade-off

SerpAPI introduces:

- Search quotas
- API costs at scale
- Dependency on an external service

### Production direction

A production system could use:

- Paid SerpAPI
- Retailer APIs
- Multiple product-data providers
- A caching layer
- Retry and circuit-breaker mechanisms

---

# 4. Why use structured Pydantic output?

### Decision

Use **Pydantic structured outputs** for LLM tasks where the result becomes application data.

### Reasoning

Several LLM responses are not merely text. They become fields consumed by downstream Python code.

For example:

```text
confidence_score
pricing flags
recommendation reasoning
specification matches
follow-up suggestions
```

A structured schema makes the boundary between the LLM and application logic explicit.

Instead of:

```text
LLM → arbitrary paragraph → regex → application
```

the system can use:

```text
LLM
 ↓
Structured schema
 ↓
Validated Python object
 ↓
Application logic
```

This improves:

- Type safety
- Predictability
- Error detection
- Maintainability

### Trade-off

Structured output is less suitable when the desired result is conversational prose or when token-by-token streaming is important.

The project therefore uses structured output selectively rather than forcing every LLM interaction into the same format.

---

# 5. Why is recommendation selection deterministic?

### Decision

Do **not** ask the LLM to decide which product is the final Top Pick.

The system calculates recommendation scores in Python and uses those scores to select the Top Pick and alternatives.

### Reasoning

An LLM can generate convincing reasoning while still making factual mistakes.

If the LLM were responsible for the final product selection, it could potentially:

- Choose the wrong product
- Misread a specification
- Prefer a product for an unsupported reason
- Accidentally reproduce an incorrect price
- Change its decision between otherwise identical requests

Instead:

```text
Retrieved Products
       ↓
Deterministic scoring
       ↓
Ranked products
       ↓
Top Pick + Alternatives
       ↓
LLM explanation
```

The LLM explains the decision; deterministic code makes the decision.

### Recommendation weighting

The current scoring system uses:

| Factor | Weight |
|---|---:|
| Requirement Match | 50% |
| Rating | 15% |
| Reviews | 10% |
| Price Value | 10% |
| Pricing Trust | 15% |

### Benefit

The ranking becomes:

- Reproducible
- Inspectable
- Easier to debug
- Less vulnerable to LLM hallucination

### Trade-off

A fixed scoring formula cannot perfectly model every human purchasing preference.

For example, two users may value:

```text
lowest price
```

and

```text
best long-term reliability
```

very differently.

A future version could allow personalized weighting without giving up deterministic scoring.

---

# 6. Why separate pricing-risk analysis from recommendation scoring?

### Decision

Pricing trust is treated as its own analytical component rather than simply sorting products by discount percentage.

### Reasoning

A displayed discount does not necessarily represent a good deal.

For example:

```text
MRP: ₹1,00,000
Selling Price: ₹55,000
```

looks attractive, but the displayed MRP may not represent the normal market price.

The system therefore considers multiple signals.

Conceptually:

```text
Discount
   +
Current/MRP relationship
   +
Peer-product pricing
   +
Outlier behavior
   +
Rating/review evidence
   ↓
Pricing Risk
```

### Peer-price analysis

Similar products can be grouped and compared using robust statistics such as:

- Median
- MAD
- Modified z-score

This reduces sensitivity to extreme listings.

### Trade-off

This is still an inference system, not a true historical price tracker.

Without historical price data, the system cannot prove that an original price was artificially inflated.

Therefore, pricing analysis is treated as a **risk signal**, not a definitive fraud verdict.

---

# 7. Why use a weighted recommendation score instead of cheapest-price sorting?

### Decision

Rank products using multiple signals instead of simply selecting the lowest price.

### Reasoning

The cheapest product is not necessarily the best product.

Consider:

```text
Product A
₹55,000
Rating: 3.7
Few reviews
Weak requirement match

Product B
₹60,000
Rating: 4.5
Thousands of reviews
Excellent requirement match
```

A pure price sort would select Product A.

The recommendation engine instead considers:

```text
Requirement Match
Rating
Reviews
Price Value
Pricing Trust
```

This better represents the actual question:

> "Which product is the best fit for this user?"

rather than:

> "Which product is cheapest?"

---

# 8. Why dynamic clarification instead of hardcoded product forms?

### Decision

Generate clarification questions dynamically instead of maintaining a large hardcoded form for every product category.

### Reasoning

A static implementation might look like:

```text
Laptop → RAM, CPU, Storage
Phone  → Storage, Camera, Battery
AC     → Tonnage, Room Size
Shoes  → Size, Type
```

This works only for categories explicitly anticipated by the developer.

A shopping assistant should be able to handle products that were never hardcoded.

The dynamic approach allows the LLM to determine which specifications are useful for a category.

### Flow

```text
User Query
    ↓
Missing information detected
    ↓
Generate category-specific options
    ↓
Streamlit form
    ↓
User selections
    ↓
Refined query
    ↓
Product search
```

### Trade-off

Dynamic generation adds an additional LLM call.

The application can mitigate unnecessary repeated calls by keeping generated options in Streamlit session state during the current interaction.

---

# 9. Why multiple routing stages instead of one large classifier?

### Decision

Use separate stages for:

```text
classify_message_type
        ↓
classify_intent
        ↓
parse_query
```

rather than one large classifier.

### Reasoning

Each stage has one responsibility.

### `classify_message_type`

Answers:

> Is this message a shopping request or normal conversation?

### `classify_intent`

Answers:

> Is this a new search or a follow-up about previously displayed products?

### `parse_query`

Answers:

> Does this new search contain enough information to search effectively?

This follows the **Single Responsibility Principle**.

### Benefits

If a classification problem occurs, it is easier to determine:

```text
Which stage failed?
```

Each prompt can also be improved independently.

### Trade-off

Multiple stages can introduce additional latency and LLM calls.

However, routing allows the system to terminate early:

```text
Chitchat
   ↓
END
```

and:

```text
Follow-up
   ↓
answer_followup
   ↓
END
```

So the entire shopping pipeline is not executed for every message.

---

# 10. Why answer follow-ups from memory instead of searching again?

### Decision

Follow-up questions about recently displayed products use the stored conversation/product state instead of automatically performing another product search.

### Reasoning

Suppose the assistant has already shown:

```text
s1p1
s1p2
s1p3
```

and the user asks:

```text
Is the first one good for programming?
```

A new search would be unnecessary.

The relevant product already exists in:

```text
last_shown_deals
```

The follow-up path therefore becomes:

```text
Follow-up
    ↓
Resolve referenced product
    ↓
Use stored product data
    ↓
LLM explanation
    ↓
END
```

### Benefits

- Lower latency
- Fewer API calls
- Better conversational continuity
- More predictable answers
- Avoids changing the product being discussed

### Trade-off

Ambiguous references can still be difficult.

For example:

```text
Show me something else.
```

could mean:

- another product from the current search
- a completely new search

The intent classifier must infer the most likely interpretation from context.

---

# 11. Why keep product IDs such as `s1p1`?

### Decision

Assign products lightweight conversation-scoped identifiers.

Example:

```text
Search 1:
s1p1
s1p2
s1p3

Search 2:
s2p1
s2p2
s2p3
```

### Reasoning

Users naturally refer to:

```text
the first one
option 2
the third product
```

The application needs a reliable way to resolve those references against the products that were actually displayed.

The identifiers make the mapping explicit inside the conversation state.

### Trade-off

These IDs are conversation references, not permanent product identifiers.

They should not be treated as database-level product IDs.

---

# 12. Why keep UI and agent logic separate?

### Decision

Keep the Streamlit presentation layer separate from the LangGraph/node logic.

### Reasoning

The graph should focus on:

```text
Understanding
Routing
Retrieval
Validation
Scoring
Recommendation
```

while the UI focuses on:

```text
Chat rendering
Forms
Cards
Status indicators
Styling
```

This separation makes it possible to change the interface without rewriting the recommendation pipeline.

Conceptually:

```text
                 LangGraph
                    │
          structured recommendation
                    │
                    ▼
              Streamlit UI
```

### Benefit

A future application could replace Streamlit with another frontend while keeping most of the agent/backend architecture intact.

---

# 13. Why use live product data instead of mock products?

### Decision

Use live SerpAPI shopping data as the source of product information.

### Reasoning

A shopping assistant is fundamentally a data-trust problem.

Showing:

```text
fake price
fake product
fake rating
fake product link
```

would undermine the entire purpose of the application.

Therefore, product facts are kept tied to retrieved data.

The LLM can explain a product, but it should not become the source of truth for:

```text
price
title
image
link
product identity
```

---

# 14. Why explicitly surface API failures instead of silently using fake fallback products?

### Decision

If the live product provider fails, the system reports the failure instead of silently substituting invented or static product results.

### Reasoning

A shopping assistant must distinguish between:

```text
No products matched
```

and:

```text
The product-search service failed
```

Using fake fallback products can make a user believe that displayed prices are current when they are not.

An explicit failure is therefore more trustworthy.

### Failure flow

```text
SerpAPI
   ↓
Failure
   ↓
SerpAPIError
   ↓
search_products handles failure
   ↓
serpapi_error_message
   ↓
synthesize surfaces the explanation
```

### Trade-off

The application may visibly fail when the external service is unavailable.

That is considered preferable to silently presenting unreliable shopping information.

### Production direction

A production implementation could add:

- Retries
- Exponential backoff
- Circuit breakers
- Multiple data providers
- Monitoring
- Quota alerts

without changing the fundamental trust model.

---

# 15. Why distinguish API failure from no useful products?

### Decision

The state contains a dedicated:

```text
serpapi_error_message
```

rather than using an empty product list to represent every failure.

### Reasoning

These two situations have different meanings:

```text
raw_products = []
serpapi_error_message = None
```

can represent:

> Search completed, but there were no useful results.

Whereas:

```text
raw_products = []
serpapi_error_message = "..."
```

means:

> The search service itself failed.

The distinction allows the UI to provide a more accurate message.

### Known limitation

Some external API error responses may not clearly distinguish:

```text
legitimate zero results
```

from:

```text
service/API failure
```

If the provider exposes both situations through the same error representation, the application cannot always infer the exact cause reliably.

---

# 16. Why per-conversation logging?

### Decision

Maintain conversation-specific log files rather than putting every interaction into one large application log.

### Reasoning

A shopping conversation can involve several pipeline stages:

```text
classification
search
validation
pricing analysis
synthesis
```

When debugging a particular interaction, having a session-specific log makes it much easier to reconstruct what happened.

For example:

```text
Conversation A
    ↓
Search laptop
    ↓
Unexpected pricing result
```

can be investigated without manually separating entries from unrelated sessions.

### Trade-off

Flat per-conversation files do not scale well for a large multi-user application.

### Production direction

Use:

- Structured JSON logging
- Centralized log aggregation
- Metrics
- Tracing
- Error monitoring

---

# 17. Why use session state for current conversation memory?

### Decision

Keep the active conversation state in Streamlit session state.

### Reasoning

The application is currently designed as an interactive portfolio/demo application.

Session state provides a simple way to maintain:

```text
conversation_history
last_shown_deals
search context
current UI state
```

without introducing database infrastructure.

### Trade-off

Session state is not suitable for long-term persistence.

Refreshing or losing the session can remove the conversation context.

### Production direction

A production application would move persistent state to something such as:

```text
Redis
PostgreSQL
Document database
```

and associate the state with authenticated users.

---

# 18. Why not make the entire application LLM-driven?

### Decision

Use the LLM as a reasoning component rather than the sole decision-maker.

### Reasoning

An LLM is excellent at:

```text
Natural language
Ambiguous intent
Explanations
Semantic reasoning
Conversation
```

It is not the ideal source of truth for:

```text
Arithmetic
Ranking formulas
Product prices
IDs
Links
Peer-price statistics
State transitions
```

Therefore the architecture deliberately separates responsibilities.

```text
             LLM
              │
       Language reasoning
              │
              ▼
      Structured information
              │
              ▼
      Deterministic Python
              │
       Product decisions
              │
              ▼
             UI
```

This hybrid architecture gives the project both conversational flexibility and deterministic behavior.

---

# 19. Why use a hybrid LLM + deterministic pricing system?

### Decision

Use LLM reasoning where language interpretation is useful, while using Python for measurable pricing signals.

### Reasoning

Pricing analysis involves two different problems.

### Structured numerical analysis

Python is better suited for:

```text
discount calculations
price ratios
medians
MAD
outlier detection
weighted scores
```

### Natural-language explanation

The LLM is better suited for:

```text
"This product looks unusually expensive compared with similar listings."
```

The combination is therefore:

```text
Numerical analysis
      ↓
Risk signals
      ↓
LLM explanation
      ↓
User-friendly warning
```

This is more controllable than asking an LLM to perform the entire numerical analysis from scratch.

---

# 20. Why expose pricing risk instead of removing suspicious products?

### Decision

Flag potentially suspicious products rather than automatically deleting them from the result set.

### Reasoning

A suspicious price does not necessarily mean the product is fraudulent.

Possible explanations include:

- Genuine clearance
- Temporary promotion
- Different seller
- Different configuration
- Regional pricing
- Data inconsistencies

Therefore:

```text
Suspicious
    ≠
Definitely fake
```

The system keeps the product available while making the risk visible to the user.

This gives the user more information without pretending that the system can prove fraud.

---

# 21. Production Evolution

The current architecture is suitable for a portfolio/demo application. A production implementation would evolve several layers.

| Current approach | Production direction |
|---|---|
| Streamlit session state | Persistent user/session database |
| Flat conversation logs | Centralized structured logging |
| SerpAPI dependency | Paid/multi-provider product data layer |
| Single LLM provider | Provider abstraction + fallback model |
| `.env` secrets | Managed secret store |
| No authentication | User authentication and authorization |
| Basic API error handling | Retry + circuit breaker + monitoring |
| Current-session memory | Persistent user preferences/history |
| Live search for every new query | Search/result caching |
| Fixed recommendation weights | Configurable/personalized ranking |
| Limited price intelligence | Historical price database |
| Basic product matching | Stronger semantic deduplication |

---

# 22. Known Limitations

The architecture intentionally has several limitations that are important to keep in mind.

### 22.1 No true historical price tracking

Pricing-risk analysis uses currently available product information and peer-price signals.

It does not maintain a long-term historical graph such as:

```text
₹80k
 │
₹75k
 │
₹62k
 │
₹59k
```

Therefore, the system should describe pricing as a **risk assessment**, not verified historical price evidence.

---

### 22.2 Review text is not the primary input

The product-search pipeline primarily works with structured signals such as:

```text
rating
review count
```

rather than performing deep sentiment analysis over complete review text.

A future version could retrieve and analyze review snippets when sufficient product-level data is available.

---

### 22.3 Ambiguous follow-ups can still be difficult

Messages such as:

```text
Show me something else.
```

may have more than one valid interpretation.

The classifier uses the current conversation context, but ambiguity cannot always be eliminated.

---

### 22.4 External product data can be incomplete

A shopping listing may not contain:

- Rating
- Review count
- Image
- Direct seller link
- Complete specifications

The application therefore treats many fields as optional rather than assuming that every listing is complete.

---

### 22.5 No long-term user personalization

Current conversation context can influence recommendations, but the application does not yet maintain a permanent user preference profile such as:

```text
Preferred brands
Typical budget
Preferred retailers
Historical purchases
Ranking preferences
```

This would require persistent user-level storage.

---

# Final Architecture Philosophy

The Smart Shopping Agent follows a hybrid architecture:

```text
                    Natural Language
                           │
                           ▼
                    ┌─────────────┐
                    │     LLM     │
                    │             │
                    │ Understand  │
                    │ Classify    │
                    │ Explain     │
                    └──────┬──────┘
                           │
                           ▼
                  Structured Information
                           │
                           ▼
              ┌─────────────────────────┐
              │   Deterministic Core    │
              │                         │
              │ Product Data            │
              │ Recommendation Score    │
              │ Pricing Analysis        │
              │ State & Routing         │
              └────────────┬────────────┘
                           │
                           ▼
                    Final Recommendation
                           │
                           ▼
                      Streamlit UI
```

The central engineering principle is:

> **Let the LLM reason about language, but let deterministic code control the facts and decisions that matter.**

This keeps the system conversational enough to behave like an agent while remaining structured enough to be explainable, debuggable, and trustworthy.
