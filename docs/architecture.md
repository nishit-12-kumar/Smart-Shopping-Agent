# Smart Shopping Agent — System Architecture

## Overview

The Smart Shopping Agent is an agentic shopping system that converts natural-language product requests into structured, explainable recommendations.

Instead of relying on a single LLM call, the application uses a **LangGraph-based stateful workflow**. Each stage has a focused responsibility: understanding the message, determining intent, collecting missing requirements, retrieving live products, validating product relevance, analyzing pricing risk, and producing a deterministic recommendation with LLM-assisted explanations.

The architecture is intentionally divided into:

- **Agent orchestration** — LangGraph
- **Language reasoning** — Groq / Llama 3.3 70B
- **Live product retrieval** — SerpAPI Google Shopping
- **Deterministic product analysis** — Python
- **Presentation** — Streamlit

This separation makes the recommendation pipeline easier to reason about, test, debug, and extend.

---

## High-Level Architecture

```text
                              User
                               │
                               ▼
                    ┌───────────────────────┐
                    │     Streamlit UI      │
                    │   Chat + Spec Form    │
                    └───────────┬───────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ classify_message_type   │
                   └────────────┬────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                 CHITCHAT                SHOPPING
                    │                       │
                    ▼                       ▼
                   END             ┌─────────────────┐
                                   │ classify_intent │
                                   └────────┬────────┘
                                            │
                                  ┌─────────┴─────────┐
                                  │                   │
                              FOLLOW_UP              NEW
                                  │                   │
                                  ▼                   ▼
                         answer_followup       ┌─────────────┐
                                  │             │ parse_query │
                                  │             └──────┬──────┘
                                  │                    │
                                  │           ┌────────┴────────┐
                                  │           │                 │
                                  │     Clarification       Has Specs
                                  │           │                 │
                                  │           ▼                 ▼
                                  │          END        search_products
                                  │           │                 │
                                  │    Streamlit form          ▼
                                  │                    validate_deals
                                  │                            │
                                  │                            ▼
                                  │                    price_validity
                                  │                            │
                                  │                            ▼
                                  │                       synthesize
                                  │                            │
                                  └────────────────────────────┴──► END
```

---

## Core Design

The graph follows a **single shared state** through the pipeline.

```text
User Message
     │
     ▼
Understand message
     │
     ▼
Determine conversation intent
     │
     ├── Follow-up ──────────────► Use existing product memory
     │
     └── New search
             │
             ▼
        Check requirements
             │
             ├── Missing information ──► Streamlit clarification form
             │                              │
             │                              ▼
             │                         Refined query
             │                              │
             └──────────────────────────────┘
                            │
                            ▼
                     Live product search
                            │
                            ▼
                    Product validation
                            │
                            ▼
                    Pricing-risk analysis
                            │
                            ▼
                 Deterministic recommendation
                            │
                            ▼
                    LLM explanation
                            │
                            ▼
                         UI card
```

---

# Graph Nodes

## 1. `classify_message_type`

### Responsibility

This is the entry point of the LangGraph workflow.

It determines whether the user's message is:

- `CHITCHAT`
- `SHOPPING`

Examples of chitchat include:

```text
Hi
Thanks
What can you do?
Hello
```

Shopping messages contain a product-search or purchase intent.

### Behavior

For chitchat:

```text
classify_message_type
        ↓
Generate conversational response
        ↓
END
```

The expensive shopping pipeline is skipped.

For shopping:

```text
classify_message_type
        ↓
classify_intent
```

### Main State

**Input**

```text
user_query
```

**Output**

```text
message_type
final_recommendation
```

---

## 2. `classify_intent`

### Responsibility

Determines whether a shopping message represents:

- a **NEW** product search
- a **FOLLOW_UP** about products already shown

This node acts as the memory gate of the application.

### Example

After showing products:

```text
Is the first one good for programming?
```

The node can identify that the user is referring to an existing product rather than starting a new search.

### Memory Used

```text
last_shown_deals
```

If no previous products are available, the request is treated as a new search without requiring an unnecessary follow-up classification.

### Output

```text
intent
search_params.referenced_product
```

### Routing

```text
FOLLOW_UP → answer_followup
NEW       → parse_query
```

---

## 3. `answer_followup`

### Responsibility

Answers questions about products that have already been displayed.

Instead of searching the web again, the node uses:

```text
last_shown_deals
conversation context
referenced product
user question
```

This makes follow-up questions faster and avoids unnecessary search/API calls.

### Example

```text
User:
Show me a laptop under ₹80,000.

Agent:
[Product 1]
[Product 2]
[Product 3]

User:
Is the first one good for coding?
```

The system resolves the first product from the stored results and answers using the existing product data.

### Routing

```text
answer_followup
       ↓
      END
```

---

# 4. `parse_query`

### Responsibility

Determines whether the user's shopping request contains enough information to perform a useful product search.

A vague request such as:

```text
I want to buy a laptop
```

may require additional preferences.

A more specific request such as:

```text
Gaming laptop under ₹80,000 with 16GB RAM
```

can proceed directly to search.

### Clarification Flow

When more information is needed:

```text
parse_query
     │
     ▼
clarification_needed = True
     │
     ▼
END
     │
     ▼
Streamlit renders clarification form
```

The UI can generate category-specific specification options through `generate_spec_options()`.

Examples of information that may be collected:

- Budget
- RAM
- Processor
- Storage
- Brand
- Usage
- Size
- Category-specific requirements

After the user submits the form, the selected values are incorporated into a refined search query and the agent pipeline is invoked again.

### Output

```text
clarification_needed
search_params
```

---

# 5. `search_products`

### Responsibility

Retrieves live product information from Google Shopping using SerpAPI.

The search query is built from the user's original request and, when applicable, the selected clarification options.

### Typical Product Data

The returned product objects can contain:

```text
product_id
title
price
source
rating
reviews
thumbnail
product_link
```

The product link is extracted from the available SerpAPI shopping result data.

### Search Flow

```text
Refined Query
     │
     ▼
SerpAPI
     │
     ▼
Google Shopping Results
     │
     ▼
raw_products
```

The application keeps the retrieved product information as data rather than asking the LLM to recreate it.

---

# 6. `validate_deals`

### Responsibility

Evaluates how well each retrieved product matches the user's actual requirements.

Groq / Llama 3.3 70B is used with structured output to produce validation information.

Each product can receive:

```text
confidence_score
reasoning
review_flag
```

The confidence score represents product suitability for the user's request rather than simply whether the product is cheap.

### Review Trust Heuristic

The pipeline also uses a deterministic review signal.

For example:

```text
Very high rating
+
Very small review count
        ↓
Potential review-trust warning
```

This check does not require an additional LLM call.

### Output

```text
validated_deals
```

where each product contains the original product data plus validation signals.

---

# 7. `price_validity`

### Responsibility

Analyzes whether a product's displayed pricing appears suspicious.

The pricing system is designed to look beyond the headline discount.

### Pricing Signals

The system can consider:

```text
Discount level
      +
MRP/current-price relationship
      +
Peer-product prices
      +
Price outlier behavior
      +
Rating/review evidence
      ↓
Pricing Risk
```

Products are compared with similar listings where possible.

The peer-price analysis uses robust statistics such as:

- Median
- MAD (Median Absolute Deviation)
- Modified z-score

This helps identify unusually expensive or unusually cheap listings without allowing a single extreme product to dominate the reference price.

### Important Design Principle

A large discount is **not automatically treated as fake**.

Pricing risk is based on multiple signals and is surfaced as a warning rather than silently removing the product.

### Output

Pricing information is added to the validated product objects, including fields such as:

```text
is_suspicious_pricing
pricing_analysis
```

---

# 8. `synthesize`

### Responsibility

Builds the final recommendation shown to the user.

This is one of the most important architectural boundaries in the system.

The LLM is **not responsible for deciding the final product ranking or inventing product information**.

Instead:

```text
                    validated products
                           │
                           ▼
                Deterministic Python logic
                           │
              ┌────────────┴────────────┐
              │                         │
       Product scoring            Product selection
              │                         │
              └────────────┬────────────┘
                           ▼
                    Top Pick + Alternatives
                           │
                           ▼
                   LLM-generated reasoning
                           │
                           ▼
                 Structured recommendation
```

### Recommendation Score

The current recommendation score uses weighted components:

| Component | Weight |
|---|---:|
| Requirement Match | 50% |
| Rating | 15% |
| Reviews | 10% |
| Price Value | 10% |
| Pricing Trust | 15% |

This means a product is not automatically ranked first simply because it has the lowest price.

### LLM Responsibility

Groq is used for qualitative fields such as:

```text
why_it_wins
specs_matched
specs_warning
alternative_trade_offs
red_flags
bottom_line
follow_up_suggestions
```

### Data Integrity

Critical product fields remain tied to retrieved/deterministic data:

```text
Product title
Product price
Product image
Product link
Product ID
Recommendation score
Pricing-risk result
```

The LLM provides reasoning around those values instead of replacing them.

### Memory Update

After synthesis, the system stores information required for future conversation turns, including:

```text
last_shown_deals
conversation_history
```

---

# Shared State

All graph nodes communicate through a shared `ShoppingState`.

Conceptually:

```python
class ShoppingState(TypedDict):
    user_query: str
    clarification_needed: bool
    search_params: Optional[Dict[str, Any]]

    raw_products: List[Dict[str, Any]]
    validated_deals: List[Dict[str, Any]]

    message_type: str
    intent: str

    final_recommendation: str
    structured_recommendation: Optional[Dict]

    serpapi_error_message: Optional[str]
    errors: List[str]

    conversation_history: List[Dict[str, Any]]
    last_shown_deals: List[Dict[str, Any]]
```

The state acts as the shared memory and communication layer of the graph.

---

# Conversation Memory

The system maintains several pieces of conversational context.

## `conversation_history`

Stores information from previous turns so that the application can maintain conversational continuity.

## `last_shown_deals`

Stores the products from the most recent product search.

This is particularly important for follow-up questions.

For example:

```text
Search 1
 ├── s1p1
 ├── s1p2
 └── s1p3
```

A later follow-up can refer to:

```text
the first one
the second one
option 3
```

without requiring the user to repeat the product details.

## Product IDs

Products are assigned identifiers based on search turns, for example:

```text
s1p1
s1p2
s1p3

s2p1
s2p2
s2p3
```

This provides a lightweight way to reference previously displayed products.

---

# UI Architecture

The Streamlit application is separated from the graph logic.

The main UI responsibilities include:

```text
Chat interface
     │
     ├── User input
     ├── Conversation history
     ├── Agent execution status
     ├── Clarification form
     └── Recommendation cards
```

## Main UI Components

### `app.py`

Responsible for:

- Streamlit application lifecycle
- Session state
- User input
- Running the LangGraph pipeline
- Streaming graph updates
- Rendering conversation history

### `ui/components.py`

Responsible for reusable visual components such as:

- Top Pick card
- Alternative product cards
- Product images
- Ratings
- Recommendation score
- Specification match indicators
- Pricing-risk warnings
- Recommendation reasoning
- Follow-up suggestions

### `ui/styles.py`

Contains the visual styling used by the Streamlit application.

---

# Agent Execution in the UI

When the user sends a new message, the UI invokes the graph and listens to its streamed execution.

Conceptually:

```text
User submits message
        │
        ▼
run_agent_pipeline()
        │
        ▼
LangGraph execution
        │
        ├── Graph updates
        │      ↓
        │   Progress/status UI
        │
        └── LLM messages
               ↓
          Streaming output
        │
        ▼
Final graph state
        │
        ▼
Recommendation rendering
```

This allows the interface to show execution progress instead of appearing frozen while the agent works.

---

# Clarification UI Flow

Clarification is handled outside the normal final recommendation rendering flow.

```text
User
 │
 ▼
parse_query
 │
 └── clarification_needed = True
          │
          ▼
    Streamlit UI
          │
          ▼
generate_spec_options()
          │
          ▼
Dynamic specification form
          │
          ▼
User selects requirements
          │
          ▼
Refined search query
          │
          ▼
LangGraph pipeline
          │
          ▼
Product recommendations
```

The specification options are generated according to the product/category instead of maintaining a single hardcoded form for every product type.

---

# Error Handling

The application distinguishes between different failure conditions instead of treating every empty result as the same problem.

## SerpAPI Failure

If the SerpAPI client encounters an actual API failure such as:

- quota exhaustion
- invalid API key
- connection failure

the client raises a `SerpAPIError`.

The search node catches the error and stores a user-facing message in:

```text
serpapi_error_message
```

The pipeline can then finish gracefully without fabricating product results.

---

## No Matching Products

A genuine search with no useful products is different from an API failure.

In this case, the application can return a normal recommendation message suggesting that the user adjust requirements such as:

- Budget
- Brand
- Product specifications

---

## Missing Product Information

Product listings may not contain every field.

The UI and recommendation logic therefore avoid assuming that optional information exists.

Examples:

```text
Missing rating
Missing review count
Missing image
Missing direct product link
```

The system can fall back to available information rather than inventing values.

---

# Reliability Boundary

A central architectural principle is:

```text
             LLM
              │
      Language + Reasoning
              │
              ▼
    ┌─────────────────────┐
    │  Deterministic Core │
    └─────────────────────┘
              │
       Product Decisions
              │
              ▼
             UI
```

### LLM is best suited for:

- Natural-language classification
- Intent understanding
- Product-fit reasoning
- Explanation generation
- Follow-up responses
- Dynamic question generation

### Deterministic Python is used for:

- State management
- Product IDs
- Product ranking
- Recommendation scoring
- Pricing-risk calculations
- Peer-price statistics
- Data validation
- Product selection
- Error handling

This reduces the risk of inconsistent recommendations and prevents generated text from becoming the source of truth for product data.

---

# External Services

The system currently depends on two primary external services.

| Service | Role |
|---|---|
| **Groq / Llama 3.3 70B** | Language understanding, classification, validation, explanations |
| **SerpAPI Google Shopping** | Live product listings and shopping data |

The Streamlit application acts as the presentation layer, while LangGraph coordinates the internal workflow.

---

# End-to-End Example

Consider:

```text
I want a laptop for programming under ₹80,000.
```

The complete flow is:

```text
1. Streamlit receives the message
        ↓
2. classify_message_type
        ↓
   SHOPPING
        ↓
3. classify_intent
        ↓
   NEW
        ↓
4. parse_query
        ↓
   Requirements are sufficient
        ↓
5. search_products
        ↓
   SerpAPI Google Shopping
        ↓
6. validate_deals
        ↓
   Product-fit confidence scores
        ↓
7. price_validity
        ↓
   Pricing-risk analysis
        ↓
8. synthesize
        ↓
   Deterministic ranking
        +
   LLM explanation
        ↓
9. Streamlit recommendation card
        ↓
10. Products saved to last_shown_deals
```

If the user then asks:

```text
Is the first one good for coding?
```

the flow changes:

```text
User follow-up
      ↓
classify_message_type
      ↓
SHOPPING
      ↓
classify_intent
      ↓
FOLLOW_UP
      ↓
answer_followup
      ↓
Use last_shown_deals
      ↓
END
```

No new product search is required.

---

# Project-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                         Streamlit UI                        │
│                                                             │
│  Chat │ Clarification Form │ Recommendation Cards │ Status │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       LangGraph Layer                        │
│                                                             │
│  classify_message_type                                      │
│          ↓                                                  │
│  classify_intent                                            │
│       ↙       ↘                                             │
│ follow-up    new search                                     │
│    ↓             ↓                                          │
│ answer       parse_query                                    │
│ follow-up         ↓                                         │
│             search_products                                 │
│                    ↓                                        │
│             validate_deals                                  │
│                    ↓                                        │
│             price_validity                                  │
│                    ↓                                        │
│                synthesize                                   │
└───────────────┬───────────────────────┬─────────────────────┘
                │                       │
                ▼                       ▼
       ┌────────────────┐      ┌────────────────────┐
       │ Groq / Llama   │      │ SerpAPI Google     │
       │ 3.3 70B        │      │ Shopping           │
       └────────────────┘      └────────────────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Deterministic Core  │
                 │                     │
                 │ Scoring             │
                 │ Pricing Analysis    │
                 │ State               │
                 │ Product Selection   │
                 └──────────┬──────────┘
                            │
                            ▼
                    Structured Result
                            │
                            ▼
                      Streamlit UI
```

---

## Architectural Principles

The system is built around five main principles:

1. **Specialized nodes** instead of one large agent.
2. **Conditional routing** so unnecessary work is skipped.
3. **Live product data** rather than LLM-generated product information.
4. **Deterministic scoring and pricing analysis** for reproducible decisions.
5. **Conversational memory** so users can naturally continue discussing previously displayed products.

Together, these components form the architecture of the Smart Shopping Agent: a conversational shopping system that combines **agentic orchestration, live retrieval, deterministic decision logic, and LLM-powered explanations**.
