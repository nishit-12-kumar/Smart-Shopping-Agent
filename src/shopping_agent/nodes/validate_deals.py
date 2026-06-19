from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import GroqClient
from src.shopping_agent.utils.logger import agent_logger

# 1. Define the Strict Output Schema using Pydantic
class ProductEvaluation(BaseModel):
    confidence_score: int = Field(description="A score from 0 to 100 indicating how good of a deal this is.")
    reasoning: str = Field(description="A one-line explanation of why this score was given, considering price, specs, and reviews.")

class DealValidationOutput(BaseModel):
    evaluations: List[ProductEvaluation] = Field(description="List of evaluations matching the order of the input products.")


def _flag_low_review_count(product: dict):
    rating = product.get("rating")
    reviews = product.get("reviews")
    if rating and reviews and rating >= 4.5 and reviews < 50:
        return f"⚠️ High rating ({rating}★) but only {reviews} reviews — insufficient data to fully trust this rating."
    return None

def validate_deals_node(state: ShoppingState) -> ShoppingState:
    """
    Evaluates the raw products using Groq LLM and assigns a confidence score.
    """
    agent_logger.info("Entering validate_deals_node.")
    
    try:
        raw_products = state.get("raw_products", [])
        user_query = state.get("user_query", "")
        
        if not raw_products:
            agent_logger.warning("No raw products found to validate. Skipping.")
            return state

        # Initialize the Groq LLM and bind the Pydantic schema
        groq_client = GroqClient().get_llm()
        structured_llm = groq_client.with_structured_output(DealValidationOutput)

        # 2. Construct the Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert shopping negotiator and deal validator. "
                       "Evaluate the provided products against the user's original request. "
                       "Assign a confidence score (0-100) based on relevance, price competitiveness, and review reliability."),
            ("human", "User Request: {query}\n\nProducts to Evaluate:\n{products}")
        ])

        # 3. Create the Chain and Execute
        chain = prompt | structured_llm
        
        agent_logger.info(f"Sending {len(raw_products)} products to Groq for evaluation.")
        
        # Format products into a clean string to save tokens
        products_text = "\n".join([
            f"- {p['title']} | Price: {p['price']} | Rating: {p['rating']} ({p['reviews']} reviews)" 
            for p in raw_products
        ])
        
        result = chain.invoke({"query": user_query, "products": products_text})
        
        # 4. Merge evaluations back into the product dictionaries
        validated_deals = []
        for product, eval_data in zip(raw_products, result.evaluations):
            validated_item = product.copy()
            validated_item["confidence_score"] = eval_data.confidence_score
            validated_item["reasoning"] = eval_data.reasoning
            validated_item["review_flag"] = _flag_low_review_count(product)
            validated_deals.append(validated_item)
            
        state["validated_deals"] = validated_deals
        agent_logger.info("Deal validation completed successfully.")

    except Exception as e:
        agent_logger.error(f"Error in validate_deals_node: {str(e)}", exc_info=True)
        if "errors" not in state: state["errors"] = []
        state["errors"].append(f"Deal validation failed: {str(e)}")
        state["validated_deals"] = state.get("raw_products", []) # Fallback to unvalidated products
        
    return state