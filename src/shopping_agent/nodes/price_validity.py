from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.shopping_agent.graph.state import ShoppingState
from src.shopping_agent.services.groq_client import GroqClient
from src.shopping_agent.utils.logger import agent_logger

# 1. Define Strict Output Schema
class DiscountAnalysis(BaseModel):
    is_suspicious: bool = Field(description="True if the claimed price or discount seems artificially inflated or unrealistic for the category, False otherwise.")
    analysis_reasoning: str = Field(description="A short explanation of why the price/discount is realistic or suspicious based on standard market norms.")

class PriceValidityOutput(BaseModel):
    analyses: List[DiscountAnalysis] = Field(description="List of analyses matching the order of the input products.")

def price_validity_node(state: ShoppingState) -> ShoppingState:
    """
    Evaluates validated deals to detect fake or inflated discount patterns.
    """
    agent_logger.info("Entering price_validity_node.")
    
    try:
        deals = state.get("validated_deals", [])
        
        if not deals:
            agent_logger.warning("No validated deals found. Skipping price validity check.")
            return state

        # Initialize Groq Client with structured output
        groq_client = GroqClient().get_llm()
        structured_llm = groq_client.with_structured_output(PriceValidityOutput)

        # 2. Construct the Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert consumer advocate and pricing analyst. "
                       "Review the following products and their prices. Determine if the current price "
                       "represents a genuine deal or if it relies on a 'fake discount' dark pattern "
                       "(e.g., claiming a standard market price is actually a massive discount). "
                       "Rely on your internal knowledge of standard pricing for these product categories."),
            ("human", "Evaluate these products for suspicious pricing:\n\n{deals}")
        ])

        chain = prompt | structured_llm
        
        # Format deals for the prompt
        deals_text = "\n".join([
            f"- {p.get('title')} | Current Price: ₹{p.get('price')} | Source: {p.get('source')}" 
            for p in deals
        ])
        
        agent_logger.info(f"Sending {len(deals)} products to Groq for fake-discount analysis.")
        result = chain.invoke({"deals": deals_text})
        
        # 3. Merge the analysis back into the validated deals state
        for idx, analysis_data in enumerate(result.analyses):
            # Ensure we don't go out of bounds if the LLM miscounts (rare with structured output)
            if idx < len(deals):
                deals[idx]["is_suspicious_pricing"] = analysis_data.is_suspicious
                deals[idx]["pricing_analysis"] = analysis_data.analysis_reasoning
                
                if analysis_data.is_suspicious:
                    agent_logger.warning(f"Suspicious pricing flagged for product: {deals[idx].get('title')}")

        state["validated_deals"] = deals
        agent_logger.info("Price validity check completed successfully.")

    except Exception as e:
        agent_logger.error(f"Error in price_validity_node: {str(e)}", exc_info=True)
        if "errors" not in state: state["errors"] = []
        state["errors"].append(f"Price validity check failed: {str(e)}")
        
        # Fail open: if the node crashes, assume prices are fine so we don't break the whole app
        for deal in state.get("validated_deals", []):
            deal["is_suspicious_pricing"] = False
            deal["pricing_analysis"] = "Analysis unavailable due to system error."
            
    return state