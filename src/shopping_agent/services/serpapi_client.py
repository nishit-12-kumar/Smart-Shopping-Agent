import json
from pathlib import Path
from typing import List, Dict, Any
from serpapi import GoogleSearch
from src.shopping_agent.config import SERPAPI_API_KEY
from src.shopping_agent.utils.logger import agent_logger

class SerpApiClient:
    def __init__(self):
        self.api_key = SERPAPI_API_KEY

    def search_google_shopping(self, query: str) -> List[Dict[str, Any]]:
        """Fetches product data from Google Shopping via SerpAPI."""
        agent_logger.info(f"Initiating SerpAPI Google Shopping search for: '{query}'")
        
        params = {
            "engine": "google_shopping",
            "q": query,
            "hl": "en",
            "gl": "in",  # Setting geolocation to India for relevant local pricing
            "api_key": self.api_key,
            "num": 5     # Limit results to save LLM tokens downstream
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" in results:
                agent_logger.error(f"SerpAPI returned an error: {results['error']}")
                return self._load_fallback_data()

            shopping_results = results.get("shopping_results", [])
            
            if not shopping_results:
                agent_logger.warning("SerpAPI returned empty shopping results.")
                return []

            # Clean and structure the payload
            parsed_results = []
            for item in shopping_results[:5]:
                parsed_results.append({
                    "title": item.get("title"),
                    "price": item.get("extracted_price"), # Extracts clean float/int
                    "link": item.get("link"),
                    "source": item.get("source"),
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "thumbnail": item.get("thumbnail")   # ADD THIS — product image URL
                })
                
            agent_logger.info(f"Successfully retrieved {len(parsed_results)} products from SerpAPI.")
            return parsed_results

        except Exception as e:
            agent_logger.error(f"Failed to connect to SerpAPI: {str(e)}", exc_info=True)
            return self._load_fallback_data()
    
    def _load_fallback_data(self) -> List[Dict[str, Any]]:
        """Loads mock data if the API fails or quota is exhausted."""
        agent_logger.info("Attempting to load fallback mock data.")
        try:
            # Resolve path to data/sample_products.json
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            mock_file = base_dir / "data" / "sample_products.json"
            
            with open(mock_file, "r") as f:
                return json.load(f)
        except Exception as e:
            agent_logger.critical(f"Failed to load fallback data: {str(e)}")
            return []