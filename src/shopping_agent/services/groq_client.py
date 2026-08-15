from functools import lru_cache

from langchain_groq import ChatGroq
from src.shopping_agent.config import GROQ_API_KEY
from src.shopping_agent.utils.logger import agent_logger


@lru_cache(maxsize=None)
def get_groq_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> ChatGroq:
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
            max_tokens=1024
        )
        agent_logger.info(
            f"Groq client initialized with model: {model_name} (temperature={temperature})"
        )
        return llm
    except Exception as e:
        agent_logger.critical(f"Failed to initialize Groq client: {str(e)}", exc_info=True)
        raise


class GroqClient:
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        self.llm = get_groq_llm(model_name, temperature)

    def get_llm(self):
        """Returns the shared, cached ChatGroq instance."""
        return self.llm