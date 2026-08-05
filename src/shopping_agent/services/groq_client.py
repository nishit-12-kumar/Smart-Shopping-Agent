from langchain_groq import ChatGroq
from src.shopping_agent.config import GROQ_API_KEY
from src.shopping_agent.utils.logger import agent_logger

class GroqClient:
    """
    Thin wrapper around LangChain's ChatGroq client.

    Centralizes Groq LLM initialization (model choice, temperature, token
    limit, and API key loading from config) so every node in the pipeline
    can request a consistently configured LLM via a single call to
    GroqClient().get_llm().
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        """
        Initializes the Groq LLM client. 
        Temperature is set to 0.0 by default for analytical/reasoning tasks.

        Args:
            model_name (str): The Groq-hosted model to use. Defaults to
                "llama-3.3-70b-versatile".
            temperature (float): Sampling temperature. Defaults to 0.0 for
                deterministic, analytical output (classification, scoring,
                synthesis) rather than creative variation.

        Raises:
            Exception: Re-raises any error encountered while constructing
                the underlying ChatGroq client (e.g. invalid/missing API
                key), after logging it as critical.
        """
        try:
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=model_name,
                temperature=temperature,
                max_tokens=1024
            )
            agent_logger.info(f"Groq client initialized with model: {model_name}")
        except Exception as e:
            agent_logger.critical(f"Failed to initialize Groq client: {str(e)}", exc_info=True)
            raise

    def get_llm(self):
        """Returns the configured ChatGroq instance."""
        return self.llm