import os
from dotenv import load_dotenv
from src.shopping_agent.utils.logger import agent_logger

# Load environment variables from the .env file
load_dotenv()

try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    
    if not GROQ_API_KEY or not SERPAPI_API_KEY:
        raise ValueError("Missing critical API keys in environment variables.")
        
    agent_logger.info("Configuration loaded successfully.")
    
except Exception as e:
    agent_logger.critical(f"Failed to load configuration: {str(e)}")
    raise