class ShoppingAgentBaseException(Exception):
    """Base exception for all Smart Shopping Agent errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

# class SerpAPIError(ShoppingAgentBaseException):
#     """Raised when the SerpAPI Google Shopping search fails or returns an error payload."""
#     pass

# class GroqLLMError(ShoppingAgentBaseException):
#     """Raised when the Groq API fails to return a response or times out."""
#     pass

# class QueryParsingError(ShoppingAgentBaseException):
#     """Raised when the LLM fails to structure the parsed query correctly."""
#     pass

# class ValidationOutputError(ShoppingAgentBaseException):
#     """Raised when the Pydantic structured output validation fails during evaluation."""
#     pass