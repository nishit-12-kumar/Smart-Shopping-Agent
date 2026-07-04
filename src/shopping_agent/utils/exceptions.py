class ShoppingAgentBaseException(Exception):
    """Base exception for all Smart Shopping Agent errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class SerpAPIError(ShoppingAgentBaseException):
    """Raised when the SerpAPI Google Shopping search fails or returns an error payload."""
    pass
