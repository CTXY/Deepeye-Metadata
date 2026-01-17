# Exceptions for User Feedback subsystem

class UserFeedbackError(Exception):
    """Base exception for User Feedback errors"""
    
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"{error_code}: {message}")

class FeedbackTimeoutError(UserFeedbackError):
    """Raised when feedback collection times out"""
    
    def __init__(self, timeout_seconds: int):
        super().__init__("FEEDBACK_TIMEOUT", f"Feedback collection timed out after {timeout_seconds} seconds")

class LLMProviderError(UserFeedbackError):
    """Raised when LLM provider encounters an error"""
    
    def __init__(self, provider: str, message: str):
        super().__init__("LLM_PROVIDER_ERROR", f"LLM provider {provider} error: {message}")

class InvalidFeedbackContextError(UserFeedbackError):
    """Raised when feedback context is invalid"""
    
    def __init__(self, message: str = "Invalid feedback context"):
        super().__init__("INVALID_CONTEXT", message)
