# Core exceptions for CAF system

class CAFSystemError(Exception):
    """Base exception for CAF System errors"""
    
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"{error_code}: {message}")

class SessionNotStartedError(CAFSystemError):
    """Raised when operation requires active session"""
    
    def __init__(self, message: str = "No active session"):
        super().__init__("SESSION_NOT_STARTED", message)

class DatabaseNotBoundError(CAFSystemError):
    """Raised when database is not bound"""
    
    def __init__(self, message: str = "Database not bound"):
        super().__init__("DATABASE_NOT_BOUND", message)

class MemoryQueryFailedError(CAFSystemError):
    """Raised when memory query fails"""
    
    def __init__(self, message: str = "Memory query failed"):
        super().__init__("MEMORY_QUERY_FAILED", message)

class FeedbackRequestFailedError(CAFSystemError):
    """Raised when feedback request fails"""
    
    def __init__(self, message: str = "Feedback request failed"):
        super().__init__("FEEDBACK_REQUEST_FAILED", message)

class SessionFinalizationFailedError(CAFSystemError):
    """Raised when session finalization fails"""
    
    def __init__(self, message: str = "Session finalization failed"):
        super().__init__("SESSION_FINALIZATION_FAILED", message)
