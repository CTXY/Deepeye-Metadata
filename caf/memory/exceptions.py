# Exceptions for Memory subsystem

class MemoryAPIError(Exception):
    """Base exception for Memory API errors"""
    
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"{error_code}: {message}")

class DatabaseNotBoundError(MemoryAPIError):
    """Raised when database context is not bound"""
    
    def __init__(self, message: str = "Database context not bound"):
        super().__init__("DATABASE_NOT_BOUND", message)

class MemoryStoreNotFoundError(MemoryAPIError):
    """Raised when requested memory store is not found"""
    
    def __init__(self, memory_type: str):
        super().__init__("MEMORY_STORE_NOT_FOUND", f"Memory store not found: {memory_type}")

class InvalidQueryError(MemoryAPIError):
    """Raised when memory query is invalid"""
    
    def __init__(self, message: str = "Invalid memory query"):
        super().__init__("INVALID_QUERY", message)
