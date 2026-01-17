# Base class for all Memory Stores

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Tuple, List
import logging

from ..types import MemoryQuery, MemoryResponse
from ..types_per_term import FlatMemoryResponse, PerTermMemoryResponse, JoinRelationship

logger = logging.getLogger(__name__)

class BaseMemoryStore(ABC):
    """Base class for all Memory Stores"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_database_id: Optional[str] = None
        self._setup_storage()
        
        logger.debug(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def _setup_storage(self) -> None:
        """Setup storage backend (SQLite/PostgreSQL etc.)"""
        pass
    
    def bind_database(self, database_id: str) -> None:
        """
        Bind to specific database (Observer pattern response)
        Each concrete MemoryStore can override this method for specific logic
        """
        self.current_database_id = database_id
        logger.debug(f"{self.__class__.__name__} bound to database: {database_id}")
    
    @abstractmethod
    def search(self, query: MemoryQuery, return_per_term: bool = False) -> Union[FlatMemoryResponse, Tuple[PerTermMemoryResponse, List[JoinRelationship]], MemoryResponse]:
        """
        Search memory
        
        Args:
            query: Memory query object
            return_per_term: If True, return per-term grouped results; if False, return flat results
            
        Returns:
            Response format depends on memory store type and return_per_term flag
        """
        pass
    
    @abstractmethod
    def store(self, data: Any) -> None:
        """Store data"""
        pass
    
    def update(self, memory_id: str, data: Any) -> None:
        """Update memory (optional implementation)"""
        logger.warning(f"Update not implemented for {self.__class__.__name__}")
        pass
    
    def delete(self, memory_id: str) -> None:
        """Delete memory (optional implementation)"""
        logger.warning(f"Delete not implemented for {self.__class__.__name__}")
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources (optional implementation)"""
        logger.debug(f"{self.__class__.__name__} cleanup completed")
        pass
