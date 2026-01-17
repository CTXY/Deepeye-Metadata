# Base Search Engine - Abstract interface for all memory search engines

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from ..types import MemoryQuery, MemoryResponse

logger = logging.getLogger(__name__)

class BaseSearchEngine(ABC):
    """
    Abstract base class for all memory search engines
    
    This provides a unified interface for different types of memory search:
    - SemanticSearchEngine: For semantic memory (facts, knowledge)
    - EpisodicSearchEngine: For episodic memory (events, experiences)
    """
    
    def __init__(self, config: Dict[str, Any], storage_path: Path):
        """
        Initialize the search engine
        
        Args:
            config: Configuration dictionary for the search engine
            storage_path: Path to storage directory
        """
        self.config = config
        self.storage_path = storage_path
        self.current_database_id: Optional[str] = None
        self.indexes_built = False
        
        logger.debug(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def bind_database(self, database_id: str) -> None:
        """
        Bind the search engine to a specific database
        
        Args:
            database_id: Unique identifier for the database
        """
        pass
    
    @abstractmethod 
    def search(self, query: MemoryQuery, dataframes: Dict[str, pd.DataFrame]) -> MemoryResponse:
        """
        Perform search on the bound database
        
        Args:
            query: Memory query object containing search parameters
            dataframes: Database metadata dataframes
            
        Returns:
            MemoryResponse containing search results
        """
        pass
    
    @abstractmethod
    def rebuild_indexes(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """
        Rebuild search indexes for the current database
        
        Args:
            dataframes: Database metadata dataframes
        """
        pass
    
    @abstractmethod
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about current search indexes
        
        Returns:
            Dictionary containing index status and metadata
        """
        pass
    
    def get_engine_type(self) -> str:
        """
        Get the type of memory this engine searches
        
        Returns:
            String identifier for the engine type
        """
        return self.__class__.__name__.replace('SearchEngine', '').lower()
    
    def is_ready(self) -> bool:
        """
        Check if the search engine is ready to perform searches
        
        Returns:
            True if engine is bound to database and indexes are built
        """
        return self.current_database_id is not None and self.indexes_built


class SearchEngineRegistry:
    """
    Registry for managing different types of search engines
    
    Allows for dynamic registration and creation of search engines
    for different memory types.
    """
    
    _engines: Dict[str, type] = {}
    
    @classmethod
    def register(cls, memory_type: str, engine_class: type) -> None:
        """
        Register a search engine for a memory type
        
        Args:
            memory_type: Type of memory (semantic, episodic)
            engine_class: Search engine class that implements BaseSearchEngine
        """
        if not issubclass(engine_class, BaseSearchEngine):
            raise ValueError(f"Engine class must inherit from BaseSearchEngine")
        
        cls._engines[memory_type] = engine_class
        logger.info(f"Registered search engine for memory type: {memory_type}")
    
    @classmethod
    def create_engine(cls, memory_type: str, config: Dict[str, Any], 
                     storage_path: Path) -> BaseSearchEngine:
        """
        Create a search engine instance for the specified memory type
        
        Args:
            memory_type: Type of memory to create engine for
            config: Configuration for the engine
            storage_path: Storage path for the engine
            
        Returns:
            Search engine instance
            
        Raises:
            ValueError: If memory_type is not registered
        """
        if memory_type not in cls._engines:
            available_types = list(cls._engines.keys())
            raise ValueError(f"Unknown memory type: {memory_type}. "
                           f"Available types: {available_types}")
        
        engine_class = cls._engines[memory_type]
        return engine_class(config, storage_path)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """
        Get list of available memory types
        
        Returns:
            List of registered memory type names
        """
        return list(cls._engines.keys())
    
    @classmethod
    def is_registered(cls, memory_type: str) -> bool:
        """
        Check if a memory type is registered
        
        Args:
            memory_type: Memory type to check
            
        Returns:
            True if the memory type has a registered engine
        """
        return memory_type in cls._engines
