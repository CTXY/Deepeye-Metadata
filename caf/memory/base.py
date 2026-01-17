# MemoryBase - Core memory system coordinator

from typing import Dict, Optional, Any, Union, Tuple, List
from datetime import datetime
import logging

from .types import MemoryQuery, MemoryResponse, MemoryType, EpisodicRecord
from .types_per_term import FlatMemoryResponse, PerTermMemoryResponse, JoinRelationship
from .exceptions import MemoryAPIError, DatabaseNotBoundError, MemoryStoreNotFoundError
from .stores.base_store import BaseMemoryStore
from .stores.semantic import SemanticMemoryStore
from .stores.episodic import EpisodicMemoryStore
from .stores.guidance import GuidanceMemoryStore
from ..config.loader import CAFConfig

logger = logging.getLogger(__name__)

class MemoryBase:
    """
    Memory system core - manages all Memory-related logic
    
    This class coordinates different memory stores and provides unified access
    to the memory subsystem. It implements database context management and
    query routing.
    """
    
    def __init__(self, config: CAFConfig):
        self.config = config
        self.current_database_id: Optional[str] = None
        self.memory_stores: Dict[MemoryType, BaseMemoryStore] = {}
        
        # Initialize memory stores
        self._initialize_stores()
        
        logger.info("MemoryBase initialized successfully")
    
    def _initialize_stores(self) -> None:
        """Initialize various Memory Stores"""
        try:
            self.memory_stores = {
                MemoryType.EPISODIC: EpisodicMemoryStore(self.config.memory),
                MemoryType.SEMANTIC: SemanticMemoryStore(self.config.memory),
                MemoryType.GUIDANCE: GuidanceMemoryStore(self.config.memory)
            }
            
            # Set memory_base reference for stores that need it
            for store in self.memory_stores.values():
                if hasattr(store, '_memory_base'):
                    store._memory_base = self
            
            logger.info("All memory stores initialized")
        except Exception as e:
            logger.error(f"Failed to initialize memory stores: {e}")
            raise MemoryAPIError("INITIALIZATION_FAILED", f"Memory store initialization failed: {e}")
    
    def bind_database(self, database_id: str) -> None:
        """
        Bind database and notify all MemoryStore observers
        Implements Observer pattern: MemoryBase as Subject, MemoryStore as Observer
        """
        self.current_database_id = database_id
        
        # Notify all MemoryStores to bind to new database
        for memory_type, store in self.memory_stores.items():
            store.bind_database(database_id)

        logger.info(f"Database bound to: {database_id}")
    
    def query(self, query: MemoryQuery, return_per_term: bool = False) -> Union[FlatMemoryResponse, Tuple[PerTermMemoryResponse, List[JoinRelationship]]]:
        """
        Query router - dispatch to corresponding Store based on memory_type
        Supports database filtering
        
        Returns:
            - FlatMemoryResponse if return_per_term=False
            - Tuple[PerTermMemoryResponse, List[JoinRelationship]] if return_per_term=True
        """
        
        # Get target store
        if query.memory_type not in self.memory_stores:
            raise MemoryStoreNotFoundError(query.memory_type.value)
        
        target_store = self.memory_stores[query.memory_type]
        
        try:
            # Execute query
            start_time = datetime.now()
            response = target_store.search(query, return_per_term=return_per_term)
            end_time = datetime.now()
            
            # Update query time
            if return_per_term:
                # response is tuple of (PerTermMemoryResponse, List[JoinRelationship])
                per_term_response, join_relationships = response
                query_time_ms = int((end_time - start_time).total_seconds() * 1000)
                # Note: PerTermMemoryResponse doesn't have query_time_ms field in the current design
                # If needed, we could add it to the model
                logger.debug(f"Query executed successfully for {query.memory_type.value} in {query_time_ms}ms (per_term=True)")
                return response
            else:
                # response is FlatMemoryResponse
                response.query_time_ms = int((end_time - start_time).total_seconds() * 1000)
                logger.debug(f"Query executed successfully for {query.memory_type.value} in {response.query_time_ms}ms")
                return response
            
        except Exception as e:
            logger.error(f"Query failed for {query.memory_type.value}: {e}")
            raise MemoryAPIError("QUERY_FAILED", f"Memory query failed: {e}")
    
    def store_session(self, data) -> None:
        """
        Store session data - accepts EpisodicRecord
        
        Args:
            data: EpisodicRecord object
        """
        try:
            # Store to Episodic Memory
            episodic_store = self.memory_stores[MemoryType.EPISODIC]
            episodic_store.store(data)
            
            session_id = data.session_id if hasattr(data, 'session_id') else 'unknown'
            logger.info(f"Session stored successfully: {session_id}")
            
        except Exception as e:
            session_id = data.session_id if hasattr(data, 'session_id') else 'unknown'
            logger.error(f"Failed to store session {session_id}: {e}")
            raise MemoryAPIError("STORAGE_FAILED", f"Session storage failed: {e}")
    
    def get_memory_store(self, memory_type: MemoryType) -> BaseMemoryStore:
        """Get specific memory store (for advanced usage)"""
        if memory_type not in self.memory_stores:
            raise MemoryStoreNotFoundError(memory_type.value)
        return self.memory_stores[memory_type]
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Cleanup memory stores
            for memory_type, store in self.memory_stores.items():
                if hasattr(store, 'cleanup'):
                    store.cleanup()
            
            logger.info("MemoryBase cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
