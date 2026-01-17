# Unified Vector Matcher - Centralized vector search using FAISS

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from ..indexing.column_indexer import ColumnFacetProvider
from ....embedding.client import BaseEmbeddingClient
from ....utils.faiss_manager import FAISSManager
from caf.config.paths import PathConfig

logger = logging.getLogger(__name__)

@dataclass
class VectorMatchResult:
    """Result from vector matching"""
    column_id: str
    score: float
    facet: str  # Which facet produced this match
    
class VectorMatcher:
    """
    Unified Vector Matcher using FAISS
    
    Provides consistent interface with other matchers (BM25Matcher, ValueMatcher)
    and leverages FAISSManager for efficient vector search and caching.
    
    Features:
    - Multi-facet vector indexing (names, description, values, etc.)
    - FAISS-based efficient search with caching
    - Consistent interface with other matchers
    - Proper embedding normalization and similarity computation
    """
    
    def __init__(self, cache_path: Optional[Path] = None, facets: Optional[List[str]] = None):
        """
        Initialize unified vector matcher
        
        Args:
            cache_path: Path for FAISS index caching (defaults to unified cache)
            facets: List of facets to build indexes for
        """
        self.cache_path = cache_path if cache_path else PathConfig.get_vector_cache_path()
        # Create cache directory on initialization
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.facets = facets or ['description']  # Default to description facet
        
        # FAISS managers for each facet
        self.faiss_managers: Dict[str, FAISSManager] = {}
        self.vector_indexes: Dict[str, Dict[str, Any]] = {}  # facet -> index_data
        
        # Column provider
        self._provider: Optional[ColumnFacetProvider] = None
        
        # Clients (will be set by caller)
        self.embedding_client: Optional[BaseEmbeddingClient] = None
        
        logger.debug(f"VectorMatcher initialized for facets: {self.facets}")
    
    def build_indexes(self,
                      provider: ColumnFacetProvider,
                      embedding_client: BaseEmbeddingClient,
                      database_id: str) -> None:
        """
        Build vector indexes for all configured facets
        
        Args:
            provider: Column facet provider for accessing metadata
            embedding_client: Embedding client for computing vectors
            database_id: Database identifier for caching
        """
        logger.info("Building unified vector indexes...")
        
        self.embedding_client = embedding_client
        self._provider = provider
        
        # Reset existing indexes
        self.faiss_managers = {}
        self.vector_indexes = {}
        
        # Build index for each facet
        for facet in self.facets:
            self._build_facet_index(facet, database_id)
    
    def search(self, query: str, facets: Optional[List[str]] = None, 
               top_k: int = 50) -> List[VectorMatchResult]:
        """
        Search across specified facets using vector similarity
        
        Args:
            query: Search query
            facets: Facets to search (None = search all configured facets)
            top_k: Maximum number of results to return per facet
            
        Returns:
            List of VectorMatchResult objects sorted by score
        """
        if not self.embedding_client:
            logger.warning("No embedding client available for vector search")
            return []
        
        search_facets = facets or self.facets
        all_results = []
        
        # Search each facet
        for facet in search_facets:
            if facet not in self.vector_indexes:
                continue
            
            facet_results = self._search_facet(query, facet, top_k)
            all_results.extend(facet_results)
        
        # Sort by score and return
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results
    
    def search_facet(self, query: str, facet: str, top_k: int = 50) -> List[VectorMatchResult]:
        """
        Search a specific facet only
        
        Args:
            query: Search query
            facet: Facet to search
            top_k: Maximum number of results
            
        Returns:
            List of VectorMatchResult objects for the specific facet
        """
        if facet not in self.vector_indexes:
            logger.warning(f"No vector index for facet '{facet}'")
            return []
        
        return self._search_facet(query, facet, top_k)
    
    def _build_facet_index(self, facet: str, database_id: str) -> None:
        """Build vector index for a specific facet using FAISS"""
        if not self._provider or not self.embedding_client:
            logger.error("Missing required components for index building")
            return
        
        # Extract documents for this facet
        documents = {}  # doc_id -> text
        
        for col_id in self._provider.iter_column_ids():
            txt = self._provider.get_faceted_texts(col_id).get(facet, "")
            if txt and txt.strip():
                documents[col_id] = txt
        
        if not documents:
            logger.warning(f"No documents for facet '{facet}'")
            return
        
        # Create FAISS manager for this facet (use unified cache path directly)
        self.faiss_managers[facet] = FAISSManager(self.cache_path)
        
        # Build or load FAISS index
        cache_key = f"{facet}_columns"
        index_data = self.faiss_managers[facet].build_or_load_index(
            database_id=database_id,
            cache_key=cache_key,
            documents=documents,
            embedding_client=self.embedding_client
        )
        
        if index_data:
            self.vector_indexes[facet] = index_data
            logger.debug(f"Built vector index for facet '{facet}' with {len(documents)} documents")
        else:
            logger.error(f"Failed to build vector index for facet '{facet}'")
    
    def _search_facet(self, query: str, facet: str, top_k: int) -> List[VectorMatchResult]:
        """Search a specific facet using FAISS"""
        if facet not in self.vector_indexes or not self.embedding_client:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_client.encode_single(query)
            
            # Search using FAISS
            index_data = self.vector_indexes[facet]
            faiss_manager = self.faiss_managers[facet]
            
            search_results = faiss_manager.search(
                query_embedding=query_embedding,
                index_data=index_data,
                top_k=top_k
            )
            
            # Convert to VectorMatchResult format
            results = []
            for doc_id, similarity in search_results:
                results.append(VectorMatchResult(
                    column_id=doc_id,
                    score=float(similarity),
                    facet=facet
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed for facet '{facet}': {e}")
            return []
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about built indexes"""
        info = {
            'facets': list(self.vector_indexes.keys()),
            'cache_path': str(self.cache_path)
        }
        
        for facet, index_data in self.vector_indexes.items():
            info[f'{facet}_documents'] = index_data.get('num_documents', 0)
            info[f'{facet}_embedding_dim'] = index_data.get('embedding_dim', 0)
            info[f'{facet}_cached'] = index_data.get('cached', False)
        
        return info
    
    def clear_cache(self, database_id: str) -> None:
        """Clear cached vector indexes for a database"""
        for facet, manager in self.faiss_managers.items():
            try:
                manager.clear_cache(database_id)
                logger.debug(f"Cleared vector cache for facet '{facet}', database '{database_id}'")
            except Exception as e:
                logger.warning(f"Failed to clear vector cache for facet '{facet}': {e}")
