# Episodic Search Engine - Core search functionality for episodic memory

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from ...base import BaseSearchEngine
from ....types import MemoryQuery, MemoryResponse, MemoryItem, EpisodicRecord, MemoryType
from ...embedding.client import EmbeddingConfig
from ...embedding.client import create_embedding_client
from .similarity.nlq_similarity import NLQSimilarity
from .similarity.sql_skeleton import SQLSkeletonSimilarity
from .retrievers.multi_layer_retriever import MultiLayerRetriever
from .cache.embedding_cache import EmbeddingCache
from caf.config.paths import PathConfig

logger = logging.getLogger(__name__)

class EpisodicSearchEngine(BaseSearchEngine):
    """
    Episodic Search Engine implementing DAIL-SQL inspired multi-layer retrieval strategy
    
    Features:
    - NLQ similarity using embeddings (with masked question support)
    - SQL skeleton similarity using Jaccard similarity
    - Multi-layer retrieval: same database vs cross database strategies  
    - Embedding caching for performance optimization
    - Ground truth exclusion
    """
    
    def __init__(self, config: Dict[str, Any], storage_path: Path):
        super().__init__(config, storage_path)
        
        # Core configuration
        self.episodic_config = config.get('episodic', {})
        self.search_config = self.episodic_config.get('search', {})
        
        # Initialize components
        self._initialize_components()
        
        logger.info("EpisodicSearchEngine initialized with DAIL-SQL strategy")
    
    def _initialize_components(self) -> None:
        """Initialize search components"""
        # Initialize similarity calculators
        self.nlq_similarity = NLQSimilarity(self.search_config)
        self.sql_similarity = SQLSkeletonSimilarity(self.search_config)
        
        # Initialize multi-layer retriever
        self.retriever = MultiLayerRetriever(
            nlq_similarity=self.nlq_similarity,
            sql_similarity=self.sql_similarity,
            config=self.search_config
        )
        
        # Initialize embedding cache (using unified cache path)
        cache_path = PathConfig.get_episodic_cache_path()
        self.embedding_cache = EmbeddingCache(cache_path)
        
        # Embedding client (initialized later when needed)
        self.embedding_client = None
        
        # Memory base reference (set by parent store)
        self._memory_base = None
        
        logger.debug("EpisodicSearchEngine components initialized")
    
    def set_memory_base(self, memory_base) -> None:
        """Set memory base for schema extraction in similarity calculators"""
        self._memory_base = memory_base
        if self.nlq_similarity:
            self.nlq_similarity.set_memory_base(memory_base)
        if self.sql_similarity:
            self.sql_similarity.set_memory_base(memory_base)
        logger.debug("Memory base reference set for EpisodicSearchEngine")
    
    def bind_database(self, database_id: str) -> None:
        """Bind to specific database"""
        if self.current_database_id == database_id and self.indexes_built:
            return
            
        self.current_database_id = database_id
        self.indexes_built = False
        
        # Initialize embedding client if not already done
        self._ensure_embedding_client()
        
        # Bind components to database
        self.nlq_similarity.bind_database(database_id, self.embedding_client)
        self.embedding_cache.bind_database(database_id)
        
        logger.info(f"EpisodicSearchEngine bound to database: {database_id}")
    
    def search(self, query: MemoryQuery, records: List[EpisodicRecord]) -> MemoryResponse:
        """
        Enhanced episodic search with multi-layer strategy
        
        Args:
            query: Memory query object
            records: List of episodic records to search
            
        Returns:
            MemoryResponse with ranked results
        """
        if not self.current_database_id:
            raise ValueError("No database bound to search engine")
        
        logger.debug(f"Starting episodic search for query: {query.query_content[:100]}...")
        
        # Ensure components are ready
        self._ensure_ready()
        
        # Extract generated_sql from query context if available
        generated_sql = self._extract_generated_sql(query)
        
        # Use multi-layer retriever for DAIL-SQL strategy
        candidates = self.retriever.retrieve(
            query=query,
            records=records,
            generated_sql=generated_sql,
            current_database_id=self.current_database_id,
            exclude_target_question=True
        )
        
        # Convert candidates to MemoryItems
        items = self._candidates_to_memory_items(candidates, query)
        
        logger.info(f"Episodic search returned {len(items)} items")
        
        return MemoryResponse(
            items=items,
            total_count=len(items),
            query_time_ms=0  # Will be set by caller
        )
    
    def _extract_generated_sql(self, query: MemoryQuery) -> Optional[str]:
        """Extract generated SQL from query context"""
        generated_sql = None
        if query.context and 'generated_sql' in query.context:
            generated_sql = query.context['generated_sql']
        elif hasattr(query, 'generated_sql'):
            generated_sql = query.generated_sql
        return generated_sql
    
    def _ensure_embedding_client(self) -> None:
        """Ensure embedding client is initialized"""
        if self.embedding_client is None:
            embedding_config = self._get_embedding_config()
            self.embedding_client = create_embedding_client(embedding_config)
            logger.debug("Embedding client initialized for episodic search")
    
    def _ensure_ready(self) -> None:
        """Ensure all components are ready for search"""
        self._ensure_embedding_client()
        
        # Check if embedding cache needs to be built/loaded
        if not self.embedding_cache.is_ready():
            logger.info("Building embedding cache for episodic records...")
            # This will be called when records are first searched
        
        self.indexes_built = True
    
    def _get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration for episodic memory"""
        from caf.config.paths import PathConfig
        
        embedding_cfg = self.search_config.get('embedding', {})
        
        # Default to DAIL-SQL's model
        return EmbeddingConfig(
            provider=embedding_cfg.get('provider', 'sentence_transformers'),
            model_name=embedding_cfg.get('model_name', 'sentence-transformers/all-mpnet-base-v2'),
            device=embedding_cfg.get('device', 'cpu'),
            batch_size=embedding_cfg.get('batch_size', 32),
            normalize_embeddings=embedding_cfg.get('normalize_embeddings', True),
            max_seq_length=embedding_cfg.get('max_seq_length', 512),
            cache_dir=embedding_cfg.get('cache_dir') or str(PathConfig.get_model_cache_path())
        )
    
    def _candidates_to_memory_items(self, candidates: List[Any], query: MemoryQuery) -> List[MemoryItem]:
        """Convert retrieval candidates to MemoryItem objects"""
        items = []
        
        for candidate in candidates:
            record = candidate.record
            
            # Prepare content dict 
            content = record.model_dump() if hasattr(record, 'model_dump') else record.dict()
            
            # Add retrieval metadata
            content.update({
                'nlq_similarity': candidate.nlq_similarity,
                'sql_skeleton_similarity': candidate.sql_skeleton_similarity, 
                'final_score': candidate.final_score,
                'retrieval_type': candidate.retrieval_type,
                'search_query': query.query_content
            })
            
            # Create MemoryItem
            item = MemoryItem(
                content=content,
                score=candidate.final_score,
                tags={
                    'source': 'episodic_search_engine',
                    'session_id': record.session_id,
                    'database_id': record.database_id,
                    'round_id': record.round_id,
                    'retrieval_type': candidate.retrieval_type,
                    'similarity_method': 'dail_sql_multilayer',
                    'timestamp': record.timestamp
                }
            )
            
            items.append(item)
        
        return items
    
    def rebuild_indexes(self, records: List[EpisodicRecord]) -> None:
        """Rebuild search indexes for episodic records"""
        logger.info("Rebuilding episodic search indexes...")
        
        self._ensure_embedding_client()
        
        # Rebuild embedding cache
        self.embedding_cache.rebuild_cache(records, self.embedding_client)
        
        # Reset components
        self.nlq_similarity.reset()
        
        self.indexes_built = True
        logger.info("Episodic search indexes rebuilt successfully")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about current indexes"""
        if not self.indexes_built:
            return {'status': 'not_built'}
        
        info = {
            'status': 'built',
            'database_id': self.current_database_id,
            'embedding_cache_size': self.embedding_cache.get_cache_size(),
            'nlq_similarity_ready': self.nlq_similarity.is_ready(),
            'sql_similarity_ready': self.sql_similarity.is_ready()
        }
        
        return info
