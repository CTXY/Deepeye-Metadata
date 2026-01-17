# NLQ Similarity Calculator with Embeddings and Masked Question Support

import logging
import re
from typing import Optional, Dict, Any, List
import numpy as np

from ....embedding.client import BaseEmbeddingClient
from .masked_question import MaskedQuestionProcessor

logger = logging.getLogger(__name__)

class NLQSimilarity:
    """
    NLQ (Natural Language Question) similarity calculator using embeddings
    
    Features:
    - Embedding-based similarity (following DAIL-SQL)
    - Masked question support with schema linking
    - Caching for performance optimization
    - Fallback to keyword-based similarity if embeddings fail
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('nlq_similarity', {})
        
        # Configuration
        self.enable_masking = self.config.get('enable_masked_question', True)
        self.similarity_metric = self.config.get('similarity_metric', 'cosine')  # cosine or euclidean
        self.fallback_to_keywords = self.config.get('fallback_to_keywords', True)
        
        # Components
        self.masked_processor = MaskedQuestionProcessor(self.config) if self.enable_masking else None
        self.embedding_client: Optional[BaseEmbeddingClient] = None
        self.current_database_id: Optional[str] = None
        
        # Memory base reference (set by parent engine)
        self._memory_base = None
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.debug("NLQSimilarity initialized")
    
    def set_memory_base(self, memory_base) -> None:
        """Set memory base for schema extraction in masked processor"""
        self._memory_base = memory_base
        if self.masked_processor:
            self.masked_processor.set_memory_base(memory_base)
        logger.debug("Memory base reference set for NLQSimilarity")
    
    def bind_database(self, database_id: str, embedding_client: BaseEmbeddingClient) -> None:
        """Bind to database and set embedding client"""
        self.current_database_id = database_id
        self.embedding_client = embedding_client
        
        # Bind masked processor if enabled
        if self.masked_processor:
            self.masked_processor.bind_database(database_id)
        
        logger.debug(f"NLQSimilarity bound to database: {database_id}")
    
    def calculate_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity between two natural language questions
        
        Args:
            query1: First question
            query2: Second question
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Apply masking if enabled
            if self.enable_masking and self.masked_processor:
                masked_query1 = self.masked_processor.mask_question(query1)
                masked_query2 = self.masked_processor.mask_question(query2)
            else:
                masked_query1 = query1
                masked_query2 = query2
            
            # Get embeddings (with caching)
            emb1 = self._get_embedding_cached(masked_query1)
            emb2 = self._get_embedding_cached(masked_query2)
            
            if emb1 is None or emb2 is None:
                if self.fallback_to_keywords:
                    logger.warning("Embedding failed, falling back to keyword similarity")
                    return self._keyword_similarity(query1, query2)
                else:
                    return 0.0
            
            # Calculate similarity
            if self.similarity_metric == 'cosine':
                similarity = self._cosine_similarity(emb1, emb2)
            else:  # euclidean
                similarity = self._euclidean_to_similarity(emb1, emb2)
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"NLQ similarity calculation failed: {e}")
            if self.fallback_to_keywords:
                return self._keyword_similarity(query1, query2)
            return 0.0
    
    def _get_embedding_cached(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with caching"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if not self.embedding_client:
            logger.warning("No embedding client available")
            return None
        
        try:
            embedding = self.embedding_client.encode_single(text)
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding for text: {e}")
            return None
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Normalize text for consistent caching
        normalized = ' '.join(text.strip().lower().split())
        return f"{self.current_database_id}_{hash(normalized)}"
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        # Ensure embeddings are 1D
        if len(emb1.shape) > 1:
            emb1 = emb1.flatten()
        if len(emb2.shape) > 1:
            emb2 = emb2.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _euclidean_to_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Convert euclidean distance to similarity score"""
        # Ensure embeddings are 1D
        if len(emb1.shape) > 1:
            emb1 = emb1.flatten()
        if len(emb2.shape) > 1:
            emb2 = emb2.flatten()
        
        # Calculate euclidean distance
        distance = np.linalg.norm(emb1 - emb2)
        
        # Convert to similarity (following DAIL-SQL approach)
        # Smaller distance = higher similarity
        # Use exponential decay: similarity = exp(-distance / scale)
        scale = self.config.get('euclidean_scale', 1.0)
        similarity = np.exp(-distance / scale)
        
        return float(similarity)
    
    def _keyword_similarity(self, query1: str, query2: str) -> float:
        """Fallback keyword-based similarity (from original implementation)"""
        try:
            # Simple keyword-based similarity as fallback
            words1 = set(query1.lower().split())
            words2 = set(query2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            jaccard_score = len(intersection) / len(union) if union else 0.0
            overlap_ratio = len(intersection) / len(words1) if words1 else 0.0
            
            # Combined score similar to original episodic store's keyword similarity
            combined_score = 0.6 * jaccard_score + 0.4 * overlap_ratio
            
            return combined_score
            
        except Exception as e:
            logger.warning(f"Keyword similarity calculation failed: {e}")
            return 0.0
    
    def batch_calculate_similarities(self, target_query: str, queries: List[str]) -> List[float]:
        """
        Calculate similarities between target query and a list of queries
        
        More efficient for large batches as it can use batch embedding computation
        """
        if not queries:
            return []
        
        try:
            # Apply masking if enabled
            if self.enable_masking and self.masked_processor:
                masked_target = self.masked_processor.mask_question(target_query)
                masked_queries = [self.masked_processor.mask_question(q) for q in queries]
            else:
                masked_target = target_query
                masked_queries = queries
            
            # Get target embedding
            target_emb = self._get_embedding_cached(masked_target)
            if target_emb is None:
                if self.fallback_to_keywords:
                    return [self._keyword_similarity(target_query, q) for q in queries]
                return [0.0] * len(queries)
            
            # Get batch embeddings for queries
            query_embeddings = []
            for query in masked_queries:
                emb = self._get_embedding_cached(query)
                if emb is not None:
                    query_embeddings.append(emb)
                else:
                    query_embeddings.append(None)
            
            # Calculate similarities
            similarities = []
            for i, query_emb in enumerate(query_embeddings):
                if query_emb is not None:
                    if self.similarity_metric == 'cosine':
                        sim = self._cosine_similarity(target_emb, query_emb)
                    else:
                        sim = self._euclidean_to_similarity(target_emb, query_emb)
                    similarities.append(max(0.0, min(1.0, sim)))
                else:
                    if self.fallback_to_keywords:
                        sim = self._keyword_similarity(target_query, queries[i])
                        similarities.append(sim)
                    else:
                        similarities.append(0.0)
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Batch similarity calculation failed: {e}")
            if self.fallback_to_keywords:
                return [self._keyword_similarity(target_query, q) for q in queries]
            return [0.0] * len(queries)
    
    def is_ready(self) -> bool:
        """Check if the similarity calculator is ready"""
        return self.embedding_client is not None
    
    def reset(self) -> None:
        """Reset internal state"""
        self._embedding_cache.clear()
        if self.masked_processor:
            self.masked_processor.reset()
        logger.debug("NLQSimilarity reset")
    
    def get_cache_size(self) -> int:
        """Get current cache size"""
        return len(self._embedding_cache)
