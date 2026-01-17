# CAF Memory Search Module - Unified search architecture for all memory types

from .base import BaseSearchEngine, SearchEngineRegistry
from .engines import SemanticSearchEngine
from .embedding import create_embedding_client, EmbeddingConfig

# Register semantic search engine
SearchEngineRegistry.register('semantic', SemanticSearchEngine)

__all__ = [
    # Base Infrastructure
    'BaseSearchEngine',
    'SearchEngineRegistry',
    
    # Embedding Services  
    'create_embedding_client',
    'EmbeddingConfig',
    
    # Specific Engines
    'SemanticSearchEngine'
]

