# Embedding Module - Vector embedding services for search

from .client import create_embedding_client, BaseEmbeddingClient, EmbeddingConfig
from .providers import SentenceTransformerProvider, BGEProvider
# Simplified: Direct client creation without manager

__all__ = [
    'create_embedding_client',
    'BaseEmbeddingClient', 
    'EmbeddingConfig',
    'SentenceTransformerProvider',
    'BGEProvider',
    # Simplified: Only core embedding functionality
]