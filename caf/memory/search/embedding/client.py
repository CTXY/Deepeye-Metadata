# CAF Embedding Client - Unified embedding client for CAF system

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Optional, Any, Dict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    BGE = "bge"
    SPARSE_ENCODER = "sparse_encoder"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding client"""
    provider: Union[str, EmbeddingProvider]
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    max_seq_length: Optional[int] = None
    normalize_embeddings: bool = True
    cache_dir: Optional[str] = None
    model_path: Optional[str] = None  # Local path to pre-downloaded model
    
    def __post_init__(self):
        if isinstance(self.provider, str):
            self.provider = EmbeddingProvider(self.provider)

class BaseEmbeddingClient(ABC):
    """Base class for embedding clients"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.device = config.device
        self.batch_size = config.batch_size
        self._model = None
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the embedding model"""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts into embeddings"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass
    
    def encode_single(self, text: str, **kwargs) -> np.ndarray:
        """Encode a single text"""
        embeddings = self.encode([text], **kwargs)
        return embeddings[0] if len(embeddings.shape) > 1 else embeddings
    
    def encode_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode a batch of texts"""
        return self.encode(texts, **kwargs)
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Ensure embeddings are 1D
        if len(embeddings1.shape) > 1:
            embeddings1 = embeddings1.flatten()
        if len(embeddings2.shape) > 1:
            embeddings2 = embeddings2.flatten()
            
        # Calculate cosine similarity
        dot_product = np.dot(embeddings1, embeddings2)
        norm1 = np.linalg.norm(embeddings1)
        norm2 = np.linalg.norm(embeddings2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def batch_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarities between two sets of embeddings"""
        # Normalize embeddings
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Calculate cosine similarity matrix
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    def validate_model(self) -> Dict[str, Any]:
        """Test the embedding model"""
        try:
            test_text = "This is a test sentence for embedding validation."
            embedding = self.encode_single(test_text)
            
            return {
                'valid': True,
                'provider': self.provider.value,
                'model': self.model_name,
                'embedding_dim': len(embedding),
                'device': self.device
            }
        except Exception as e:
            return {
                'valid': False,
                'provider': self.provider.value,
                'model': self.model_name,
                'error': str(e)
            }

def create_embedding_client(config: EmbeddingConfig) -> BaseEmbeddingClient:
    """Factory function to create appropriate embedding client"""
    if config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        from .providers import SentenceTransformerProvider
        return SentenceTransformerProvider(config)
    elif config.provider == EmbeddingProvider.BGE:
        from .providers import BGEProvider
        return BGEProvider(config)
    elif config.provider == EmbeddingProvider.SPARSE_ENCODER:
        from .providers import SparseEncoderProvider
        return SparseEncoderProvider(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

