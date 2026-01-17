# CAF Embedding Providers - Specific implementations for different providers

import logging
from typing import List, Union, Optional
import numpy as np

from .client import BaseEmbeddingClient, EmbeddingConfig
from caf.config.paths import PathConfig

logger = logging.getLogger(__name__)

class SentenceTransformerProvider(BaseEmbeddingClient):
    """Sentence Transformers embedding provider"""
    
    def _initialize_model(self):
        """Initialize Sentence Transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            kwargs = {}
            if self.config.device:
                kwargs['device'] = self.config.device
            
            if self.config.model_path:
                model_source = self.config.model_path
                logger.info(f"Loading model from local path: {model_source}")
            else:
                model_source = self.model_name
                
            # Load the model
            self._model = SentenceTransformer(model_source, **kwargs)
            
            if self.config.max_seq_length:
                self._model.max_seq_length = self.config.max_seq_length
                
            logger.debug(f"Initialized SentenceTransformer model: {model_source}")
            
        except ImportError:
            raise ValueError("sentence-transformers package not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using Sentence Transformers"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            encode_kwargs = {
                'batch_size': kwargs.get('batch_size', self.batch_size),
                'normalize_embeddings': kwargs.get('normalize_embeddings', self.config.normalize_embeddings),
                'convert_to_numpy': True,
                'show_progress_bar': False
            }
            
            embeddings = self._model.encode(texts, **encode_kwargs)
            return embeddings
            
        except Exception as e:
            logger.error(f"SentenceTransformer encoding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self._model.get_sentence_embedding_dimension()


class BGEProvider(BaseEmbeddingClient):
    """BGE (BAAI General Embedding) provider using FlagEmbedding"""
    
    def _initialize_model(self):
        """Initialize BGE model"""
        try:
            from FlagEmbedding import FlagModel
            
            kwargs = {}
            if self.config.device:
                kwargs['device'] = self.config.device
            
            if self.config.model_path:
                model_source = self.config.model_path
                logger.info(f"Loading BGE model from local path: {model_source}")
            else:
                model_source = self.model_name
                
            self._model = FlagModel(model_source, **kwargs)
            logger.debug(f"Initialized BGE model: {model_source}")
            
        except ImportError:
            raise ValueError("FlagEmbedding package not installed. Install with: pip install FlagEmbedding")
        except Exception as e:
            logger.error(f"Failed to initialize BGE model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using BGE"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # BGE supports batch encoding
            embeddings = self._model.encode(texts)
            
            if self.config.normalize_embeddings:
                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"BGE encoding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        # BGE models typically have known dimensions
        test_embedding = self._model.encode(["test"])
        return test_embedding.shape[1]


class SparseEncoderProvider(BaseEmbeddingClient):
    """Sparse Encoder provider (e.g., SPLADE)"""
    
    def _initialize_model(self):
        """Initialize Sparse Encoder model"""
        try:
            # This is a placeholder - actual implementation depends on the specific sparse encoder
            # For example, if using SPLADE or other sparse models
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch
            
            # Determine model source: local path or model name
            if self.config.model_path:
                model_source = self.config.model_path
                logger.info(f"Loading Sparse Encoder model from local path: {model_source}")
            else:
                model_source = self.model_name
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_source)
            self._model = AutoModelForMaskedLM.from_pretrained(model_source)
            
            if self.config.device and torch.cuda.is_available():
                self._model = self._model.to(self.config.device)
                
            logger.debug(f"Initialized Sparse Encoder model: {model_source}")
            
        except ImportError:
            raise ValueError("transformers package not installed. Install with: pip install transformers torch")
        except Exception as e:
            logger.error(f"Failed to initialize Sparse Encoder model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using Sparse Encoder"""
        try:
            import torch
            
            if isinstance(texts, str):
                texts = [texts]
            
            # This is a simplified implementation - actual sparse encoding 
            # would depend on the specific model architecture
            embeddings = []
            
            for text in texts:
                inputs = self._tokenizer(text, return_tensors="pt", 
                                       truncation=True, padding=True,
                                       max_length=self.config.max_seq_length or 512)
                
                if self.config.device and torch.cuda.is_available():
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # Extract embeddings (this is model-specific)
                    # For demonstration, using mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding.flatten())
            
            embeddings = np.array(embeddings)
            
            if self.config.normalize_embeddings:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Sparse Encoder encoding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        # Test with a sample text to get dimension
        test_embedding = self.encode(["test"])
        return test_embedding.shape[1]

