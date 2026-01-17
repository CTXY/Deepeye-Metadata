# FAISS Vector Index Manager - Handle FAISS index building and caching

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class FAISSManager:
    """
    FAISS Vector Index Manager
    
    Handles:
    - FAISS index building and caching
    - Embedding computation and storage  
    - Mixed storage strategy (pkl + npy + FAISS native format)
    """
    
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def build_or_load_index(self, database_id: str, cache_key: str, 
                           documents: Dict[str, str], 
                           embedding_client) -> Optional[Dict[str, Any]]:
        """
        Build or load FAISS index with caching
        
        Args:
            database_id: Database identifier
            cache_key: Cache key for this specific index
            documents: Document texts to index
            embedding_client: Embedding client for computing vectors
            
        Returns:
            Dict containing FAISS index and metadata
        """
        
        # Check if cached index exists and is valid
        cached_index = self._load_cached_index(database_id, cache_key, len(documents))
        if cached_index:
            logger.debug(f"Loaded cached FAISS index for {database_id}")
            return cached_index
        
        # Build new index
        logger.info(f"Building new FAISS index for {database_id}")
        return self._build_new_index(database_id, cache_key, documents, embedding_client)
    
    def _load_cached_index(self, database_id: str, cache_key: str, num_docs: int) -> Optional[Dict[str, Any]]:
        """Load cached FAISS index if available and valid"""
        try:
            # Create cache directory path for this specific cache key
            cache_dir = self.cache_path / database_id / cache_key
            
            # File paths in organized directory structure
            embedding_file = cache_dir / "embeddings.npy"
            faiss_file = cache_dir / "faiss.index"  
            metadata_file = cache_dir / "metadata.pkl"
            
            # Check if all files exist
            if not all(f.exists() for f in [embedding_file, faiss_file, metadata_file]):
                return None
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Validate metadata
            if metadata.get('num_documents') != num_docs:
                logger.debug("Cached index has different document count, rebuilding")
                return None
            
            # Load embeddings
            embeddings = np.load(embedding_file)
            
            # Load FAISS index
            faiss_index = faiss.read_index(str(faiss_file))
            
            return {
                'faiss_index': faiss_index,
                'embeddings': embeddings,
                'doc_ids': metadata['doc_ids'],
                'embedding_dim': embeddings.shape[1],
                'num_documents': len(embeddings),
                'cached': True
            }
            
        except Exception as e:
            logger.warning(f"Failed to load cached FAISS index: {e}")
            return None
            
    
    def _build_new_index(self, database_id: str, cache_key: str, 
                        documents: Dict[str, str], embedding_client) -> Dict[str, Any]:
        """Build new FAISS index from documents"""
        try:
            
            doc_ids = list(documents.keys())
            doc_texts = list(documents.values())
            
            # Compute embeddings
            logger.debug(f"Computing embeddings for {len(doc_texts)} documents")
            embeddings = embedding_client.encode_batch(doc_texts)
            
            # Ensure embeddings are 2D numpy array
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            embedding_dim = embeddings.shape[1]
            
            # Build FAISS index
            # Use IndexFlatIP for cosine similarity (assumes normalized embeddings)
            faiss_index = faiss.IndexFlatIP(embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            faiss_index.add(embeddings.astype(np.float32))
            
            logger.info(f"Built FAISS index with {faiss_index.ntotal} vectors, dim={embedding_dim}")
            
            # Cache the index and embeddings
            self._cache_index(database_id, cache_key, faiss_index, embeddings, doc_ids)
            
            return {
                'faiss_index': faiss_index,
                'embeddings': embeddings,
                'doc_ids': doc_ids,
                'embedding_dim': embedding_dim,
                'num_documents': len(embeddings),
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            # Fallback to in-memory vector index
            return self._create_fallback_index(documents)
    
    def _cache_index(self, database_id: str, cache_key: str, 
                    faiss_index, embeddings: np.ndarray, doc_ids: List[str]) -> None:
        """Cache FAISS index, embeddings, and metadata"""
        try:
            # Create cache directory path for this specific cache key
            cache_dir = self.cache_path / database_id / cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # File paths in organized directory structure
            embedding_file = cache_dir / "embeddings.npy"
            faiss_file = cache_dir / "faiss.index"
            metadata_file = cache_dir / "metadata.pkl"
            
            # Save embeddings
            np.save(embedding_file, embeddings)
            
            # Save FAISS index
            faiss.write_index(faiss_index, str(faiss_file))
            
            # Save metadata
            metadata = {
                'doc_ids': doc_ids,
                'num_documents': len(doc_ids),
                'embedding_dim': embeddings.shape[1],
                'cache_key': cache_key,
                'database_id': database_id
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.debug(f"Cached FAISS index for {database_id} in {cache_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to cache FAISS index: {e}")
    
    def _create_fallback_index(self, documents: Dict[str, str]) -> Dict[str, Any]:
        """Create fallback vector index when FAISS is not available"""
        return {
            'faiss_index': None,
            'embeddings': None,
            'doc_ids': list(documents.keys()),
            'embedding_dim': None,
            'num_documents': len(documents),
            'cached': False,
            'fallback': True,
            'documents': documents  # Keep documents for lazy embedding
        }
    
    def search(self, query_embedding: np.ndarray, index_data: Dict[str, Any], 
              top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Search FAISS index for similar vectors
        
        Args:
            query_embedding: Query vector
            index_data: Index data from build_or_load_index
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        
        try:

            faiss_index = index_data['faiss_index']
            doc_ids = index_data['doc_ids']
            
            # Ensure query is 2D and normalized
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            query_embedding = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = faiss_index.search(query_embedding, min(top_k, faiss_index.ntotal))
            
            # Convert to results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(doc_ids):  # Valid index
                    doc_id = doc_ids[idx]
                    # Convert inner product back to similarity score (0-1 range)
                    similarity = max(0.0, float(score))  # Clamp to [0, 1]
                    results.append((doc_id, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _fallback_search(self, query_embedding: np.ndarray, index_data: Dict[str, Any], 
                        top_k: int) -> List[Tuple[str, float]]:
        """Fallback search when FAISS is not available"""
        # This would require computing embeddings on-the-fly
        # For now, return empty results
        logger.warning("FAISS fallback search not implemented, returning empty results")
        return []
    
    def clear_cache(self, database_id: str) -> None:
        """Clear cached FAISS indexes for a database"""
        pattern = f"{database_id}_*"
        extensions = ['_embeddings.npy', '_faiss.index', '_faiss_meta.pkl']
        
        deleted_count = 0
        for ext in extensions:
            for cache_file in self.cache_path.glob(f"{pattern}{ext}"):
                try:
                    cache_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted FAISS cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} FAISS cache files for database: {database_id}")
    
    def get_cache_info(self, database_id: str) -> Dict[str, Any]:
        """Get information about cached FAISS indexes"""
        pattern = f"{database_id}_*"
        cache_files = {
            'embeddings': list(self.cache_path.glob(f"{pattern}_embeddings.npy")),
            'indexes': list(self.cache_path.glob(f"{pattern}_faiss.index")),
            'metadata': list(self.cache_path.glob(f"{pattern}_faiss_meta.pkl"))
        }
        
        return {
            'cache_path': str(self.cache_path),
            'cached_files': {k: len(v) for k, v in cache_files.items()},
            'total_cache_files': sum(len(v) for v in cache_files.values())
        }
