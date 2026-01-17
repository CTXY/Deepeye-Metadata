# SPLADE Matcher - Advanced sparse retrieval using SPLADE neural models

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import pickle
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ..indexing.column_indexer import ColumnFacetProvider
from caf.config.paths import PathConfig

logger = logging.getLogger(__name__)

@dataclass
class SPLADEMatchResult:
    """Result from SPLADE matching"""
    column_id: str
    score: float
    facet: str  # Which facet produced this match

class SPLADEMatcher:
    """
    SPLADE Matcher for Advanced Sparse Retrieval
    
    Uses SPLADE (Sparse Lexical and Expansion) model for neural sparse retrieval.
    SPLADE combines the efficiency of sparse retrieval with the effectiveness 
    of neural models by learning sparse representations.
    
    Features:
    - Neural sparse document/query representations
    - Efficient inverted index storage
    - Compatible interface with BM25Matcher
    - Multi-facet support (names, description)
    """
    
    def __init__(self, 
                 facet_weights: Optional[Dict[str, float]] = None,
                 model_name: str = "/home/yangchenyu/pre-trained-models/splade-v3",
                 cache_path: Optional[Path] = None,
                 device: Optional[str] = None):
        """
        Initialize SPLADE matcher
        
        Args:
            facet_weights: Weights for different facets in combined scoring
            model_name: HuggingFace model name for SPLADE
            cache_path: Path to cache directory for models and indexes (defaults to unified cache)
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        self.facet_weights = facet_weights or {
            'names': 1.0,        # Highest weight for name matching
            'description': 1.0,   # Standard weight for descriptions  
        }
        
        self.model_name = model_name
        self.cache_path = cache_path if cache_path else PathConfig.get_splade_cache_path()
        # Create cache directory on initialization
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model components
        self.tokenizer = None
        self.model = None
        
        # SPLADE indexes for each facet
        self.splade_indexes: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(f"SPLADEMatcher initialized with model: {self.model_name}, device: {self.device}")
    
    def _load_model(self):
        """Load SPLADE model and tokenizer"""
        if self.model is not None:
            return
        
        try:
            logger.info(f"Loading SPLADE model: {self.model_name}")
            
            # Check if model_name is a local path
            model_path = Path(self.model_name)
            if model_path.exists() and model_path.is_dir():
                logger.info(f"Loading SPLADE model from local path: {model_path}")
                # Load from local directory
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.model = AutoModelForMaskedLM.from_pretrained(str(model_path))
            else:
                logger.info(f"Loading SPLADE model from HuggingFace Hub: {self.model_name}")
                # Load from HuggingFace Hub
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("SPLADE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            raise
    
    def build_indexes(self, provider: ColumnFacetProvider) -> None:
        """
        Build SPLADE indexes for all supported facets
        
        Args:
            provider: Column facet provider for accessing metadata
        """
        logger.info("Building SPLADE indexes...")
        
        # Load model if not already loaded
        self._load_model()
        
        # Reset existing indexes
        self.splade_indexes = {}
        
        # Build index for each facet
        for facet in ['names', 'description']:  # Skip 'values' as it's handled by ValueMatcher
            self._build_facet_index(facet, provider)
    
    def search(self, query: str, top_k: int = 50) -> List[SPLADEMatchResult]:
        """
        Search across all facets using SPLADE
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of SPLADEMatchResult objects sorted by weighted score
        """
        if not self.splade_indexes:
            logger.warning("No SPLADE indexes built")
            return []
        
        all_results = []
        
        # Search each facet
        for facet, weight in self.facet_weights.items():
            if facet not in self.splade_indexes:
                continue
            
            facet_results = self._search_facet(query, facet, top_k)
            
            # Apply facet weight and add to results
            for result in facet_results:
                weighted_result = SPLADEMatchResult(
                    column_id=result.column_id,
                    score=result.score * weight,
                    facet=result.facet
                )
                all_results.append(weighted_result)
        
        # Combine and deduplicate results (keep best score per column)
        combined_results = self._combine_results(all_results)
        
        # Sort by score and return top_k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]
    
    def search_facet(self, query: str, facet: str, top_k: int = 50) -> List[SPLADEMatchResult]:
        """
        Search a specific facet only
        
        Args:
            query: Search query
            facet: Facet to search ('names', 'description', etc.)
            top_k: Maximum number of results
            
        Returns:
            List of SPLADEMatchResult objects for the specific facet
        """
        if facet not in self.splade_indexes:
            logger.warning(f"No SPLADE index for facet '{facet}'")
            return []
        
        return self._search_facet(query, facet, top_k)
    
    def _build_facet_index(self, facet: str, provider: ColumnFacetProvider) -> None:
        """Build SPLADE index for a specific facet"""
        # logger.info(f"Building SPLADE index for facet '{facet}'...")
        
        # Check if cached index exists
        cache_file = self.cache_path / f"splade_index_{facet}.pkl"
        if cache_file.exists():
            try:
                # logger.info(f"Loading cached SPLADE index for facet '{facet}'")
                with open(cache_file, 'rb') as f:
                    self.splade_indexes[facet] = pickle.load(f)
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}, rebuilding...")
        
        texts = []
        column_ids = []
        
        # Collect texts for this facet
        for col_id in provider.iter_column_ids():
            txt = provider.get_faceted_texts(col_id).get(facet, "")
            if txt and txt.strip():
                texts.append(txt)
                column_ids.append(col_id)
        
        if not texts:
            logger.warning(f"No texts for facet '{facet}'")
            return
        
        try:
            # Generate SPLADE representations for all documents
            # logger.info(f"Generating SPLADE representations for {len(texts)} documents...")
            doc_representations = []
            
            # Process documents in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_reps = self._encode_documents(batch_texts)
                doc_representations.extend(batch_reps)
            
            # Build inverted index from sparse representations
            inverted_index = self._build_inverted_index(doc_representations, column_ids)
            
            # Store index data
            self.splade_indexes[facet] = {
                'inverted_index': inverted_index,
                'doc_representations': doc_representations,
                'column_ids': column_ids,
                'texts': texts
            }
            
            # Cache the index
            with open(cache_file, 'wb') as f:
                pickle.dump(self.splade_indexes[facet], f)
            
            # logger.info(f"Built SPLADE index for facet '{facet}' with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build SPLADE index for facet '{facet}': {e}")
    
    def _encode_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Encode documents using SPLADE model
        
        Args:
            texts: List of document texts
            
        Returns:
            List of sparse representations (dict of token_id: weight)
        """
        representations = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model output
                outputs = self.model(**inputs)
                logits = outputs.logits  # [1, seq_len, vocab_size]
                
                # Apply ReLU and aggregation (max pooling over sequence)
                relu_logits = torch.relu(logits)
                sparse_rep = torch.max(relu_logits, dim=1)[0]  # [1, vocab_size]
                
                # Convert to sparse representation (only non-zero values)
                sparse_rep = sparse_rep.cpu().numpy()[0]
                non_zero_indices = np.nonzero(sparse_rep)[0]
                
                # Create sparse dict representation
                sparse_dict = {}
                for idx in non_zero_indices:
                    weight = float(sparse_rep[idx])
                    if weight > 0.01:  # Filter very small weights
                        sparse_dict[int(idx)] = weight
                
                representations.append(sparse_dict)
        
        return representations
    
    def _build_inverted_index(self, doc_representations: List[Dict[int, float]], column_ids: List[str]) -> Dict[int, List[Tuple[str, float]]]:
        """
        Build inverted index from document representations
        
        Args:
            doc_representations: List of sparse document representations
            column_ids: Corresponding column IDs
            
        Returns:
            Inverted index: token_id -> [(column_id, weight), ...]
        """
        inverted_index = {}
        
        for doc_idx, (doc_rep, column_id) in enumerate(zip(doc_representations, column_ids)):
            for token_id, weight in doc_rep.items():
                if token_id not in inverted_index:
                    inverted_index[token_id] = []
                inverted_index[token_id].append((column_id, weight))
        
        return inverted_index
    
    def _search_facet(self, query: str, facet: str, top_k: int) -> List[SPLADEMatchResult]:
        """Search a specific facet using SPLADE"""
        if facet not in self.splade_indexes:
            return []
        
        try:
            # Encode query
            query_rep = self._encode_documents([query])[0]
            
            if not query_rep:
                return []
            
            # Retrieve candidates using inverted index
            index_data = self.splade_indexes[facet]
            inverted_index = index_data['inverted_index']
            
            # Score all documents
            doc_scores = {}
            
            for token_id, query_weight in query_rep.items():
                if token_id in inverted_index:
                    for column_id, doc_weight in inverted_index[token_id]:
                        if column_id not in doc_scores:
                            doc_scores[column_id] = 0.0
                        doc_scores[column_id] += query_weight * doc_weight
            
            # Create results
            results = []
            for column_id, score in doc_scores.items():
                if score > 0:
                    results.append(SPLADEMatchResult(
                        column_id=column_id,
                        score=float(score),
                        facet=facet
                    ))
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"SPLADE search failed for facet '{facet}': {e}")
            return []
    
    def _combine_results(self, results: List[SPLADEMatchResult]) -> List[SPLADEMatchResult]:
        """Combine results from multiple facets, keeping best score per column"""
        if not results:
            return []
        
        # Group by column_id, keep result with highest score
        best_results = {}
        
        for result in results:
            column_id = result.column_id
            
            if (column_id not in best_results or 
                result.score > best_results[column_id].score):
                best_results[column_id] = result
        
        return list(best_results.values())
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about built indexes"""
        info = {
            'mode': 'SPLADE',
            'facet_weights': self.facet_weights,
            'model_name': self.model_name,
            'device': str(self.device),
            'facets': list(self.splade_indexes.keys())
        }
        
        for facet, index_data in self.splade_indexes.items():
            info[f'{facet}_documents'] = len(index_data['column_ids'])
            info[f'{facet}_vocab_size'] = len(index_data['inverted_index'])
        
        return info
