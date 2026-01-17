# Unified BM25 Matcher - Centralized BM25 keyword matching with multi-facet support

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from ..indexing.column_indexer import ColumnFacetProvider

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class BM25MatchResult:
    """Result from BM25 matching"""
    column_id: str
    score: float
    facet: str  # Which facet produced this match
    
class BM25Matcher:
    """
    Unified BM25 Matcher for Multi-Facet Keyword Search
    
    Centralizes BM25 indexing and searching logic that was previously
    duplicated across SearchIndexer, AnchorColumnRetriever, and KeywordMatcher.
    
    Supports multiple facets with different weights:
    - names: High precision for column/table name matching  
    - description: Broad contextual matching
    - values: Entity value matching (if needed)
    """
    
    def __init__(self, facet_weights: Optional[Dict[str, float]] = None):
        """
        Initialize unified BM25 matcher
        
        Args:
            facet_weights: Weights for different facets in combined scoring
        """
        self.facet_weights = facet_weights or {
            'names': 1.0,        # Highest weight for name matching
            'description': 1.0,   # Standard weight for descriptions  
        }
        
        # BM25 indexes for each facet
        self.bm25_indexes: Dict[str, Dict[str, Any]] = {}
        
        # Stop words for tokenization
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("English stopwords not available")
            self.stop_words = set()
        
        logger.debug(f"UnifiedBM25Matcher initialized with facet weights: {self.facet_weights}")
    
    def build_indexes(self, provider: ColumnFacetProvider) -> None:
        """
        Build BM25 indexes for all supported facets
        
        Args:
            provider: Column facet provider for accessing metadata
        """
        logger.info("Building unified BM25 indexes...")
        
        # Reset existing indexes
        self.bm25_indexes = {}
        
        # Build index for each facet
        for facet in ['names', 'description']:  # Skip 'values' as it's handled by ValueMatcher
            self._build_facet_index(facet, provider)
            
    
    def search(self, query: str, top_k: int = 50) -> List[BM25MatchResult]:
        """
        Search across all facets using BM25
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of BM25MatchResult objects sorted by weighted score
        """
        if not self.bm25_indexes:
            logger.warning("No BM25 indexes built")
            return []
        
        all_results = []
        
        # Search each facet
        for facet, weight in self.facet_weights.items():
            if facet not in self.bm25_indexes:
                continue
            
            facet_results = self._search_facet(query, facet, top_k)
            
            # Apply facet weight and add to results
            for result in facet_results:
                weighted_result = BM25MatchResult(
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
    
    def search_facet(self, query: str, facet: str, top_k: int = 50) -> List[BM25MatchResult]:
        """
        Search a specific facet only
        
        Args:
            query: Search query
            facet: Facet to search ('names', 'description', etc.)
            top_k: Maximum number of results
            
        Returns:
            List of BM25MatchResult objects for the specific facet
        """
        if facet not in self.bm25_indexes:
            logger.warning(f"No BM25 index for facet '{facet}'")
            return []
        
        return self._search_facet(query, facet, top_k)
    
    def _build_facet_index(self, facet: str, provider: ColumnFacetProvider) -> None:
        """Build BM25 index for a specific facet"""
        texts = []
        column_ids = []
        
        for col_id in provider.iter_column_ids():
            txt = provider.get_faceted_texts(col_id).get(facet, "")
            if txt and txt.strip():
                texts.append(txt)
                column_ids.append(col_id)
        
        if not texts:
            logger.warning(f"No texts for facet '{facet}'")
            return
        
        # Tokenize documents
        tokenized_docs = []
        for text in texts:
            tokens = self._tokenize_text(text)
            tokenized_docs.append(tokens)
        
        if not tokenized_docs:
            logger.warning(f"No tokenized docs for facet '{facet}'")
            return
        
        # Build BM25 index
        try:
            bm25 = BM25Okapi(tokenized_docs)
            
            self.bm25_indexes[facet] = {
                'bm25': bm25,
                'column_ids': column_ids,
                'tokenized_docs': tokenized_docs,
                'texts': texts
            }
            
            logger.debug(f"Built BM25 index for facet '{facet}' with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index for facet '{facet}': {e}")
    
    def _search_facet(self, query: str, facet: str, top_k: int) -> List[BM25MatchResult]:
        """Search a specific facet using BM25"""
        if facet not in self.bm25_indexes:
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            if not query_tokens:
                return []
            
            # Get BM25 scores
            index_data = self.bm25_indexes[facet]
            bm25 = index_data['bm25']
            column_ids = index_data['column_ids']
            
            scores = bm25.get_scores(query_tokens)
            
            # Normalize scores to [0, 1] range
            if len(scores) > 0:
                max_score = max(scores)
                if max_score > 0:
                    scores = scores / max_score
            
            # Create results for non-zero scores
            results = []
            for i, score in enumerate(scores):
                if score > 0:
                    results.append(BM25MatchResult(
                        column_id=column_ids[i],
                        score=float(score),
                        facet=facet
                    ))
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search failed for facet '{facet}': {e}")
            return []
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing"""
        try:
            # Basic tokenization and filtering
            tokens = word_tokenize(text.lower())
            
            # Filter tokens: alphanumeric only, not stopwords, min length 2
            filtered_tokens = []
            for token in tokens:
                if (token.isalnum() and 
                    len(token) >= 2 and 
                    token not in self.stop_words):
                    filtered_tokens.append(token)
            
            return filtered_tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed for text: {e}")
            return []
    
    def _combine_results(self, results: List[BM25MatchResult]) -> List[BM25MatchResult]:
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
            'facets': list(self.bm25_indexes.keys()),
            'facet_weights': self.facet_weights
        }
        
        for facet, index_data in self.bm25_indexes.items():
            info[f'{facet}_documents'] = len(index_data['column_ids'])
        
        return info
