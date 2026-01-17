# Term Retriever - Independent retriever for domain terms

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re

from caf.memory.search.embedding.client import BaseEmbeddingClient
from caf.llm.client import BaseLLMClient
from caf.utils.query_analyzer import IntentAnalysis

logger = logging.getLogger(__name__)

@dataclass
class TermDocument:
    """Document representation for a domain term"""
    term_id: str  # Format: "term.{term_name}"
    term_name: str
    definition: str
    formula: Optional[str] = None
    example_usage: Optional[str] = None
    related_tables: List[str] = None
    related_columns: List[str] = None
    context: Optional[str] = None  # Original context field from CSV
    schemas: Optional[List[str]] = None  # Extracted schemas in table.column format

@dataclass
class TermResult:
    """Result from term retrieval"""
    term_document: TermDocument
    final_score: float
    signal_scores: Dict[str, float]  # keyword, semantic scores
    explanation: Optional[str] = None

@dataclass
class QueryTermTermSelection:
    """每个检索词的term选择结果"""
    query_term: str  # 检索词（keyword）
    selected_terms: List[TermResult]  # 该检索词对应的term结果
    all_candidates_count: int = 0  # 总候选数量

@dataclass
class PerQueryTermTermResult:
    """按检索词分组的term选择结果"""
    query_term_selections: List[QueryTermTermSelection]  # 每个检索词的选择结果
    merged_terms: List[TermResult]  # 合并并去重后的所有terms
    merged_schemas: List[str]  # 合并并去重后的所有schema (table.column格式)

class IndependentTermRetriever:
    """
    Independent Term Retriever
    
    Completely separate retrieval pipeline for domain terms/definitions.
    Operates independently from column retrieval to provide complementary
    domain knowledge information.
    
    Features:
    - Independent indexing of term definitions
    - Keyword and semantic search on terms
    - Rich term document creation
    - Independent scoring without cross-contamination
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('term_retrieval', {})
        
        # Scoring weights
        self.weights = self.config.get('weights', {
            'keyword': 0.4,
            'semantic': 0.6
        })
        
        # Parameters
        self.candidate_limit = self.config.get('candidate_limit', 20)
        
        # Data
        self.term_documents: List[TermDocument] = []
        self.bm25_index: Optional[Dict[str, Any]] = None
        self.vector_index: Optional[Dict[str, Any]] = None
        
        # Clients (will be set by engine)
        self.embedding_client: Optional[BaseEmbeddingClient] = None
        
        logger.debug("IndependentTermRetriever initialized")
    
    def initialize(self, llm_client: BaseLLMClient, embedding_client: BaseEmbeddingClient):
        """Initialize with required clients"""
        self.llm_client = llm_client
        self.embedding_client = embedding_client
    
    def build_indexes(self, database_id: str, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Build indexes specifically for term retrieval"""
        logger.info("Building term indexes...")
        
        # Create term documents
        self.term_documents = self._create_term_documents(dataframes)
        
        if not self.term_documents:
            logger.warning("No term documents created")
            return
        
        # Build BM25 index for keyword search
        self._build_bm25_index()
        
        # Build vector index for semantic search
        self._build_vector_index()
        
        logger.info(f"Term indexes built: {len(self.term_documents)} terms indexed")
    
    def retrieve_terms(
        self, 
        user_question: str, 
        top_k: int = 5, 
        intent_analysis: Optional['IntentAnalysis'] = None,
        return_per_term: bool = False
    ) -> Union[List[TermResult], PerQueryTermTermResult]:
        """
        Retrieve relevant domain terms
        
        Args:
            user_question: User's natural language question
            top_k: Number of top results to return per query term
            intent_analysis: Pre-analyzed intent analysis to avoid redundant parsing (optional)
            return_per_term: If True, return results grouped by query term; if False, return merged results
            
        Returns:
            If return_per_term=False: List of TermResult objects sorted by relevance
            If return_per_term=True: PerQueryTermTermResult with results grouped by query term
        """

        if not self.term_documents or intent_analysis is None:
            if return_per_term:
                return PerQueryTermTermResult(
                    query_term_selections=[],
                    merged_terms=[],
                    merged_schemas=[]
                )
            return []
        
        # Unified retrieval: retrieve terms for each query term (base_phrase and constraints)
        # Extract query terms from entity groups
        query_terms = []
        for entity_group in intent_analysis.entity_groups:
            if entity_group.base_phrase:
                query_terms.append(entity_group.base_phrase)
            for constraint in entity_group.constraints:
                if constraint:
                    query_terms.append(constraint)
        
        if not query_terms:
            logger.warning("No query terms found in intent_analysis")
            if return_per_term:
                return PerQueryTermTermResult(
                    query_term_selections=[],
                    merged_terms=[],
                    merged_schemas=[]
                )
            return []
        
        # Process each query term: retrieve, score, and rank
        query_term_selections = []
        all_term_results = []
        all_schemas = set()
        
        for query_term in query_terms:
            if not query_term.strip():
                continue
            
            # Retrieve candidates for this term
            candidates = self._retrieve_terms_for_phrase(query_term)
            
            # Score and rank
            ranked_results = self._score_and_rank_terms(candidates)
            
            # Take top-k for this term
            top_results = ranked_results[:top_k]
            
            # Extract schemas from selected terms
            term_schemas = set()
            for result in top_results:
                if result.term_document.schemas:
                    term_schemas.update(result.term_document.schemas)
            
            all_schemas.update(term_schemas)
            all_term_results.extend(top_results)
            
            query_term_selections.append(QueryTermTermSelection(
                query_term=query_term,
                selected_terms=top_results,
                all_candidates_count=len(ranked_results)
            ))
        
        # Remove duplicate terms (by term_id) while preserving order
        seen_term_ids = set()
        merged_terms = []
        for result in all_term_results:
            if result.term_document.term_id not in seen_term_ids:
                seen_term_ids.add(result.term_document.term_id)
                merged_terms.append(result)
        
        # Sort merged terms by final_score
        merged_terms.sort(key=lambda x: x.final_score, reverse=True)
        
        # Return format based on return_per_term flag
        if return_per_term:
            print('**************Per-term retrieval*****************')
            print(query_term_selections)
            print(merged_terms)
            print(all_schemas)
            
            return PerQueryTermTermResult(
                query_term_selections=query_term_selections,
                merged_terms=merged_terms,
                merged_schemas=sorted(list(all_schemas))
            )
        else:
            # Return merged results (top_k from merged list)
            top_results = merged_terms[:top_k]
            logger.info(f"Retrieved {len(top_results)} relevant terms from {len(query_terms)} query terms")
            return top_results
    
    def _create_term_documents(self, dataframes: Dict[str, pd.DataFrame]) -> List[TermDocument]:
        """Create rich term documents from dataframes"""
        documents = []
        
        if 'term' not in dataframes or dataframes['term'].empty:
            logger.debug("No term metadata found")
            return documents
        
        for _, row in dataframes['term'].iterrows():
            try:
                doc = self._create_single_term_document(row)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to create term document for {row.get('term_name', 'unknown')}: {e}")
                continue
        
        logger.debug(f"Created {len(documents)} term documents")
        return documents
    
    def _create_single_term_document(self, term_row: pd.Series) -> TermDocument:
        """Create a single term document"""
        term_name = term_row['term_name']
        term_id = f"term.{term_name}"
        
        # Core information
        definition = term_row.get('definition', '')
        formula = term_row.get('formula', '')
        example_usage = term_row.get('example_usage', '')
        context = term_row.get('context', '')
        
        # Extract schemas from context
        schemas = self._extract_schemas_from_context(context)
        
        # Related entities
        related_tables = term_row.get('related_tables', [])
        related_columns = term_row.get('related_columns', [])
        
        # Ensure they are lists
        if not isinstance(related_tables, list):
            related_tables = []
        if not isinstance(related_columns, list):
            related_columns = []
        
        # If we extracted schemas from context, also populate related_tables and related_columns
        if schemas:
            for schema in schemas:
                if '.' in schema:
                    table_name, column_name = schema.split('.', 1)
                    if table_name not in related_tables:
                        related_tables.append(table_name)
                    if schema not in related_columns:
                        related_columns.append(schema)
        
        return TermDocument(
            term_id=term_id,
            term_name=term_name,
            definition=definition,
            formula=formula,
            example_usage=example_usage,
            related_tables=related_tables,
            related_columns=related_columns,
            context=context,
            schemas=schemas
        )
    
    def _extract_schemas_from_context(self, context: str) -> List[str]:
        """
        Extract schema information (table.column) from context string.
        
        Context format examples:
        - "BIRD value_description: table=satscores, column=NumGE1500"
        - "BIRD value_description: table=frpm, column=Free Meal Count (K-12)"
        
        Returns:
            List of schemas in "table.column" format
        """
        if not context or not isinstance(context, str):
            return []
        
        schemas = []
        
        # Pattern 1: "table=xxx, column=yyy" format
        # Match patterns like "table=satscores, column=NumGE1500"
        pattern1 = r'table\s*=\s*([^,]+?)\s*,\s*column\s*=\s*([^,]+?)(?:\s|$|,)'
        matches1 = re.finditer(pattern1, context, re.IGNORECASE)
        for match in matches1:
            table_name = match.group(1).strip()
            column_name = match.group(2).strip()
            if table_name and column_name:
                schemas.append(f"{table_name}.{column_name}")
        
        # Pattern 2: Try to find table and column in other formats
        # Look for "table: xxx" and "column: xxx" patterns
        pattern2 = r'table\s*[:=]\s*([^,\n]+?)(?:\s|$|,|\n)'
        pattern3 = r'column\s*[:=]\s*([^,\n]+?)(?:\s|$|,|\n)'
        
        table_matches = re.finditer(pattern2, context, re.IGNORECASE)
        column_matches = re.finditer(pattern3, context, re.IGNORECASE)
        
        tables = [m.group(1).strip() for m in table_matches]
        columns = [m.group(1).strip() for m in column_matches]
        
        # If we found both tables and columns, create pairs
        if tables and columns:
            # Use the first table with all columns, or pair them up
            for table in tables[:1]:  # Usually there's one table
                for column in columns:
                    schema = f"{table}.{column}"
                    if schema not in schemas:
                        schemas.append(schema)
        
        return schemas
    
    
    def _retrieve_terms_for_phrase(self, phrase: str) -> Dict[str, Dict[str, float]]:
        """Retrieve terms for a single phrase"""
        term_scores = {}  # term_id -> {signal: score}
        
        # Signal A: Keyword matching
        keyword_results = self._term_keyword_search(phrase)
        for term_id, score in keyword_results:
            if term_id not in term_scores:
                term_scores[term_id] = {}
            term_scores[term_id]['keyword'] = score
        
        # Signal B: Semantic matching
        semantic_results = self._term_semantic_search(phrase)
        for term_id, score in semantic_results:
            if term_id not in term_scores:
                term_scores[term_id] = {}
            term_scores[term_id]['semantic'] = score
        
        return term_scores
    
    def _term_keyword_search(self, query_text: str) -> List[Tuple[str, float]]:
        """Keyword search on terms using BM25"""
        if not self.bm25_index:
            return []
        
        if not query_text or not query_text.strip():
            return []
        
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            
            # Tokenize query
            stop_words = set(stopwords.words('english'))
            query_tokens = word_tokenize(query_text.lower())
            query_tokens = [t for t in query_tokens if t.isalnum() and t not in stop_words]
            
            if not query_tokens:
                return []
            
            # Get BM25 scores
            bm25 = self.bm25_index['bm25']
            term_ids = self.bm25_index['term_ids']
            scores = bm25.get_scores(query_tokens)
            
            # Normalize
            if len(scores) > 0:
                max_score = max(scores)
                if max_score > 0:
                    scores = scores / max_score
            
            # Create results
            results = [(term_ids[i], float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:self.candidate_limit]
            
        except Exception as e:
            logger.error(f"Term keyword search failed: {e}")
            return []
    
    def _term_semantic_search(self, query_text: str) -> List[Tuple[str, float]]:
        """Semantic search on terms using embeddings"""
        if not self.vector_index or not self.embedding_client:
            return []
        
        if not query_text or not query_text.strip():
            return []
        
        semantic_query = query_text
        
        try:
            # Encode query
            query_embedding = self.embedding_client.encode_single(semantic_query)
            
            # Get similarities
            doc_embeddings = self.vector_index['embeddings']
            term_ids = self.vector_index['term_ids']
            
            similarities = self.embedding_client.batch_similarity(
                query_embedding.reshape(1, -1),
                doc_embeddings
            )[0]
            
            # Create results
            results = [(term_ids[i], float(similarities[i])) 
                      for i in range(len(similarities)) if similarities[i] > 0]
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:self.candidate_limit]
            
        except Exception as e:
            logger.error(f"Term semantic search failed: {e}")
            return []
    
    def _score_and_rank_terms(self, candidates: Dict[str, Dict[str, float]]) -> List[TermResult]:
        """Score and rank term candidates"""
        results = []
        
        # Find corresponding documents
        doc_lookup = {doc.term_id: doc for doc in self.term_documents}
        
        for term_id, signal_scores in candidates.items():
            if term_id not in doc_lookup:
                continue
            
            term_doc = doc_lookup[term_id]
            
            # Calculate weighted final score
            final_score = 0.0
            normalized_signals = {}
            
            for signal, weight in self.weights.items():
                score = signal_scores.get(signal, 0.0)
                final_score += weight * score
                normalized_signals[signal] = score
            
            result = TermResult(
                term_document=term_doc,
                final_score=final_score,
                signal_scores=normalized_signals
            )
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index for term documents"""
        from rank_bm25 import BM25Okapi
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            stop_words = set()
        
        # Extract texts and IDs
        texts = []
        term_ids = []
        
        for doc in self.term_documents:
            # Use term_name and definition for indexing (core semantic content)
            search_text = f"Term: {doc.term_name} Definition: {doc.definition}"
            texts.append(search_text)
            term_ids.append(doc.term_id)
        
        if not texts:
            return
        
        # Tokenize
        tokenized_docs = []
        for text in texts:
            try:
                tokens = word_tokenize(text.lower())
                tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
                tokenized_docs.append(tokens)
            except:
                tokenized_docs.append([])
        
        # Build BM25
        if tokenized_docs:
            bm25 = BM25Okapi(tokenized_docs)
            self.bm25_index = {
                'bm25': bm25,
                'term_ids': term_ids,
                'tokenized_docs': tokenized_docs
            }
        
        logger.debug(f"Built BM25 index for {len(texts)} terms")
    
    def _build_vector_index(self) -> None:
        """Build vector index for term documents"""
        if not self.embedding_client:
            logger.warning("No embedding client, skipping term vector index")
            return
        
        # Extract texts and IDs
        texts = []
        term_ids = []
        
        for doc in self.term_documents:
            # Use term_name and definition for indexing (core semantic content)
            search_text = f"Term: {doc.term_name} Definition: {doc.definition}"
            texts.append(search_text)
            term_ids.append(doc.term_id)
        
        if not texts:
            return
        
        try:
            # Compute embeddings
            logger.debug(f"Computing embeddings for {len(texts)} term documents")
            embeddings = self.embedding_client.encode_batch(texts)
            
            self.vector_index = {
                'embeddings': embeddings,
                'term_ids': term_ids,
                'texts': texts
            }
            
            logger.debug(f"Built vector index for {len(texts)} terms")
            
        except Exception as e:
            logger.error(f"Failed to build term vector index: {e}")
    
    def get_term_info(self) -> Dict[str, Any]:
        """Get information about indexed terms"""
        return {
            'total_terms': len(self.term_documents),
            'bm25_ready': self.bm25_index is not None,
            'vector_ready': self.vector_index is not None,
            'sample_terms': [doc.term_name for doc in self.term_documents[:5]]
        }

