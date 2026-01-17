# Guidance Memory Store - Operational guidance from historical error patterns

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime

from .base_store import BaseMemoryStore
from ..types import MemoryQuery, MemoryType, GuidanceItem, GuidanceResponse
from ..types_per_term import FlatMemoryResponse, PerTermMemoryResponse, JoinRelationship
from ..search.engines.episodic.similarity.sql_skeleton import (
    generate_sql_skeleton,
    normalize_sql,
    jaccard_similarity,
    extract_schema_info_from_semantic_memory
)

logger = logging.getLogger(__name__)


class GuidanceMemoryStore(BaseMemoryStore):
    """
    Guidance Memory Store - stores operational guidance from historical error patterns
    
    Key features:
    - Load insights from JSONL file
    - SQL skeleton-based similarity matching
    - Keyword matching with sql_risk_atoms
    - Union Top-K aggregation for multiple SQLs
    - Database-independent (general SQL patterns)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Guidance Memory Store
        
        Args:
            config: Configuration dictionary with guidance settings
        """
        # Extract guidance-specific config
        self.guidance_config = config.get('guidance', {})
        self.insights_path = Path(self.guidance_config.get(
            'insights_path', 
            './output/error_analysis/damo/insights.jsonl'
        ))
        
        # Storage for loaded insights
        self.insights: List[Dict[str, Any]] = []
        
        # Schema info for SQL skeleton generation (extracted from semantic memory)
        self._schema_info: Optional[Dict[str, Any]] = None
        self._memory_base = None  # Set by MemoryBase
        
        super().__init__(config)
        
        logger.info(f"GuidanceMemoryStore initialized with {len(self.insights)} insights")
    
    def _setup_storage(self) -> None:
        """Setup storage by loading insights from JSONL file"""
        try:
            if not self.insights_path.exists():
                logger.warning(f"Insights file not found: {self.insights_path}")
                self.insights = []
                return
            
            # Load insights from JSONL
            self.insights = []
            with open(self.insights_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        insight = json.loads(line)
                        self.insights.append(insight)
            
            logger.info(f"Loaded {len(self.insights)} insights from {self.insights_path}")
            
        except Exception as e:
            logger.error(f"Failed to load insights: {e}")
            self.insights = []
    
    def bind_database(self, database_id: str) -> None:
        """
        Bind to specific database and extract schema info
        
        Args:
            database_id: Database identifier
        """
        super().bind_database(database_id)
        
        # Extract schema info from semantic memory for SQL skeleton generation
        self._extract_schema_info_from_memory()
        
        logger.debug(f"GuidanceMemoryStore bound to database: {database_id}")
    
    def search(
        self, 
        query: MemoryQuery, 
        return_per_term: bool = False
    ) -> Union[GuidanceResponse, FlatMemoryResponse, Tuple[PerTermMemoryResponse, List[JoinRelationship]]]:
        """
        Search for relevant guidance insights based on generated SQL(s)
        
        Args:
            query: Memory query object with generated_sql in context
            return_per_term: Not used for guidance store (ignored)
            
        Returns:
            GuidanceResponse with top-k most relevant insights
        """
        start_time = datetime.now()
        
        if not self.insights:
            logger.warning("No insights loaded in guidance store")
            return GuidanceResponse(
                items=[],
                query_time_ms=0,
                total_insights_searched=0,
                total_sqls_processed=0
            )
        
        # Extract generated SQLs from query
        generated_sqls = self._extract_generated_sqls(query)
        if not generated_sqls:
            logger.warning("No generated SQLs found in query")
            return GuidanceResponse(
                items=[],
                query_time_ms=0,
                total_insights_searched=len(self.insights),
                total_sqls_processed=0
            )
        
        # Get top_k from context or use default
        top_k = query.context.get('top_k', 5) if query.context else 5
        
        # Retrieve top-k insights using Union Top-K strategy
        guidance_items = self._retrieve_union_top_k(generated_sqls, top_k)
        
        end_time = datetime.now()
        query_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        response = GuidanceResponse(
            items=guidance_items,
            query_time_ms=query_time_ms,
            total_insights_searched=len(self.insights),
            total_sqls_processed=len(generated_sqls)
        )
        
        logger.info(f"Guidance search returned {len(guidance_items)} items for {len(generated_sqls)} SQLs")
        return response
    
    def store(self, data: Any) -> None:
        """
        Store new insight (not implemented)
        
        For now, insights are loaded from file only.
        Future: support dynamic addition of insights.
        """
        raise NotImplementedError("Storing new insights is not yet implemented")
    
    def _extract_generated_sqls(self, query: MemoryQuery) -> List[str]:
        """
        Extract generated SQL(s) from query
        
        Supports:
        - Single SQL string in context['generated_sql']
        - List of SQLs in context['generated_sqls']
        - Single SQL in query.generated_sql
        
        Returns:
            List of SQL strings
        """
        sqls = []
        
        if query.context:
            # Check for single SQL
            if 'generated_sql' in query.context and query.context['generated_sql']:
                sql = query.context['generated_sql']
                if isinstance(sql, str):
                    sqls.append(sql)
            
            # Check for multiple SQLs
            if 'generated_sqls' in query.context and query.context['generated_sqls']:
                sql_list = query.context['generated_sqls']
                if isinstance(sql_list, list):
                    sqls.extend([s for s in sql_list if isinstance(s, str)])
        
        # Check query.generated_sql attribute
        if hasattr(query, 'generated_sql') and query.generated_sql:
            if isinstance(query.generated_sql, str):
                sqls.append(query.generated_sql)
            elif isinstance(query.generated_sql, list):
                sqls.extend([s for s in query.generated_sql if isinstance(s, str)])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sqls = []
        for sql in sqls:
            sql_normalized = sql.strip()
            if sql_normalized and sql_normalized not in seen:
                seen.add(sql_normalized)
                unique_sqls.append(sql_normalized)
        
        return unique_sqls
    
    def _retrieve_union_top_k(
        self, 
        generated_sqls: List[str], 
        top_k: int
    ) -> List[GuidanceItem]:
        """
        Union Top-K aggregation strategy
        
        1. For each generated SQL, retrieve top-10 insights
        2. Collect all results with (insight_id, score) pairs
        3. Deduplicate by insight_id (keep highest score)
        4. Sort by score descending
        5. Return top-k
        
        Args:
            generated_sqls: List of generated SQL strings
            top_k: Number of top insights to return
            
        Returns:
            List of GuidanceItem objects
        """
        # Retrieve top-10 for each SQL (more than final top_k for better coverage)
        retrieve_per_sql = max(10, top_k * 2)
        
        all_results: Dict[str, Tuple[Dict[str, Any], float]] = {}  # insight_id -> (insight_data, max_score)
        
        for sql in generated_sqls:
            # Calculate scores for all insights
            scored_insights = []
            for insight in self.insights:
                score = self._calculate_relevance_score(sql, insight)
                if score > 0:  # Only keep insights with positive relevance
                    scored_insights.append((insight, score))
            
            # Sort by score and take top-N
            scored_insights.sort(key=lambda x: x[1], reverse=True)
            top_insights = scored_insights[:retrieve_per_sql]
            
            # Add to all_results, keeping highest score for each insight_id
            for insight, score in top_insights:
                insight_id = insight['insight_id']
                if insight_id not in all_results or score > all_results[insight_id][1]:
                    all_results[insight_id] = (insight, score)
        
        # Sort all results by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to GuidanceItem objects and return top-k
        guidance_items = []
        for insight_data, score in sorted_results[:top_k]:
            item = GuidanceItem(
                insight_id=insight_data['insight_id'],
                relevance_score=round(score, 4),
                retrieval_key=insight_data.get('retrieval_key', {}),
                guidance=insight_data.get('guidance', {}),
                qualified_incorrect_sql=insight_data.get('qualified_incorrect_sql'),
                qualified_correct_sql=insight_data.get('qualified_correct_sql'),
                source_question_ids=insight_data.get('source_question_ids', []),
                verification_success_count=insight_data.get('verification_success_count'),
                verification_total_count=insight_data.get('verification_total_count'),
                verification_success_rate=insight_data.get('verification_success_rate'),
                created_at=insight_data.get('created_at')
            )
            guidance_items.append(item)
        
        return guidance_items
    
    def _calculate_relevance_score(
        self, 
        generated_sql: str, 
        insight: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance score between generated SQL and insight
        
        Score = 0.7 * max_skeleton_similarity + 0.3 * keyword_match_score
        
        Where:
        - max_skeleton_similarity = max(similarity with incorrect_sql, similarity with correct_sql)
        - keyword_match_score = Jaccard similarity between SQL tokens and sql_risk_atoms
        
        Args:
            generated_sql: Generated SQL string
            insight: Insight dictionary with SQL examples and retrieval_key
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # 1. Calculate SQL skeleton similarity
            skeleton_sim = self._calculate_skeleton_similarity(generated_sql, insight)
            
            # 2. Calculate keyword matching score
            keyword_sim = self._calculate_keyword_similarity(generated_sql, insight)
            
            # 3. Weighted combination
            final_score = 0.7 * skeleton_sim + 0.3 * keyword_sim
            
            return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Failed to calculate relevance score: {e}")
            return 0.0
    
    def _calculate_skeleton_similarity(
        self, 
        generated_sql: str, 
        insight: Dict[str, Any]
    ) -> float:
        """
        Calculate SQL skeleton similarity using Jaccard similarity
        
        Compares generated SQL with both incorrect and correct SQLs,
        returns the maximum similarity.
        
        Args:
            generated_sql: Generated SQL string
            insight: Insight dictionary with SQL examples
            
        Returns:
            Maximum skeleton similarity score
        """
        try:
            # Generate skeleton for generated SQL
            generated_skeleton = self._generate_skeleton_safe(generated_sql)
            
            similarities = []
            
            # Compare with incorrect SQL
            if insight.get('qualified_incorrect_sql'):
                incorrect_skeleton = self._generate_skeleton_safe(
                    insight['qualified_incorrect_sql']
                )
                sim = jaccard_similarity(generated_skeleton, incorrect_skeleton)
                similarities.append(sim)
            
            # Compare with correct SQL
            if insight.get('qualified_correct_sql'):
                correct_skeleton = self._generate_skeleton_safe(
                    insight['qualified_correct_sql']
                )
                sim = jaccard_similarity(generated_skeleton, correct_skeleton)
                similarities.append(sim)
            
            # Return maximum similarity
            return max(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.debug(f"SQL skeleton generation failed, falling back to token similarity: {e}")
            return self._calculate_token_similarity(generated_sql, insight)
    
    def _generate_skeleton_safe(self, sql: str) -> str:
        """
        Safely generate SQL skeleton with fallback mechanism
        
        If schema_info is available, use full skeleton generation.
        Otherwise, use a simplified approach that masks identifiers.
        
        Args:
            sql: SQL string
            
        Returns:
            SQL skeleton string
        """
        if self._schema_info:
            # Full skeleton generation with schema info
            return generate_sql_skeleton(sql, self._schema_info)
        else:
            # Fallback: simplified skeleton without schema info
            # This will mask qualified names (table.column) and values,
            # but keep unqualified identifiers (which could be columns or tables)
            return self._generate_simplified_skeleton(sql)
    
    def _generate_simplified_skeleton(self, sql: str) -> str:
        """
        Generate simplified SQL skeleton without schema info
        
        This method masks:
        - Qualified identifiers (table.column -> _)
        - Unqualified identifiers that are not SQL keywords -> _
        - String values ('value' -> _)
        - Numeric values (123 -> _)
        
        But keeps SQL keywords and structure.
        
        Args:
            sql: SQL string
            
        Returns:
            Simplified skeleton
        """
        try:
            from sql_metadata import Parser
            
            # SQL keywords (common ones)
            SQL_KEYWORDS = {
                'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'outer',
                'on', 'and', 'or', 'not', 'in', 'between', 'like', 'is', 'null',
                'group', 'by', 'having', 'order', 'asc', 'desc', 'limit', 'offset',
                'union', 'intersect', 'except', 'as', 'distinct', 'all', 'exists',
                'case', 'when', 'then', 'else', 'end', 'cast', 'count', 'sum', 'avg',
                'max', 'min', 'over', 'partition', 'window', 'with', 'recursive'
            }
            
            # Operators
            OPERATORS = {'=', '!=', '<>', '>', '<', '>=', '<=', '+', '-', '*', '/', '%'}
            
            # Normalize SQL first
            normalized = normalize_sql(sql)
            
            # Parse into tokens
            parsed = Parser(normalized)
            new_tokens = []
            
            for token in parsed.tokens:
                token_value = token.value
                token_lower = token_value.lower()
                
                # Keep SQL keywords
                if token_lower in SQL_KEYWORDS:
                    new_tokens.append(token_lower)
                # Keep operators
                elif token_value in OPERATORS:
                    new_tokens.append(token_value)
                # Keep special symbols
                elif token_value in ('(', ')', ',', ';', '*'):
                    new_tokens.append(token_value)
                # Mask qualified identifiers (e.g., table.column)
                elif '.' in token_value and not token_value.replace('.', '').replace('-', '').isdigit():
                    new_tokens.append('_')
                # Mask string values
                elif token_value.startswith("'") and token_value.endswith("'"):
                    new_tokens.append('_')
                # Mask numeric values (integers)
                elif token_value.isdigit() or (token_value.startswith('-') and token_value[1:].isdigit()):
                    new_tokens.append('_')
                # Mask floats
                elif self._is_float(token_value):
                    new_tokens.append('_')
                # Mask all other identifiers (likely table/column names)
                else:
                    new_tokens.append('_')
            
            skeleton = ' '.join(new_tokens)
            
            # Apply basic cleanup
            skeleton = self._cleanup_skeleton_basic(skeleton)
            
            return skeleton
            
        except Exception as e:
            logger.debug(f"Simplified skeleton generation failed: {e}")
            # Ultimate fallback: just normalize
            return normalize_sql(sql)
    
    @staticmethod
    def _is_float(s: str) -> bool:
        """Check if string represents a float"""
        if s.startswith('-'):
            s = s[1:]
        parts = s.split('.')
        return len(parts) == 2 and all(p.isdigit() for p in parts)
    
    @staticmethod
    def _cleanup_skeleton_basic(skeleton: str) -> str:
        """Basic cleanup for simplified skeleton"""
        # Collapse comma-separated placeholders
        while '_ , _' in skeleton:
            skeleton = skeleton.replace('_ , _', '_')
        
        # Remove comparison operators
        for op in ['=', '!=', '>', '>=', '<', '<=']:
            pattern = f'_ {op} _'
            if pattern in skeleton:
                skeleton = skeleton.replace(pattern, '_')
        
        # Clean up WHERE conditions
        while 'where _ and _' in skeleton or 'where _ or _' in skeleton:
            skeleton = skeleton.replace('where _ and _', 'where _')
            skeleton = skeleton.replace('where _ or _', 'where _')
        
        # Remove extra spaces
        while '  ' in skeleton:
            skeleton = skeleton.replace('  ', ' ')
        
        return skeleton.strip()
    
    def _calculate_token_similarity(
        self, 
        generated_sql: str, 
        insight: Dict[str, Any]
    ) -> float:
        """
        Fallback: Calculate token-based similarity using normalized SQLs
        
        Args:
            generated_sql: Generated SQL string
            insight: Insight dictionary
            
        Returns:
            Token similarity score
        """
        try:
            generated_normalized = normalize_sql(generated_sql)
            generated_tokens = set(generated_normalized.split())
            
            similarities = []
            
            # Compare with incorrect SQL
            if insight.get('qualified_incorrect_sql'):
                incorrect_normalized = normalize_sql(insight['qualified_incorrect_sql'])
                incorrect_tokens = set(incorrect_normalized.split())
                sim = self._jaccard_set_similarity(generated_tokens, incorrect_tokens)
                similarities.append(sim)
            
            # Compare with correct SQL
            if insight.get('qualified_correct_sql'):
                correct_normalized = normalize_sql(insight['qualified_correct_sql'])
                correct_tokens = set(correct_normalized.split())
                sim = self._jaccard_set_similarity(generated_tokens, correct_tokens)
                similarities.append(sim)
            
            return max(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Token similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_keyword_similarity(
        self, 
        generated_sql: str, 
        insight: Dict[str, Any]
    ) -> float:
        """
        Calculate keyword matching score with sql_risk_atoms
        
        Extracts tokens from generated SQL skeleton (or normalized SQL)
        and compares with sql_risk_atoms using Jaccard similarity.
        
        Args:
            generated_sql: Generated SQL string
            insight: Insight dictionary with retrieval_key
            
        Returns:
            Keyword matching score
        """
        try:
            # Extract SQL tokens (keywords)
            if self._schema_info:
                skeleton = generate_sql_skeleton(generated_sql, self._schema_info)
                sql_tokens = set(skeleton.split()) - {'_'}  # Remove placeholders
            else:
                normalized = normalize_sql(generated_sql)
                sql_tokens = set(normalized.split())
            
            # Get sql_risk_atoms from insight
            retrieval_key = insight.get('retrieval_key', {})
            risk_atoms = retrieval_key.get('sql_risk_atoms', [])
            
            if not risk_atoms:
                return 0.0
            
            # Normalize risk atoms to lowercase
            risk_atoms_set = {atom.lower().strip() for atom in risk_atoms}
            
            # Calculate Jaccard similarity
            return self._jaccard_set_similarity(sql_tokens, risk_atoms_set)
            
        except Exception as e:
            logger.warning(f"Keyword similarity calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def _jaccard_set_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_schema_info_from_memory(self) -> None:
        """Extract schema info from semantic memory for SQL skeleton generation"""
        if not self._memory_base:
            logger.debug("No memory_base reference, cannot extract schema info")
            return
        
        try:
            semantic_store = self._memory_base.memory_stores.get(MemoryType.SEMANTIC)
            if not semantic_store:
                logger.warning("No semantic memory store available")
                return
            
            self._schema_info = extract_schema_info_from_semantic_memory(semantic_store)
            if self._schema_info:
                logger.debug("Extracted schema info for SQL skeleton generation")
            else:
                logger.debug("No schema info available from semantic memory")
                
        except Exception as e:
            logger.warning(f"Failed to extract schema info: {e}")
            self._schema_info = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the guidance store"""
        return {
            'total_insights': len(self.insights),
            'insights_path': str(self.insights_path),
            'schema_info_available': self._schema_info is not None,
            'current_database_id': self.current_database_id
        }
