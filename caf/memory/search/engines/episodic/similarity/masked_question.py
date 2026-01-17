# Masked Question Processor with Schema Linking (DAIL-SQL implementation)

import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
import string
from caf.memory.types import MemoryType

logger = logging.getLogger(__name__)

class MaskedQuestionProcessor:
    """
    Implements DAIL-SQL's masked question approach with schema linking
    
    Features:
    - Schema linking to identify table/column references in questions
    - Value detection and masking (numbers, dates, strings) 
    - Configurable mask tokens
    - Caching for performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('masked_question', {})
        
        # Mask tokens (following DAIL-SQL)
        self.mask_token = self.config.get('mask_token', '<mask>')  # For schema elements
        self.value_token = self.config.get('value_token', '<unk>')  # For values
        
        # Configuration
        self.enable_schema_linking = self.config.get('enable_schema_linking', True)
        self.enable_value_masking = self.config.get('enable_value_masking', True)
        self.case_sensitive = self.config.get('case_sensitive', False)
        
        # Database metadata (set when bound to database)
        self.current_database_id: Optional[str] = None
        self.table_names: Set[str] = set()
        self.column_names: Set[str] = set()
        self.table_column_map: Dict[str, Set[str]] = {}  # table -> columns
        
        # Semantic store reference (set by parent component)
        self._semantic_store = None
        self._memory_base = None
        
        # Compiled regex patterns for performance
        self._compile_patterns()
        
        # Cache for processed questions
        self._mask_cache: Dict[str, str] = {}
        
        logger.debug("MaskedQuestionProcessor initialized")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for value detection"""
        # Number patterns (integers, floats, percentages)
        self.number_patterns = [
            re.compile(r'\b\d+\.?\d*\b'),  # Basic numbers
            re.compile(r'\b\d+%\b'),       # Percentages
            re.compile(r'\$\d+(?:\.\d+)?\b'),  # Currency
        ]
        
        # Date patterns  
        self.date_patterns = [
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),    # YYYY-MM-DD
            re.compile(r'\b\d{2}/\d{2}/\d{4}\b'),    # MM/DD/YYYY
            re.compile(r'\b\d{2}-\d{2}-\d{4}\b'),    # MM-DD-YYYY
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),  # M/D/YY or MM/DD/YYYY
        ]
        
        # String literal patterns
        self.string_patterns = [
            re.compile(r"'[^']*'"),  # Single quoted strings
            re.compile(r'"[^"]*"'),  # Double quoted strings
        ]
    
    def bind_database(self, database_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Bind to database and load metadata for schema linking
        
        Args:
            database_id: Database identifier
            metadata: Optional metadata dict containing table/column info
        """
        self.current_database_id = database_id
        
        # Load database metadata if provided
        if metadata:
            self._load_metadata(metadata)
        else:
            # Try to load from semantic memory store if available
            self._load_metadata_from_semantic_store()
        
        # Clear cache when binding to new database
        self._mask_cache.clear()
        
        logger.debug(f"MaskedQuestionProcessor bound to database: {database_id}")
        logger.debug(f"Loaded {len(self.table_names)} tables, {len(self.column_names)} columns")
    
    def _load_metadata(self, metadata: Dict[str, Any]) -> None:
        """Load metadata from provided dict"""
        self.table_names.clear()
        self.column_names.clear()
        self.table_column_map.clear()
        
        # Extract table and column names
        if 'tables' in metadata:
            for table_info in metadata['tables']:
                if isinstance(table_info, dict):
                    table_name = table_info.get('table_name', '')
                else:
                    table_name = str(table_info)
                
                if table_name:
                    self.table_names.add(table_name.lower() if not self.case_sensitive else table_name)
        
        if 'columns' in metadata:
            for column_info in metadata['columns']:
                if isinstance(column_info, dict):
                    table_name = column_info.get('table_name', '')
                    column_name = column_info.get('column_name', '')
                else:
                    # Assume format "table.column"
                    parts = str(column_info).split('.')
                    if len(parts) >= 2:
                        table_name, column_name = parts[-2], parts[-1]
                    else:
                        table_name, column_name = '', str(column_info)
                
                if table_name and column_name:
                    norm_table = table_name.lower() if not self.case_sensitive else table_name
                    norm_column = column_name.lower() if not self.case_sensitive else column_name
                    
                    self.column_names.add(norm_column)
                    
                    if norm_table not in self.table_column_map:
                        self.table_column_map[norm_table] = set()
                    self.table_column_map[norm_table].add(norm_column)
    
    def set_semantic_store(self, semantic_store) -> None:
        """Set semantic memory store reference for loading metadata"""
        self._semantic_store = semantic_store
        logger.debug("Semantic store reference set for MaskedQuestionProcessor")
    
    def set_memory_base(self, memory_base) -> None:
        """Set memory base reference for accessing semantic store"""
        self._memory_base = memory_base
        # Try to get semantic store from memory base
        if memory_base and hasattr(memory_base, 'memory_stores'):
            semantic_store = memory_base.memory_stores.get(MemoryType.SEMANTIC)
            if semantic_store:
                self.set_semantic_store(semantic_store)
        logger.debug("Memory base reference set for MaskedQuestionProcessor")
    
    def _load_metadata_from_semantic_store(self) -> None:
        """Load metadata from semantic memory store"""
        self.table_names.clear()
        self.column_names.clear() 
        self.table_column_map.clear()
        
        # Try to get semantic store
        semantic_store = self._semantic_store
        if not semantic_store and self._memory_base:
            from ....types import MemoryType
            semantic_store = self._memory_base.memory_stores.get(MemoryType.SEMANTIC) if hasattr(self._memory_base, 'memory_stores') else None
        
        if not semantic_store:
            logger.debug("No semantic store available, using empty metadata")
            return
        
        try:
            # Check if semantic store has dataframes
            if not hasattr(semantic_store, 'dataframes') or not semantic_store.dataframes:
                logger.debug("Semantic store dataframes not available")
                return
            
            # Load table names
            if 'table' in semantic_store.dataframes and not semantic_store.dataframes['table'].empty:
                table_df = semantic_store.dataframes['table']
                if 'table_name' in table_df.columns:
                    for table_name in table_df['table_name'].dropna().unique():
                        if table_name:
                            norm_name = table_name.lower() if not self.case_sensitive else table_name
                            self.table_names.add(norm_name)
            
            # Load column names
            if 'column' in semantic_store.dataframes and not semantic_store.dataframes['column'].empty:
                column_df = semantic_store.dataframes['column']
                if 'table_name' in column_df.columns and 'column_name' in column_df.columns:
                    for _, row in column_df.iterrows():
                        table_name = row.get('table_name')
                        column_name = row.get('column_name')
                        
                        if table_name and column_name:
                            norm_table = table_name.lower() if not self.case_sensitive else table_name
                            norm_column = column_name.lower() if not self.case_sensitive else column_name
                            
                            self.column_names.add(norm_column)
                            
                            if norm_table not in self.table_column_map:
                                self.table_column_map[norm_table] = set()
                            self.table_column_map[norm_table].add(norm_column)
            
            logger.debug(f"Loaded metadata from semantic store: {len(self.table_names)} tables, {len(self.column_names)} columns")
            
        except Exception as e:
            logger.warning(f"Failed to load metadata from semantic store: {e}")
            # Continue with empty metadata
    
    def mask_question(self, question: str) -> str:
        """
        Apply masking to question based on schema linking and value detection
        
        Args:
            question: Original question text
            
        Returns:
            Masked question text
        """
        if not question.strip():
            return question
        
        # Check cache first
        cache_key = self._get_cache_key(question)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        masked_question = question
        
        # Apply masking steps
        if self.enable_value_masking:
            masked_question = self._mask_values(masked_question)
        
        if self.enable_schema_linking and (self.table_names or self.column_names):
            masked_question = self._mask_schema_elements(masked_question)
        
        # Cache result
        self._mask_cache[cache_key] = masked_question
        
        return masked_question
    
    def _mask_values(self, text: str) -> str:
        """Mask numerical values, dates, and string literals"""
        masked_text = text
        
        # Mask string literals first (to avoid masking numbers within strings)
        for pattern in self.string_patterns:
            masked_text = pattern.sub(self.value_token, masked_text)
        
        # Mask dates
        for pattern in self.date_patterns:
            masked_text = pattern.sub(self.value_token, masked_text)
        
        # Mask numbers
        for pattern in self.number_patterns:
            masked_text = pattern.sub(self.value_token, masked_text)
        
        return masked_text
    
    def _mask_schema_elements(self, text: str) -> str:
        """Mask table and column references using schema linking"""
        masked_text = text
        tokens = self._tokenize_question(text)
        
        # Find schema matches
        schema_matches = self._find_schema_matches(tokens)
        
        # Apply masking in reverse order to preserve indices
        for start_idx, end_idx, match_type in reversed(schema_matches):
            # Replace matched tokens with mask
            before_tokens = tokens[:start_idx]
            after_tokens = tokens[end_idx:]
            
            # Reconstruct text with masking
            before_text = ' '.join(before_tokens) if before_tokens else ''
            after_text = ' '.join(after_tokens) if after_tokens else ''
            
            if before_text and after_text:
                masked_text = f"{before_text} {self.mask_token} {after_text}"
            elif before_text:
                masked_text = f"{before_text} {self.mask_token}"
            elif after_text:
                masked_text = f"{self.mask_token} {after_text}"
            else:
                masked_text = self.mask_token
        
        return masked_text
    
    def _tokenize_question(self, text: str) -> List[str]:
        """Tokenize question for schema linking"""
        # Simple whitespace tokenization with punctuation handling
        # Remove punctuation and split on whitespace
        translator = str.maketrans('', '', string.punctuation)
        cleaned_text = text.translate(translator)
        tokens = cleaned_text.split()
        return tokens
    
    def _find_schema_matches(self, tokens: List[str]) -> List[Tuple[int, int, str]]:
        """
        Find schema element matches in tokenized question
        
        Returns list of (start_idx, end_idx, match_type) tuples
        """
        matches = []
        
        # Normalize tokens for matching
        norm_tokens = [token.lower() if not self.case_sensitive else token for token in tokens]
        
        # Single token matches
        for i, token in enumerate(norm_tokens):
            # Check table names
            if token in self.table_names:
                matches.append((i, i + 1, 'table'))
            # Check column names
            elif token in self.column_names:
                matches.append((i, i + 1, 'column'))
        
        # Multi-token matches (e.g., "customer id" -> "customer_id")
        for i in range(len(norm_tokens) - 1):
            # Check 2-token combinations
            combined = '_'.join(norm_tokens[i:i+2])
            if combined in self.column_names or combined in self.table_names:
                match_type = 'table' if combined in self.table_names else 'column'
                matches.append((i, i + 2, match_type))
        
        # Remove overlapping matches (keep longer matches)
        matches = self._remove_overlapping_matches(matches)
        
        return matches
    
    def _remove_overlapping_matches(self, matches: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """Remove overlapping matches, keeping longer ones"""
        if not matches:
            return matches
        
        # Sort by start position, then by length (descending)
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        
        filtered_matches = []
        last_end = -1
        
        for start, end, match_type in matches:
            if start >= last_end:  # No overlap
                filtered_matches.append((start, end, match_type))
                last_end = end
        
        return filtered_matches
    
    def _get_cache_key(self, question: str) -> str:
        """Generate cache key for question"""
        normalized = ' '.join(question.strip().split())
        return f"{self.current_database_id}_{hash(normalized)}"
    
    def reset(self) -> None:
        """Reset processor state"""
        self._mask_cache.clear()
        logger.debug("MaskedQuestionProcessor reset")
    
    def get_cache_size(self) -> int:
        """Get current cache size"""
        return len(self._mask_cache)
    
    def get_metadata_info(self) -> Dict[str, Any]:
        """Get information about loaded metadata"""
        return {
            'database_id': self.current_database_id,
            'table_count': len(self.table_names),
            'column_count': len(self.column_names),
            'cache_size': self.get_cache_size()
        }
