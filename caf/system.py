# CAF System - Unified cognitive augmentation system

import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime
from pathlib import Path
from caf.memory.generators import BirdMetadataExtractor
        

from .config.loader import CAFConfig
from caf.memory.base import MemoryBase
from caf.memory.types import (
    MemoryQuery, MemoryResponse, EpisodicRecord, SQLExecutionResult, 
    MemoryType, GoldenPair, SchemaMetadataResponse, SQLInsightsResponse,
    GuidanceResponse
)
from caf.memory.types_per_term import FlatMemoryResponse, PerTermMemoryResponse, JoinRelationship
from caf.user.manager import UserFeedbackManager
from caf.user.types import FeedbackContext, UserFeedback, FeedbackType
from caf.exceptions import CAFSystemError
from caf.llm.client import LLMConfig, create_llm_client
from caf.utils.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)

class CAFSystem:
    """
    CAF Cognitive Augmentation System - Unified external interface
    
    This is the main entry point for the CAF framework, providing unified access
    to memory and user feedback capabilities for NL2SQL reasoning systems.
    """
    
    def __init__(self, config: CAFConfig, config_path: Optional[str] = None):
        """
        Initialize the entire CAF system
        Contains tightly integrated Memory and User Feedback components
        
        Args:
            config: CAF configuration object
            config_path: Optional path to config file for global config initialization.
                        If not provided, will try to extract from config object.
        """
        self.config = config
        
        # Initialize global configuration for LLM providers
        # This ensures GlobalConfigManager is properly initialized before any LLM calls
        self._initialize_global_config(config_path)
        
        # Initialize core components - internal tight integration
        self._memory_base = MemoryBase(config)
        # self._feedback_manager = UserFeedbackManager(config.feedback, self._memory_base)
        
        # Session and Database management
        self._current_session_id: Optional[str] = None
        self._current_database_id: Optional[str] = None
        self._session_context: Dict[str, Dict[str, Any]] = {}  # session_id -> {database_id, start_time, temp_feedbacks}
        
        # Temporary feedback storage (before session completion)
        self._temp_feedback_storage: List[Dict[str, Any]] = []
        
        logger.info("CAFSystem initialized successfully")
    
    def _initialize_global_config(self, config_path: Optional[str] = None) -> None:
        """
        Initialize global configuration manager for LLM providers.
        
        This method ensures GlobalConfigManager is properly initialized before any LLM calls.
        It checks if global config is already initialized to avoid redundant initialization.
        
        Args:
            config_path: Optional path to config file. If not provided, will try to get from config object.
        """
        from .config.global_config import get_global_config, initialize_global_config
        
        # Check if global config is already initialized with proper config
        try:
            global_config = get_global_config()
            # Check if it has any config sources (file or environment)
            if global_config._config_sources:
                logger.debug("Global config already initialized, skipping re-initialization")
                return
        except Exception:
            pass  # Not initialized yet, continue with initialization
        
        # Try to get config_path from CAFConfig object if not provided
        if not config_path and hasattr(self.config, '_config_path'):
            config_path = self.config._config_path
        
        # Initialize global config
        if config_path:
            # If config_path is available, use it directly (most efficient)
            initialize_global_config(config_path)
            logger.debug(f"Initialized global config from path: {config_path}")
        else:
            # If no config_path, initialize with config data from CAFConfig
            # This ensures LLM providers can access configuration through global config
            from .config.global_config import GlobalConfigManager
            manager = GlobalConfigManager()
            manager.load_from_env()
            # Load config data from CAFConfig._raw_data
            if hasattr(self.config, '_raw_data'):
                manager._config_sources['file'] = self.config._raw_data
            logger.debug("Initialized global config from CAFConfig object")
    
    def bind_database(self, database_id: str) -> None:
        """
        Bind database to the system
        This should be called before starting a session to select the target database
        
        Args:
            database_id: Database identifier to bind
        """
        if self._current_database_id != database_id:

            self._current_database_id = database_id
            
            # Bind database through MemoryBase unified management (Observer pattern)
            self._memory_base.bind_database(database_id)
            
            logger.info(f"Database bound: {database_id}")
    
    def start_session(self, session_id: str) -> None:
        """
        Start new session
        Database must be bound first using bind_database()
        
        Args:
            session_id: Session identifier
        """
        if self._current_database_id is None:
            raise CAFSystemError("DATABASE_NOT_BOUND", "Please bind database first using bind_database() before starting session")
        
        self._current_session_id = session_id
        
        # Create session context
        self._session_context[session_id] = {
            'database_id': self._current_database_id,
            'start_time': datetime.utcnow().isoformat(),
            'temp_feedbacks': [],
            'round_count': 0  # Track number of rounds stored
        }
        
        # Clear previous temporary storage
        self._temp_feedback_storage.clear()
        
        logger.info(f"Session started: {session_id} with database: {self._current_database_id}")
    
    def retrieve_memory(self, 
        memory_type: str,
        query_content: str,
        context: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 20,
        similarity_threshold: Optional[float] = None,
        generated_sql: Optional[str] = None,
        return_per_term: bool = False) -> Union[FlatMemoryResponse, Tuple[PerTermMemoryResponse, List[JoinRelationship]]]:
        """
        Read from memory system (automatically limited to current database)
        
        Args:
            memory_type: Type of memory ("semantic", "episodic")
            query_content: Query content (natural language question)
            context: Additional context (optional)
            limit: Maximum number of results (default: 20)
            similarity_threshold: Minimum similarity threshold for results (optional)
            generated_sql: Pre-generated SQL query (optional). If not provided, will generate automatically.
            return_per_term: If True, return results grouped by query term (default: False)

        Returns:
            - FlatMemoryResponse if return_per_term=False: Contains column_items, term_items, 
              all_tables, all_schemas, join_relationships
            - Tuple[PerTermMemoryResponse, List[JoinRelationship]] if return_per_term=True: 
              Results grouped by query term plus global join relationships
        """
        if self._current_database_id is None:
            raise CAFSystemError("DATABASE_NOT_BOUND", "Please bind database first using bind_database()")
        
        try:
            # If generated_sql is not provided, generate it automatically
            if generated_sql is None:
                try:
                    generated_sql = self._generate_initial_sql(query_content, context)
                    logger.info(f"Auto-generated initial SQL: {generated_sql}")
                except Exception as e:
                    logger.warning(f"Failed to auto-generate SQL: {e}. Continuing without generated_sql.")
                    # Continue without generated_sql - this is not a critical error
            
            # Initialize QueryAnalyzer
            llm_config = self._get_default_llm_config()
            if not llm_config:
                raise CAFSystemError("LLM_NOT_CONFIGURED", 
                                   "Cannot analyze query: LLM configuration not available. "
                                   "Please configure LLM in CAF config.")
            
            llm_client = create_llm_client(llm_config)
            
            # Get query analyzer config from memory config if available
            query_analyzer_config = {}
            if hasattr(self.config, 'memory') and isinstance(self.config.memory, dict):
                query_analyzer_config = self.config.memory.get('query_analyzer', {})
            
            query_analyzer = QueryAnalyzer(
                llm_client=llm_client,
                config=query_analyzer_config
            )
            
            intent_analysis = query_analyzer.analyze(query_content)

            # Create MemoryQuery object internally - users don't need to know about it
            memory_type_enum = MemoryType(memory_type)
            query = MemoryQuery(
                memory_type=memory_type_enum,
                query_content=query_content,
                context=context,
                limit=limit,
                similarity_threshold=similarity_threshold,
                generated_sql=generated_sql,
                intent_analysis=intent_analysis
            )
            
            response = self._memory_base.query(query, return_per_term=return_per_term)
            
            return response

        except ValueError as e:
            # Handle invalid memory_type
            valid_types = [t.value for t in MemoryType]
            raise CAFSystemError("INVALID_MEMORY_TYPE", f"Invalid memory type '{memory_type}'. Valid types: {valid_types}")
        except Exception as e:
            logger.error(f"Memory query failed: {e}")
            raise CAFSystemError("MEMORY_QUERY_FAILED", f"Memory query failed: {e}")
    
    def get_schema_metadata(
        self,
        relevant_schema: Dict[str, Optional[List[str]]],
        include_joins: bool = True,
        include_fields: Optional[List[str]] = None
    ) -> SchemaMetadataResponse:
        """
        Get metadata for already-selected schema (from reasoning module's schema linking)
        
        This API is designed for scenarios where the reasoning module has already performed
        schema linking and obtained relevant tables/columns. It simply fetches the metadata
        (descriptions, joins, etc.) without performing retrieval.
        
        Args:
            relevant_schema: Dict mapping table_name to list of column_names
                           Format: {'table_name': ['col1', 'col2', ...]}
                           If column list is None for a table, returns all columns for that table
            include_joins: Whether to include join relationships between the given tables
                         Default: True
            include_fields: List of metadata field names to include for columns
                          If None, defaults to ['long_description']
                          Available fields include:
                          - 'long_description': Detailed description with statistics
                          - 'short_description': Concise semantic summary
                          - 'description': General description
                          - 'data_type': Database data type
                          - 'semantic_type': Inferred semantic type
                          - 'encoding_mapping': Value encoding information
                          - 'semantic_tags': Business rules and tags
                          - 'top_k_values': Most common values
                          - 'pattern_description': Pattern information
                          - 'min_value', 'max_value': Value ranges
                          - etc. (see ColumnMetadata for full list)
        
        Returns:
            SchemaMetadataResponse with:
            - tables: Dict[table_name, table_metadata]
              - Each table_metadata includes: description, row_definition, and other table fields
            - columns: Dict["table.column", column_metadata]
              - Column metadata includes requested fields (e.g., long_description, data_type, etc.)
            - joins: List[join_relationship_dicts]
              - Each join dict includes: source_table, target_table, source_columns, target_columns,
                relationship_type, cardinality, and business_meaning
            - query_time_ms: Query time in milliseconds
        
        Example:
            >>> relevant_schema = {
            ...     'orders': ['order_id', 'customer_id', 'order_date'],
            ...     'customers': ['customer_id', 'name'],
            ...     'products': None  # All columns
            ... }
            >>> response = caf_system.get_schema_metadata(
            ...     relevant_schema=relevant_schema,
            ...     include_joins=True,
            ...     include_fields=['long_description', 'data_type', 'semantic_type']
            ... )
            >>> print(response.columns['orders.order_id'])
            {'table_name': 'orders', 'column_name': 'order_id', 
             'long_description': '...', 'data_type': 'INTEGER'}
        """
        if self._current_database_id is None:
            raise CAFSystemError("DATABASE_NOT_BOUND", "Please bind database first using bind_database()")
        
        import time
        start_time = time.time()
        
        try:
            # Get semantic memory store
            semantic_store = self._memory_base.get_memory_store(MemoryType.SEMANTIC)
            if semantic_store is None:
                raise CAFSystemError("SEMANTIC_STORE_NOT_AVAILABLE", 
                                   "Semantic memory store is not available")
            
            # Get schema metadata (tables and columns)
            result = semantic_store.get_schema_metadata_for_tables(
                relevant_schema=relevant_schema,
                include_fields=include_fields
            )
            
            # Get join relationships if requested
            if include_joins:
                table_names = list(relevant_schema.keys())
                joins = semantic_store.get_direct_joins_for_tables(table_names)
                result['joins'] = joins
            else:
                result['joins'] = []
            
            # Calculate query time
            query_time_ms = int((time.time() - start_time) * 1000)
            
            # Build response
            response = SchemaMetadataResponse(
                tables=result['tables'],
                columns=result['columns'],
                joins=result['joins'],
                query_time_ms=query_time_ms
            )
            
            logger.debug(f"Schema metadata retrieved: {len(result['tables'])} tables, "
                        f"{len(result['columns'])} columns, {len(result['joins'])} joins")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get schema metadata: {e}")
            raise CAFSystemError("SCHEMA_METADATA_FAILED", f"Failed to get schema metadata: {e}")
    
    def retrieve_sql_guidance(
        self,
        generated_sqls: Union[str, List[str]],
        top_k: int = 5,
        natural_question: Optional[str] = None
    ) -> GuidanceResponse:
        """
        Retrieve operational guidance based on generated SQL(s)
        
        This method retrieves relevant insights from historical error patterns
        to help identify potential issues in generated SQL queries. The insights
        are based on SQL skeleton similarity and keyword matching.
        
        Args:
            generated_sqls: Single SQL string or list of SQL strings to analyze
            top_k: Number of top insights to return (default: 5)
            natural_question: Optional natural language question for enhanced matching
                            (currently not used, reserved for future enhancement)
        
        Returns:
            GuidanceResponse containing:
            - items: List of GuidanceItem objects with relevance scores
            - query_time_ms: Query execution time
            - total_insights_searched: Total number of insights in the store
            - total_sqls_processed: Number of SQLs processed
        
        Raises:
            CAFSystemError: If database is not bound or guidance store is not available
        
        Example:
            >>> # Single SQL
            >>> response = caf_system.retrieve_sql_guidance(
            ...     generated_sqls="SELECT * FROM orders WHERE price = (SELECT MAX(price) FROM orders)"
            ... )
            >>> for item in response.items:
            ...     print(f"Score: {item.relevance_score}")
            ...     print(f"Advice: {item.guidance['actionable_advice']}")
            
            >>> # Multiple SQLs
            >>> response = caf_system.retrieve_sql_guidance(
            ...     generated_sqls=[
            ...         "SELECT * FROM t1 JOIN t2 WHERE ...",
            ...         "SELECT COUNT(*) FROM t1 GROUP BY ..."
            ...     ],
            ...     top_k=3
            ... )
        """
        if self._current_database_id is None:
            raise CAFSystemError("DATABASE_NOT_BOUND", 
                               "Please bind database first using bind_database()")
        
        try:
            # Normalize input to list
            if isinstance(generated_sqls, str):
                sql_list = [generated_sqls]
            else:
                sql_list = generated_sqls
            
            # Build query for guidance memory
            query = MemoryQuery(
                memory_type=MemoryType.GUIDANCE,
                query_content="",  # Not used for guidance
                context={
                    'generated_sqls': sql_list,
                    'top_k': top_k,
                    'natural_question': natural_question
                }
            )
            
            # Query guidance memory store
            response = self._memory_base.query(query)
            
            if not isinstance(response, GuidanceResponse):
                raise CAFSystemError("INVALID_RESPONSE", 
                                   "Guidance store returned unexpected response type")
            
            logger.info(f"Retrieved {len(response.items)} guidance items for {len(sql_list)} SQL(s)")
            return response
            
        except Exception as e:
            logger.error(f"Failed to retrieve SQL guidance: {e}")
            raise CAFSystemError("GUIDANCE_RETRIEVAL_FAILED", 
                               f"Failed to retrieve SQL guidance: {e}")
    
    def identify_mapping(self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Identify all possible NL Term -> Mapping for a given query
        
        Args:
            query: Natural language query/question
        """
        if self._current_database_id is None:
            raise CAFSystemError("DATABASE_NOT_BOUND", "Please bind database first using bind_database()")
        
        # Default to semantic memory only if not specified
        memory_types = ["semantic"] 

        try:
            generated_sql = self._generate_initial_sql(query, context)
            logger.info(f"Auto-generated initial SQL: {generated_sql}")
        except Exception as e:
            generated_sql = None
            logger.warning(f"Failed to auto-generate SQL: {e}. Continuing without generated_sql.")

        # Initialize QueryAnalyzer
        llm_config = self._get_default_llm_config()
        if not llm_config:
            raise CAFSystemError("LLM_NOT_CONFIGURED", 
                                "Cannot analyze query: LLM configuration not available. "
                                "Please configure LLM in CAF config.")
        
        llm_client = create_llm_client(llm_config)
        
        # Get query analyzer config from memory config if available
        query_analyzer_config = {}
        if hasattr(self.config, 'memory') and isinstance(self.config.memory, dict):
            query_analyzer_config = self.config.memory.get('query_analyzer', {})
        
        query_analyzer = QueryAnalyzer(
            llm_client=llm_client,
            config=query_analyzer_config
        )
        
        intent_analysis = query_analyzer.analyze(query)

        print('**************Intent Analysis*****************')
        print(intent_analysis)
        
        for memory_type in memory_types:
            memory_type_enum = MemoryType(memory_type)
            query = MemoryQuery(
                memory_type=memory_type_enum,
                query_content=query,
                context=context,
                generated_sql=generated_sql,
                intent_analysis=intent_analysis
            )
            
            response = self._memory_base.query(query, return_per_term=True)
            
            print('**************Memory Query Completed: {memory_type}*****************')
            print(response)
        
        return result
            
            
    # def request_feedback(self,
    #     feedback_type: str,
    #     user_query: str,
    #     generated_sql: str,
    #     execution_result: Optional[Dict] = None,
    #     options: Optional[List[str]] = None,
    #     ground_truth_sql: Optional[str] = None,
    #     db_schema: Optional[str] = None,
    #     timeout_seconds: Optional[int] = 300
    # ) -> UserFeedback:
    #     """
    #     Request user feedback and temporarily store
    #     Does not immediately trigger evolution, waits for session completion
        
    #     Args:
    #         feedback_type: Type of feedback ("sql_validation", "result_quality", "clarification", "preference")
    #         user_query: Original user query
    #         generated_sql: Generated SQL
    #         execution_result: Execution result (optional)
    #         options: Choice options for multi-choice feedback (optional)
    #         ground_truth_sql: Ground truth SQL for LLM mode (optional)
    #         db_schema: Database schema info for LLM mode (optional)
    #         timeout_seconds: Feedback timeout (default: 300)
        
    #     Returns:
    #         UserFeedback: Feedback response
    #     """
    #     if self._current_session_id is None:
    #         raise CAFSystemError("SESSION_NOT_STARTED", "Please start session first")
        
    #     try:
    #         # Create FeedbackContext object internally - users don't need to know about it
    #         feedback_type_enum = FeedbackType(feedback_type)
    #         context = FeedbackContext(
    #             feedback_type=feedback_type_enum,
    #             user_query=user_query,
    #             generated_sql=generated_sql,
    #             execution_result=execution_result,
    #             options=options,
    #             ground_truth_sql=ground_truth_sql,
    #             db_schema=db_schema,
    #             timeout_seconds=timeout_seconds
    #         )
            
    #         # Collect user feedback
    #         feedback = self._feedback_manager.collect_feedback(context)
            
    #         # Temporarily store feedback, associated with current session
    #         feedback_entry = {
    #             'feedback': feedback.dict(),
    #             'context': context.dict(),  # Use dict() to ensure proper serialization
    #             'timestamp': datetime.utcnow().isoformat(),
    #             'session_id': self._current_session_id,
    #             'database_id': self._current_database_id
    #         }
            
    #         # Store in both global temporary storage and session context
    #         self._temp_feedback_storage.append(feedback_entry)
    #         self._session_context[self._current_session_id]['temp_feedbacks'].append(feedback_entry)
            
    #         logger.info(f"Feedback collected and temporarily stored for session: {self._current_session_id}")
    #         return feedback
            
    #     except ValueError as e:
    #         # Handle invalid feedback_type
    #         valid_types = [t.value for t in FeedbackType]
    #         raise CAFSystemError("INVALID_FEEDBACK_TYPE", f"Invalid feedback type '{feedback_type}'. Valid types: {valid_types}")
    #     except Exception as e:
    #         logger.error(f"Feedback request failed: {e}")
    #         raise CAFSystemError("FEEDBACK_REQUEST_FAILED", f"Feedback request failed: {e}")
    
    def store_round(self,
                   user_query: str,
                   generated_sql: Optional[str] = None,
                   execution_result: Optional[Dict[str, Any]] = None,
                   execution_success: Optional[bool] = None,
                   user_feedback: Optional[Dict] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   round_id: Optional[int] = None) -> None:
        """
        Store a single round of interaction to episodic memory
        
        This method should be called after each SQL generation and user interaction.
        Each round is stored as a separate EpisodicRecord in the flattened structure.
        
        Args:
            user_query: Original user query for this round
            generated_sql: Generated SQL for this round
            execution_result: SQL execution result (optional)
            execution_success: Whether SQL execution was successful (optional)
            user_feedback: User feedback data (optional)
            metadata: Additional metadata (optional)
            round_id: Round ID (auto-incremented if not provided)
        """
        if self._current_session_id is None:
            raise CAFSystemError("SESSION_NOT_STARTED", "Please start session first before storing rounds")
        
        if self._current_database_id is None:
            raise CAFSystemError("DATABASE_NOT_BOUND", "Please bind database first before storing rounds")
        
        try:
            # Get or increment round_id
            if round_id is None:
                # Get current round count from session context
                if self._current_session_id in self._session_context:
                    existing_rounds = self._session_context[self._current_session_id].get('round_count', 0)
                    round_id = existing_rounds + 1
                    self._session_context[self._current_session_id]['round_count'] = round_id
                else:
                    round_id = 1
                    if self._current_session_id not in self._session_context:
                        self._session_context[self._current_session_id] = {}
                    self._session_context[self._current_session_id]['round_count'] = round_id
            
            # Handle execution_result conversion
            if execution_success is not None and execution_result is None:
                execution_result = {"execution_success": execution_success}
            elif execution_success is not None and isinstance(execution_result, dict):
                execution_result["execution_success"] = execution_success
            
            # Convert execution_result dict to SQLExecutionResult if needed
            sql_execution_result = None
            if execution_result:
                if isinstance(execution_result, dict):
                    sql_execution_result = SQLExecutionResult(**execution_result)
                elif isinstance(execution_result, SQLExecutionResult):
                    sql_execution_result = execution_result
            
            # Extract label and error_types from user_feedback
            label = None
            error_types = None
            feedback_text = None
            if user_feedback:
                if isinstance(user_feedback, dict):
                    label = user_feedback.get('label')
                    error_types = user_feedback.get('error_types')
                    feedback_text = user_feedback.get('feedback_text')
                elif hasattr(user_feedback, 'label'):
                    label = user_feedback.label
                    error_types = user_feedback.error_types
                    feedback_text = user_feedback.feedback_text
            
            # Create EpisodicRecord (flattened structure)
            episodic_record = EpisodicRecord(
                session_id=self._current_session_id,
                database_id=self._current_database_id,
                user_query=user_query,
                context=None,  # Can be set from metadata if needed
                round_id=round_id,
                generated_sql=generated_sql,
                execution_result=sql_execution_result,
                label=label,
                error_types=error_types,
                feedback_text=feedback_text,
                source_model=metadata.get('source_model') if metadata else None,
                timestamp=datetime.utcnow().isoformat(),
                metadata=metadata
            )
            
            # Store to episodic memory
            episodic_store = self._memory_base.get_memory_store(MemoryType.EPISODIC)
            if episodic_store:
                episodic_store.store_record(episodic_record)
                logger.info(f"Round {round_id} stored for session: {self._current_session_id}")
            else:
                raise CAFSystemError("EPISODIC_STORE_NOT_AVAILABLE", "Episodic memory store not available")
            
        except Exception as e:
            logger.error(f"Failed to store round for session {self._current_session_id}: {e}")
            raise CAFSystemError("ROUND_STORAGE_FAILED", f"Round storage failed: {e}")
    
    def finalize_session(self) -> None:
        """
        Complete current session and cleanup resources
        
        Note: This method does NOT store episodic memory. Use store_round() 
        to store each round of interaction immediately after it occurs.
        
        This method:
        - Cleans up temporary storage and session context
        """
        if self._current_session_id is None:
            raise CAFSystemError("SESSION_NOT_STARTED", "No active session to end")
        
        try:
            session_id = self._current_session_id
            
            # Cleanup temporary storage and session context
            self._temp_feedback_storage.clear()
            if session_id in self._session_context:
                del self._session_context[session_id]
            self._current_session_id = None
            # Note: _current_database_id is NOT cleared here - database remains bound
            
            logger.info(f"Session ended: {session_id}")
            
        except Exception as e:
            logger.error(f"Session end failed: {e}")
            raise CAFSystemError("SESSION_END_FAILED", f"Session end failed: {e}")
    
    def unbind_database(self) -> None:
        """
        Unbind current database
        This is optional - database can remain bound across multiple sessions
        
        Note: This will fail if there is an active session. End the session first.
        """
        if self._current_session_id is not None:
            raise CAFSystemError("SESSION_ACTIVE", "Cannot unbind database while session is active. Please end session first.")
        
        if self._current_database_id is None:
            logger.warning("No database bound to unbind")
            return
        
        database_id = self._current_database_id
        self._current_database_id = None
        
        # Note: MemoryBase doesn't have an unbind method, so we just clear our reference
        # The memory stores will continue to use the last bound database until a new one is bound
        
        logger.info(f"Database unbound: {database_id}")
    
    def finalize_session(self) -> None:
        """
        Deprecated: Use end_session() instead
        This method is kept for backward compatibility
        """
        logger.warning("finalize_session() is deprecated. Use end_session() instead.")
        self.end_session()
    
    # Utility methods
    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self._current_session_id
    
    def get_current_database_id(self) -> Optional[str]:
        """Get current database ID"""
        return self._current_database_id
    
    def get_session_feedbacks(self) -> List[Dict[str, Any]]:
        """Get all feedbacks for current session"""
        if self._current_session_id and self._current_session_id in self._session_context:
            return self._session_context[self._current_session_id]['temp_feedbacks']
        return []
    
    def is_session_active(self) -> bool:
        """Check if session is active"""
        return self._current_session_id is not None
    
    
    def _get_default_llm_config(self) -> Optional[LLMConfig]:
        """
        Get default LLM configuration from CAF config.
        
        Returns:
            LLMConfig object or None if not available
        """
        try:
            # Access LLM config from _raw_data
            llm_cfg = None
            if hasattr(self.config, '_raw_data') and 'llm' in self.config._raw_data:
                llm_cfg = self.config._raw_data['llm']
            else:
                # Fallback to getattr approach
                llm_cfg = getattr(self.config, 'llm', {}) or {}
            
            if llm_cfg:
                return LLMConfig(
                    provider=llm_cfg.get('provider', 'openai'),
                    model_name=llm_cfg.get('model_name', 'gpt-4o-mini'),
                    api_key=llm_cfg.get('api_key'),
                    base_url=llm_cfg.get('base_url'),
                    temperature=llm_cfg.get('temperature', 0.1),
                    max_tokens=llm_cfg.get('max_tokens', 4000),
                    timeout=llm_cfg.get('timeout', 60)
                )
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to create LLM config: {e}")
            return None
    
    def get_memory_store(self, memory_type: MemoryType):
        """
        Get a specific memory store by type.
        
        This provides a controlled interface for accessing memory stores
        without exposing internal implementation details.
        
        Args:
            memory_type: The type of memory store to retrieve
            
        Returns:
            Memory store object or None if not available
        """
        try:
            return self._memory_base.memory_stores.get(memory_type)
        except Exception as e:
            logger.error(f"Failed to get memory store {memory_type}: {e}")
            return None

    def delete_semantic_field(
        self,
        database_id: str,
        metadata_type: str,
        field_name: str,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
        term_name: Optional[str] = None,
        remove_versions: bool = False,
        save: bool = True,
    ) -> bool:
        """
        Delete a field in semantic memory (database/table/column/term).

        This is a thin wrapper around SemanticMemoryStore.delete_field to expose
        a public CAFSystem API.
        """
        semantic_store = self._memory_base.get_memory_store(MemoryType.SEMANTIC)
        if semantic_store is None:
            raise CAFSystemError("SEMANTIC_STORE_NOT_AVAILABLE", "Semantic memory store is not available")

        return semantic_store.delete_field(
            database_id=database_id,
            metadata_type=metadata_type,
            field_name=field_name,
            table_name=table_name,
            column_name=column_name,
            term_name=term_name,
            save=save,
            remove_versions=remove_versions,
        )
    
    def cleanup(self) -> None:
        """Cleanup CAF system resources"""
        try:
            # Save any incomplete sessions
            if self._temp_feedback_storage:
                self._handle_incomplete_session()
            
            # Cleanup components
            if hasattr(self._memory_base, 'cleanup'):
                self._memory_base.cleanup()
            
            # if hasattr(self._feedback_manager, 'cleanup'):
            #     self._feedback_manager.cleanup()
            
            # Clear state
            self._temp_feedback_storage.clear()
            self._session_context.clear()
            self._current_session_id = None
            self._current_database_id = None
            
            logger.info("CAFSystem cleanup completed")
            
        except Exception as e:
            logger.error(f"CAFSystem cleanup failed: {e}")
    
    def _handle_incomplete_session(self) -> None:
        """Handle incomplete session during cleanup"""
        logger.warning("Incomplete session detected during cleanup")
        # Future: Could save incomplete session data for recovery
        # For now, just clear the temporary storage
        self._temp_feedback_storage.clear()
    
    def _generate_initial_sql(self, 
                             question: str,
                             context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate initial SQL query from natural language question
        
        This method extracts schema information from semantic memory and uses
        LLM to generate an initial SQL query, helping to understand the user's intent.
        
        Args:
            question: Natural language question
            context: Additional context (optional)
            
        Returns:
            Generated SQL query string
            
        Raises:
            CAFSystemError: If schema extraction or SQL generation fails
        """
        try:
            # Step 1: Extract schema from semantic memory
            schema = self._extract_schema_from_semantic_memory()
            if not schema or not schema.table_names:
                raise CAFSystemError("SCHEMA_NOT_AVAILABLE", 
                                   "Cannot generate SQL: No schema information available in semantic memory. "
                                   "Please ensure semantic memory is populated for this database.")
            
            # Step 2: Get LLM configuration
            llm_config = self._get_default_llm_config()
            if not llm_config:
                raise CAFSystemError("LLM_NOT_CONFIGURED", 
                                   "Cannot generate SQL: LLM configuration not available. "
                                   "Please configure LLM in CAF config.")
            
            # Step 3: Initialize SQL generator
            from .llm.sql_generator import SQLGenerator, DatabaseSchema
            sql_generator = SQLGenerator(llm_config=llm_config)
            
            # Step 4: Generate SQL
            evidence = context.get("evidence") if context else None
            generated_sql = sql_generator.generate_sql(
                question=question,
                schema=schema,
                evidence=evidence,
                context=context
            )
            
            return generated_sql
            
        except CAFSystemError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate initial SQL: {e}")
            raise CAFSystemError("SQL_GENERATION_FAILED", f"Failed to generate initial SQL: {e}")
    
    def _extract_schema_from_semantic_memory(self):
        """
        Extract database schema from semantic memory store
        
        Returns:
            DatabaseSchema object with table and column information
        """
        from .llm.sql_generator import DatabaseSchema
        
        try:
            semantic_store = self._memory_base.memory_stores.get(MemoryType.SEMANTIC)
            if not semantic_store:
                raise CAFSystemError("SEMANTIC_STORE_NOT_AVAILABLE", 
                                   "Semantic memory store not available")
            
            # Ensure dataframes are initialized
            if not hasattr(semantic_store, 'dataframes') or not semantic_store.dataframes:
                raise CAFSystemError("DATAFRAMES_NOT_INITIALIZED", 
                                   "Semantic memory dataframes not initialized. "
                                   "Please ensure database is properly bound.")
            
            # Get table and column metadata from semantic store
            if 'table' not in semantic_store.dataframes:
                raise CAFSystemError("NO_TABLE_METADATA", 
                                   "No table metadata found in semantic memory")
            
            table_df = semantic_store.dataframes['table']
            if table_df.empty:
                raise CAFSystemError("NO_TABLE_METADATA", 
                                   "No table metadata found in semantic memory")
            
            if 'column' not in semantic_store.dataframes:
                raise CAFSystemError("NO_COLUMN_METADATA", 
                                   "No column metadata found in semantic memory")
            
            column_df = semantic_store.dataframes['column']
            if column_df.empty:
                raise CAFSystemError("NO_COLUMN_METADATA", 
                                   "No column metadata found in semantic memory")
            
            # Extract table names
            table_names = table_df['table_name'].unique().tolist()
            
            # Extract columns for each table
            columns = {}
            table_descriptions = {}
            
            for table_name in table_names:
                table_columns = column_df[column_df['table_name'] == table_name]
                
                column_list = []
                for _, row in table_columns.iterrows():
                    col_info = {
                        'column_name': row.get('column_name', ''),
                        'data_type': row.get('data_type', 'TEXT'),
                        'is_primary_key': row.get('is_primary_key', False)
                    }
                    column_list.append(col_info)
                
                columns[table_name] = column_list
                
                # Extract column descriptions if available
                if 'description' in table_columns.columns:
                    desc_dict = {}
                    for _, row in table_columns.iterrows():
                        col_name = row.get('column_name', '')
                        desc = row.get('description', '')
                        if desc:
                            desc_dict[col_name] = desc
                    if desc_dict:
                        table_descriptions[table_name] = desc_dict
            
            # Create DatabaseSchema object
            schema = DatabaseSchema(
                database_id=self._current_database_id,
                table_names=table_names,
                columns=columns,
                table_descriptions=table_descriptions if table_descriptions else None
            )
            
            logger.debug(f"Extracted schema: {len(table_names)} tables, {sum(len(cols) for cols in columns.values())} columns")
            return schema
            
        except CAFSystemError:
            raise
        except Exception as e:
            logger.error(f"Failed to extract schema from semantic memory: {e}")
            raise CAFSystemError("SCHEMA_EXTRACTION_FAILED", f"Failed to extract schema: {e}")
    
    # ==================== Metadata Import & Generation (Offline) ====================
    
    def import_bird_metadata(
        self,
        database_path: str,
        bird_data_dir: str,
        force_regenerate: bool = False,
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Import metadata from BIRD dataset documentation (offline operation).
        
        This method imports existing metadata from BIRD dataset's database_description
        CSV files and dev.json evidence fields. It does NOT require bind_database()
        to be called first.
        
        BIRD dataset provides high-quality human-annotated metadata including:
        - Column descriptions from database_description/*.csv files
        - Value descriptions (encoded values, business rules)
        - Term definitions from dev.json evidence fields
        
        Args:
            database_path: Path to database file (required)
            bird_data_dir: Path to BIRD dataset root directory (required)
                          e.g., "/path/to/bird" which contains "dev/dev_databases/"
            force_regenerate: Force reimport even if metadata exists (default: False)
            continue_on_error: Continue importing other tables if one fails (default: True)
            
        Returns:
            Dict with import summary:
                - database_id: str
                - duration_seconds: float
                - tables_processed: int
                - columns_imported: int
                - terms_imported: int
                - errors: List[str]
                - warnings: List[str]
                
        Example:
            >>> caf_system = CAFSystem(config)
            >>> result = caf_system.import_bird_metadata(
            ...     database_path="/path/to/california_schools.sqlite",
            ...     bird_data_dir="/path/to/bird",
            ...     force_regenerate=False
            ... )
            >>> print(f"Imported {result['columns_imported']} columns")
            
        Note:
            - This is an OFFLINE operation (no need to bind_database first)
            - Schema matching is performed automatically (BIRD table/column names -> actual database)
            - LLM is used to parse value_description fields
            - Errors are collected and reported in the result
        """
        
        # Validate paths
        db_path = Path(database_path)
        if not db_path.exists():
            raise CAFSystemError("DATABASE_NOT_FOUND", f"Database file not found: {database_path}")
        
        bird_dir = Path(bird_data_dir)
        if not bird_dir.exists():
            raise CAFSystemError("BIRD_DATA_DIR_NOT_FOUND", f"BIRD data directory not found: {bird_data_dir}")
        
        logger.info(f"Starting BIRD metadata import for: {database_path}")
        logger.info(f"BIRD data directory: {bird_data_dir}")
        
        start_time = datetime.utcnow()
        errors = []
        warnings = []
        
        try:
            # Get semantic store
            semantic_store = self._memory_base.get_memory_store(MemoryType.SEMANTIC)
            
            # Extract LLM configuration
            llm_cfg = getattr(self.config, 'llm', {}) or {}
            llm_config = {
                'provider': llm_cfg.get('provider', 'openai'),
                'model_name': llm_cfg.get('model_name', 'gpt-4o-mini'),
                'api_key': llm_cfg.get('api_key'),
                'base_url': llm_cfg.get('base_url'),
                'temperature': llm_cfg.get('temperature', 0.1),
                'max_tokens': llm_cfg.get('max_tokens', 800),
                'timeout': llm_cfg.get('timeout', 60)
            }
            
            # Initialize BirdMetadataExtractor
            extractor = BirdMetadataExtractor(
                bird_data_dir=bird_dir,
                semantic_store=semantic_store,
                llm_config=llm_config,
                only_value_desc=None  # Import all value descriptions
            )
            
            # Extract database_id before import
            database_id = None
            try:
                database_id = extractor.schema_extractor.extract_database_id(database_path)
            except Exception as e:
                logger.warning(f"Failed to extract database_id: {e}")
            
            # Extract and import metadata from database path
            success = extractor.extract_and_import(str(database_path))
            
            if not success:
                errors.append(f"Failed to import metadata for {database_path}")
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Build result summary
            result = {
                'database_id': database_id,
                'database_path': database_path,
                'bird_data_dir': bird_data_dir,
                'duration_seconds': duration,
                'success': success,
                'errors': errors,
                'warnings': warnings,
            }
            
            # Try to get more detailed statistics if available
            if success and semantic_store:
                try:
                    # Count imported items from semantic store
                    # Note: This is approximate as we don't track exact counts during import
                    result['import_summary'] = 'Metadata imported successfully'
                except Exception as e:
                    logger.debug(f"Failed to get detailed statistics: {e}")
            
            # Display summary
            self._display_bird_import_summary(result)
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"BIRD metadata import failed: {e}")
            
            if continue_on_error:
                errors.append(str(e))
                return {
                    'database_id': None,
                    'database_path': database_path,
                    'bird_data_dir': bird_data_dir,
                    'duration_seconds': duration,
                    'success': False,
                    'errors': errors,
                    'warnings': warnings,
                }
            else:
                raise CAFSystemError("BIRD_IMPORT_FAILED", f"BIRD metadata import failed: {e}")
    
    def import_and_generate_metadata(
        self,
        database_path: str,
        bird_data_dir: str,
        *,
        force_regenerate: bool = False,
        enable_ddl_analysis: Optional[bool] = None,
        enable_profiling: Optional[bool] = None,
        enable_llm_analysis: Optional[bool] = None,
        enable_join_path_discovery: Optional[bool] = None,
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Import BIRD metadata first, then generate additional metadata (offline operation).
        
        This is a convenient combination method that:
        1. Imports existing metadata from BIRD dataset (high-quality human annotations)
        2. Generates additional metadata to fill gaps (DDL analysis, profiling, LLM analysis)
        
        The two-step process ensures:
        - BIRD metadata (human-annotated) has highest priority
        - Auto-generated metadata fills in missing information
        - No conflicts due to source priority system
        
        Args:
            database_path: Path to database file (required)
            bird_data_dir: Path to BIRD dataset root directory (required)
            force_regenerate: Force regenerate all metadata (default: False)
            enable_ddl_analysis: Enable DDL analysis (None = use config, default: True)
            enable_profiling: Enable data profiling (None = use config, default: True)
            enable_llm_analysis: Enable LLM analysis (None = use config, default: True)
            enable_join_path_discovery: Enable join path discovery (None = use config, default: True)
            continue_on_error: Continue if one step fails (default: True)
            
        Returns:
            Dict with combined results:
                - bird_import_result: Dict from import_bird_metadata()
                - metadata_generation_result: Result from generate_metadata()
                - overall_success: bool
                - total_duration_seconds: float
                
        Example:
            >>> caf_system = CAFSystem(config)
            >>> result = caf_system.import_and_generate_metadata(
            ...     database_path="/path/to/california_schools.sqlite",
            ...     bird_data_dir="/path/to/bird",
            ...     enable_llm_analysis=True
            ... )
            
        Note:
            - This is an OFFLINE operation (no need to bind_database first)
            - BIRD metadata is imported first (Step 1)
            - Additional metadata is generated second (Step 2)
            - If continue_on_error=True, Step 2 runs even if Step 1 fails
        """
        from datetime import datetime
        
        logger.info("="*80)
        logger.info("=== Combined Import & Generation ===")
        logger.info("="*80)
        
        start_time = datetime.utcnow()
        bird_result = None
        gen_result = None
        overall_success = True
        
        # Step 1: Import BIRD metadata
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Importing BIRD metadata")
        logger.info("="*80)
        
        try:
            bird_result = self.import_bird_metadata(
                database_path=database_path,
                bird_data_dir=bird_data_dir,
                force_regenerate=force_regenerate,
                continue_on_error=continue_on_error
            )
            
            if not bird_result.get('success', False):
                logger.warning("⚠️ BIRD metadata import failed or incomplete")
                overall_success = False
                
                if not continue_on_error:
                    logger.error("Stopping due to BIRD import failure (continue_on_error=False)")
                    total_duration = (datetime.utcnow() - start_time).total_seconds()
                    return {
                        'bird_import_result': bird_result,
                        'metadata_generation_result': None,
                        'overall_success': False,
                        'total_duration_seconds': total_duration,
                    }
            else:
                logger.info("✅ BIRD metadata import completed successfully")
                
        except Exception as e:
            logger.error(f"❌ BIRD metadata import failed: {e}")
            overall_success = False
            bird_result = {
                'success': False,
                'errors': [str(e)]
            }
            
            if not continue_on_error:
                logger.error("Stopping due to error (continue_on_error=False)")
                total_duration = (datetime.utcnow() - start_time).total_seconds()
                return {
                    'bird_import_result': bird_result,
                    'metadata_generation_result': None,
                    'overall_success': False,
                    'total_duration_seconds': total_duration,
                }
        
        # Step 2: Generate additional metadata
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Generating additional metadata")
        logger.info("="*80)
        
        try:
            gen_result = self.generate_metadata(
                database_path=database_path,
                enable_ddl_analysis=enable_ddl_analysis,
                enable_profiling=enable_profiling,
                enable_llm_analysis=enable_llm_analysis,
                enable_join_path_discovery=enable_join_path_discovery,
                force_regenerate=False,  # Don't force regenerate in Step 2, respect BIRD data
            )
            
            logger.info("✅ Metadata generation completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Metadata generation failed: {e}")
            overall_success = False
            gen_result = {
                'success': False,
                'errors': [str(e)]
            }
        
        # Calculate total duration
        total_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Display combined summary
        logger.info("\n" + "="*80)
        logger.info("=== Combined Import & Generation Summary ===")
        logger.info("="*80)
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"Overall Status: {'✅ Success' if overall_success else '⚠️ Completed with errors'}")
        logger.info(f"  - BIRD Import: {'✅' if bird_result and bird_result.get('success') else '❌'}")
        logger.info(f"  - Metadata Generation: {'✅' if gen_result and not isinstance(gen_result, dict) else '❌'}")
        logger.info("="*80 + "\n")
        
        return {
            'bird_import_result': bird_result,
            'metadata_generation_result': gen_result,
            'overall_success': overall_success,
            'total_duration_seconds': total_duration,
        }

    def generate_metadata(
        self,
        database_path: str,
        enable_ddl_analysis: Optional[bool],
        enable_profiling: Optional[bool],
        enable_llm_analysis: Optional[bool],
        enable_join_path_discovery: Optional[bool],
        force_regenerate: bool,
    ) -> None:
        """
        Generate metadata for a single database (offline operation).
        
        This method is designed for offline metadata generation and does NOT require
        bind_database() to be called first. It independently generates comprehensive
        database metadata including DDL analysis, data profiling, and LLM-based
        semantic descriptions.
        
        Args:
            database_path: Path to database file (required)
            enable_ddl_analysis: Enable DDL analysis (None = use config, default: True)
            enable_profiling: Enable data profiling (None = use config, default: True)
            enable_llm_analysis: Enable LLM analysis (None = use config, default: True)
            enable_join_path_discovery: Enable join path discovery (None = use config, default: True)
            force_regenerate: Force regenerate all metadata, ignoring existing data
            
        Example:
            >>> caf_system = CAFSystem(config)
            >>> caf_system.generate_metadata(
            ...     database_path="/path/to/database.sqlite",
            ...     enable_llm_analysis=True
            ... )
            
        Note:
            - This is an OFFLINE operation (no need to bind_database first)
            - Progress is logged during generation
            - Summary is displayed at the end
            - All errors are caught and logged internally
        """
        from pathlib import Path
        from .memory.generators.metadata_generator import MetadataGenerator
        
        # Validate database path
        db_path = Path(database_path)
        if not db_path.exists():
            raise CAFSystemError("DATABASE_NOT_FOUND", f"Database file not found: {database_path}")
        
        logger.info(f"Starting metadata generation for: {database_path}")
        
        # Step 1: Get SemanticMemoryStore from MemoryBase
        semantic_store = self._memory_base.get_memory_store(MemoryType.SEMANTIC)
        if not semantic_store:
            raise CAFSystemError("SEMANTIC_STORE_NOT_AVAILABLE", 
                               "Semantic memory store not available")
        
        # Step 2: Resolve configuration priority (parameters > config defaults)
        # Get generators config from memory section with defaults all True
        memory_config = self.config.memory if hasattr(self.config, 'memory') else {}
        generators_config = memory_config.get('generators', {}) if isinstance(memory_config, dict) else {}
        
        # Parameter overrides config, config defaults to True
        final_enable_ddl = (
            enable_ddl_analysis 
            if enable_ddl_analysis is not None 
            else generators_config.get('enable_ddl_analysis', True)
        )
        final_enable_profiling = (
            enable_profiling 
            if enable_profiling is not None 
            else generators_config.get('enable_profiling', True)
        )
        final_enable_llm = (
            enable_llm_analysis 
            if enable_llm_analysis is not None 
            else generators_config.get('enable_llm_analysis', True)
        )
        final_enable_join = (
            enable_join_path_discovery 
            if enable_join_path_discovery is not None 
            else generators_config.get('enable_join_path_discovery', True)
        )
        
        logger.info(f"Generation configuration: DDL={final_enable_ddl}, "
                   f"Profiling={final_enable_profiling}, LLM={final_enable_llm}, "
                   f"JoinDiscovery={final_enable_join}, Force={force_regenerate}")
        
        # Step 3: Prepare generator configuration
        generator_config = {
            'enable_ddl_analysis': final_enable_ddl,
            'enable_profiling': final_enable_profiling,
            'enable_llm_analysis': final_enable_llm,
            'enable_join_path_discovery': final_enable_join,
            'force_regenerate': force_regenerate,
        }
        
        # Add LLM configuration if LLM analysis is enabled
        if final_enable_llm:
            llm_cfg = getattr(self.config, 'llm', {}) or {}
            generator_config['llm'] = {
                'provider': llm_cfg.get('provider', 'openai'),
                'model_name': llm_cfg.get('model_name', 'gpt-4o-mini'),
                'api_key': llm_cfg.get('api_key'),
                'base_url': llm_cfg.get('base_url'),
                'temperature': llm_cfg.get('temperature', 0.1),
                'max_tokens': llm_cfg.get('max_tokens', 4000),
                'timeout': llm_cfg.get('timeout', 60),
            }

        # Step 4: Initialize MetadataGenerator
        logger.info("Initializing metadata generator...")
        generator = MetadataGenerator(
            semantic_store=semantic_store,
            config=generator_config,
            memory_config=memory_config
        )
        
        
        logger.info("=== Starting Metadata Generation ===")
        result = generator.generate_database_metadata(database_path)
        
        # Step 6: Display summary
        self._display_generation_summary(result)
    
    def _display_bird_import_summary(self, result: Dict[str, Any]) -> None:
        """Display BIRD import summary in a formatted way."""
        try:
            logger.info("\n" + "="*80)
            logger.info("=== BIRD Metadata Import Summary ===")
            logger.info("="*80)
            logger.info(f"Database ID: {result.get('database_id', 'Unknown')}")
            logger.info(f"Database Path: {result.get('database_path', 'Unknown')}")
            logger.info(f"BIRD Data Dir: {result.get('bird_data_dir', 'Unknown')}")
            logger.info(f"Duration: {result.get('duration_seconds', 0):.2f} seconds")
            logger.info(f"Status: {'✅ Success' if result.get('success', False) else '❌ Failed'}")
            
            # Show errors
            errors = result.get('errors', [])
            if errors:
                logger.error(f"\n❌ Errors encountered: {len(errors)}")
                for error in errors[:5]:  # Show first 5 errors
                    logger.error(f"   - {error}")
                if len(errors) > 5:
                    logger.error(f"   ... and {len(errors) - 5} more errors")
            
            # Show warnings
            warnings = result.get('warnings', [])
            if warnings:
                logger.warning(f"\n⚠️ Warnings: {len(warnings)}")
                for warning in warnings[:5]:  # Show first 5 warnings
                    logger.warning(f"   - {warning}")
                if len(warnings) > 5:
                    logger.warning(f"   ... and {len(warnings) - 5} more warnings")
            
            logger.info("="*80 + "\n")
            
            if result.get('success', False):
                logger.info("✅ BIRD metadata import completed successfully!")
            else:
                logger.error("❌ BIRD metadata import failed")
                
        except Exception as e:
            logger.error(f"Failed to display BIRD import summary: {e}")
            logger.debug("Raw result object: %s", result, exc_info=True)
    
    def _display_generation_summary(self, result: Any) -> None:
        """Display generation summary in a formatted way."""
        try:
            summary = result.get_summary()
            
            logger.info("\n" + "="*80)
            logger.info("=== Metadata Generation Summary ===")
            logger.info("="*80)
            logger.info(f"Database ID: {summary['database_id']}")
            logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
            logger.info(f"Status: {'✅ Success' if summary['errors'] == 0 else '⚠️ Completed with errors'}")
            
            # Show generation counts
            ddl_counts = summary['generation_counts']['ddl_analysis']
            profiling_counts = summary['generation_counts']['data_profiling']
            llm_counts = summary['generation_counts']['llm_analysis']
            
            logger.info("\nGenerated Metadata:")
            logger.info(f"  📊 DDL Analysis:")
            logger.info(f"     - {ddl_counts['database']} database info")
            logger.info(f"     - {ddl_counts['table']} tables")
            logger.info(f"     - {ddl_counts['column']} columns")
            logger.info(f"     - {ddl_counts['relationship']} relationships")
            
            if profiling_counts['table'] > 0 or profiling_counts['column'] > 0:
                logger.info(f"  📈 Data Profiling:")
                logger.info(f"     - {profiling_counts['table']} tables profiled")
                logger.info(f"     - {profiling_counts['column']} columns profiled")
            
            if llm_counts['database'] > 0 or llm_counts['table'] > 0 or llm_counts['column'] > 0:
                logger.info(f"  🤖 LLM Analysis:")
                logger.info(f"     - {llm_counts['database']} database descriptions")
                logger.info(f"     - {llm_counts['table']} table descriptions")
                logger.info(f"     - {llm_counts['column']} column descriptions")
            
            logger.info(f"\n  🔄 Conflicts resolved: {summary['conflicts_resolved']}")
            
            # Show errors and warnings
            if summary['errors'] > 0:
                logger.error(f"\n❌ Errors encountered: {summary['errors']}")
                for error in summary['error_messages'][:5]:  # Show first 5 errors
                    logger.error(f"   - {error}")
                if len(summary['error_messages']) > 5:
                    logger.error(f"   ... and {len(summary['error_messages']) - 5} more errors")
            
            if summary['warnings'] > 0:
                logger.warning(f"\n⚠️ Warnings: {summary['warnings']}")
                for warning in summary['warning_messages'][:5]:  # Show first 5 warnings
                    logger.warning(f"   - {warning}")
                if len(summary['warning_messages']) > 5:
                    logger.warning(f"   ... and {len(summary['warning_messages']) - 5} more warnings")
            
            logger.info("="*80 + "\n")
            
            if summary['errors'] == 0:
                logger.info("✅ Metadata generation completed successfully!")
            else:
                logger.warning("⚠️ Metadata generation completed with errors")
                
        except Exception as e:
            logger.error(f"Failed to display generation summary: {e}")
            logger.debug("Raw result object: %s", result, exc_info=True)
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
