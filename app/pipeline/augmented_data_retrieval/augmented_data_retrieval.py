from app.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from app.config import config
from app.logger import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import pprint

# Import CAF system
try:
    import caf
    CAF_AVAILABLE = True
except ImportError:
    logger.warning("CAF system not available. Augmented data retrieval will be skipped.")
    CAF_AVAILABLE = False


class AugmentedDataRetrievalRunner:
    """
    Augmented Data Retrieval Runner
    
    This runner performs three main tasks:
    1. Semantic Memory Retrieval: Extract encoding_mappings from retrieved schema
    2. Schema Metadata Retrieval: Get long_description and join relationships for linked schemas
    3. SQL Guidance Retrieval: Retrieve guidance from historical insights based on SQL candidates
    """
    
    _dataset: BaseDataset = None
    _caf_system = None
    _thread_pool_executor: ThreadPoolExecutor = None
    
    def __init__(self):
        # Load from schema_linking output (augmented data retrieval happens after schema linking)
        self._dataset = load_dataset(config.schema_linking_config.save_path)
        
        if CAF_AVAILABLE:
            # Initialize CAF system
            try:
                caf_config_path = Path("config/caf_config.yaml")
                if not caf_config_path.exists():
                    logger.warning(f"CAF config not found at {caf_config_path}. Augmented data retrieval will be skipped.")
                    self._caf_system = None
                else:
                    self._caf_system = caf.initialize(config_path=str(caf_config_path))
                    logger.info("CAF system initialized successfully for augmented data retrieval")
            except Exception as e:
                logger.error(f"Failed to initialize CAF system: {e}")
                self._caf_system = None
        else:
            self._caf_system = None
        
        # Use single thread for CAF operations to avoid thread safety issues
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=1)
    
    def _extract_encoding_mappings_from_semantic_search(self, semantic_response: Any) -> Dict[str, Dict[str, Any]]:
        """
        Extract encoding_mappings from semantic search response.
        
        Args:
            semantic_response: The response from caf_system.retrieve_memory(memory_type="semantic")
        
        Returns:
            Dict mapping "table.column" to encoding_mapping dict
        """
        encoding_mappings = {}
        
        if not hasattr(semantic_response, 'column_items') or not semantic_response.column_items:
            return encoding_mappings
        
        for col_item in semantic_response.column_items:
            # Check if column_metadata exists and has encoding_mapping
            if col_item.column_metadata and 'encoding_mapping' in col_item.column_metadata:
                encoding_map = col_item.column_metadata['encoding_mapping']
                
                # Only add if encoding_mapping is not empty/null
                if encoding_map and encoding_map is not None:
                    # Handle pandas nan
                    import math
                    if isinstance(encoding_map, float) and math.isnan(encoding_map):
                        continue
                    
                    schema_key = f"{col_item.table_name}.{col_item.column_name}"
                    encoding_mappings[schema_key] = encoding_map
        
        logger.debug(f"Extracted {len(encoding_mappings)} encoding mappings from semantic search")
        return encoding_mappings
    
    def _integrate_encoding_mappings_to_linked_schema(self, data_item: DataItem) -> None:
        """
        Integrate columns from encoding_mappings into database_schema_after_schema_linking.
        
        This method adds columns that have encoding_mappings to the linked schema so they can be used
        in downstream processing (SQL generation, etc.).
        
        Args:
            data_item: The data item containing encoding_mappings and database_schema_after_schema_linking
        """
        if not data_item.encoding_mappings:
            return
        
        # Ensure database_schema_after_schema_linking exists
        if not data_item.database_schema_after_schema_linking or 'tables' not in data_item.database_schema_after_schema_linking:
            logger.warning(f"[{data_item.question_id}] No database_schema_after_schema_linking found, cannot integrate encoding mappings")
            return
        
        tables = data_item.database_schema_after_schema_linking['tables']
        added_columns = []
        
        # Extract table and column names from encoding_mappings keys (format: "table.column")
        for schema_key in data_item.encoding_mappings.keys():
            if '.' not in schema_key:
                logger.warning(f"[{data_item.question_id}] Invalid schema key format in encoding_mappings: {schema_key}")
                continue
            
            table_name, column_name = schema_key.split('.', 1)
            
            # Check if table exists in schema
            if table_name not in tables:
                logger.warning(f"[{data_item.question_id}] Table '{table_name}' from encoding_mappings not found in linked schema")
                continue
            
            # Check if column already exists
            if 'columns' not in tables[table_name]:
                logger.warning(f"[{data_item.question_id}] No 'columns' key in table '{table_name}'")
                continue
            
            if column_name in tables[table_name]['columns']:
                logger.debug(f"[{data_item.question_id}] Column '{table_name}.{column_name}' already exists in linked schema")
                continue
            
            # Get column information from full database schema
            if data_item.database_schema and 'tables' in data_item.database_schema:
                db_table = data_item.database_schema['tables'].get(table_name)
                if db_table and 'columns' in db_table:
                    column_info = db_table['columns'].get(column_name)
                    if column_info:
                        # Add column to linked schema
                        tables[table_name]['columns'][column_name] = column_info
                        added_columns.append(f"{table_name}.{column_name}")
                        logger.debug(f"[{data_item.question_id}] Added column '{table_name}.{column_name}' to database_schema_after_schema_linking")
                    else:
                        logger.warning(f"[{data_item.question_id}] Column '{column_name}' not found in database_schema table '{table_name}'")
                else:
                    logger.warning(f"[{data_item.question_id}] Table '{table_name}' not found in database_schema")
            else:
                logger.warning(f"[{data_item.question_id}] No database_schema available to get column info")
        
        if added_columns:
            logger.info(f"[{data_item.question_id}] Added {len(added_columns)} columns to database_schema_after_schema_linking: {added_columns}")
        
        # Also update final_linked_tables_and_columns for consistency
        if data_item.final_linked_tables_and_columns is None:
            data_item.final_linked_tables_and_columns = {}
        
        for schema_key in data_item.encoding_mappings.keys():
            if '.' not in schema_key:
                continue
            
            table_name, column_name = schema_key.split('.', 1)
            
            # Add table if it doesn't exist
            if table_name not in data_item.final_linked_tables_and_columns:
                data_item.final_linked_tables_and_columns[table_name] = []
            
            # Add column if it doesn't exist (deduplication)
            if column_name not in data_item.final_linked_tables_and_columns[table_name]:
                data_item.final_linked_tables_and_columns[table_name].append(column_name)
        
        logger.debug(f"[{data_item.question_id}] Integrated {len(data_item.encoding_mappings)} encoding mapping columns into linked schema")
    
    def _format_join_relationships(self, join_relationships: List[Any]) -> List[Dict[str, Any]]:
        """
        Format join relationships into serializable format.
        
        Args:
            join_relationships: List of JoinRelationship objects from CAF
        
        Returns:
            List of dictionaries containing join relationship information
        """
        formatted_joins = []
        
        for join in join_relationships:
            join_dict = {
                "table1": join.table1,
                "column1": join.column1,
                "table2": join.table2,
                "column2": join.column2,
                "join_type": join.join_type,
                "confidence": join.confidence
            }
            formatted_joins.append(join_dict)
        
        return formatted_joins
    
    def _format_schema_metadata(self, schema_metadata_response: Any) -> Dict[str, Dict[str, Any]]:
        """
        Format schema metadata response into a dictionary.
        
        Args:
            schema_metadata_response: Response from caf_system.get_schema_metadata()
        
        Returns:
            Dict with:
            - Column metadata: "table.column" -> metadata dict
            - Table metadata: "__tables__" -> {table_name -> table_metadata dict}
        """
        metadata_dict = {}
        
        # Format column metadata
        if hasattr(schema_metadata_response, 'columns') and schema_metadata_response.columns:
            # schema_metadata_response.columns is already a dict with "table.column" keys
            for col_key, col_meta in schema_metadata_response.columns.items():
                metadata_dict[col_key] = col_meta
        
        # Format table metadata
        if hasattr(schema_metadata_response, 'tables') and schema_metadata_response.tables:
            metadata_dict['__tables__'] = schema_metadata_response.tables
        
        logger.debug(f"Formatted {len([k for k in metadata_dict.keys() if not k.startswith('__')])} column metadata entries and {len(metadata_dict.get('__tables__', {}))} table metadata entries")
        return metadata_dict
    
    def _format_guidance_items(self, guidance_response: Any) -> List[Dict[str, Any]]:
        """
        Format guidance response into list of dictionaries.
        
        Extracts structured information from guidance items:
        - insight_id: Unique identifier for the insight
        - intent: The intent of the guidance
        - strategy_incorrect: Incorrect strategy pattern and implication
        - strategy_correct: Correct strategy pattern and implication
        - actionable_advice: Actionable advice for the developer
        - qualified_incorrect_sql: Example of incorrect SQL
        - qualified_correct_sql: Example of correct SQL
        
        Args:
            guidance_response: Response from caf_system.retrieve_sql_guidance()
        
        Returns:
            List of guidance item dictionaries with structured fields
        """
        guidance_items = []
        
        if not hasattr(guidance_response, 'items') or not guidance_response.items:
            return guidance_items
        
        for item in guidance_response.items:
            # Extract structured fields from guidance dict
            guidance_dict = {
                "insight_id": item.insight_id,
            }
            
            # Extract fields from guidance dict if available
            if item.guidance:
                if "intent" in item.guidance:
                    guidance_dict["intent"] = item.guidance["intent"]
                
                if "strategy_incorrect" in item.guidance:
                    guidance_dict["strategy_incorrect"] = item.guidance["strategy_incorrect"]
                
                if "strategy_correct" in item.guidance:
                    guidance_dict["strategy_correct"] = item.guidance["strategy_correct"]
                
                if "actionable_advice" in item.guidance:
                    guidance_dict["actionable_advice"] = item.guidance["actionable_advice"]
            
            # Extract example SQLs
            if item.qualified_incorrect_sql:
                guidance_dict["qualified_incorrect_sql"] = item.qualified_incorrect_sql
            
            if item.qualified_correct_sql:
                guidance_dict["qualified_correct_sql"] = item.qualified_correct_sql
            
            guidance_items.append(guidance_dict)
        
        logger.debug(f"Formatted {len(guidance_items)} guidance items")
        return guidance_items
    
    def _retrieve_augmented_data(self, data_item: DataItem) -> None:
        """
        Retrieve augmented data for a single data item.
        
        This includes:
        1. Semantic search for encoding_mappings
        2. Schema metadata for long_description and joins
        3. SQL guidance from historical insights
        """
        start_time = time.time()
        
        # Initialize costs (CAF operations don't use LLM tokens)
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if not self._caf_system:
            # No CAF system available, skip augmented data retrieval
            data_item.encoding_mappings = {}
            data_item.schema_metadata = {}
            data_item.join_relationships = []
            data_item.sql_guidance_items = []
            data_item.augmented_data_retrieval_time = time.time() - start_time
            data_item.augmented_data_retrieval_llm_cost = total_token_usage
            return
        
        try:
            # Bind database
            self._caf_system.bind_database(data_item.database_id)
            
            ### DO NOT use caf to augment relevant schema information
            # # ======================================================================
            # # Step 1: Semantic Memory Retrieval for Encoding Mappings
            # # ======================================================================
            # logger.debug(f"[{data_item.question_id}] Step 1: Semantic memory retrieval")
            
            # # Prepare context with evidence if available
            # context = None
            # if data_item.evidence:
            #     context = {"evidence": data_item.evidence}
            
            # semantic_response = self._caf_system.retrieve_memory(
            #     memory_type="semantic",
            #     query_content=data_item.question,
            #     context=context,
            #     limit=20,  # Retrieve more items to get sufficient encoding_mappings
            #     return_per_term=False
            # )
            
            # # # Print semantic memory response
            # # logger.info(f"\n{'='*80}")
            # # logger.info(f"[{data_item.question_id}] Step 1: Semantic Memory Response")
            # # logger.info(f"{'='*80}")
            # # logger.info(f"Semantic Memory (semantic_response):")
            # # logger.info(pprint.pformat(semantic_response, width=120, depth=10))
            
            # # Extract encoding_mappings
            # data_item.encoding_mappings = self._extract_encoding_mappings_from_semantic_search(semantic_response)
            
            # # Print extracted encoding_mappings
            # logger.info(f"\n[{data_item.question_id}] Extracted encoding_mappings:")
            # logger.info(pprint.pformat(data_item.encoding_mappings, width=120, depth=10))
            
            # # Integrate encoding_mappings columns into final_linked_tables_and_columns
            # self._integrate_encoding_mappings_to_linked_schema(data_item)
            

            # ======================================================================
            # Step 2: Schema Metadata Retrieval
            # ======================================================================
            logger.debug(f"[{data_item.question_id}] Step 2: Schema metadata retrieval")
            
            # Use final_linked_tables_and_columns from schema linking
            relevant_schema = data_item.final_linked_tables_and_columns

            if relevant_schema:
                schema_metadata_response = self._caf_system.get_schema_metadata(
                    relevant_schema=relevant_schema,
                    include_joins=True
                )
                
                # # Print schema_metadata_response
                # logger.info(f"\n{'='*80}")
                # logger.info(f"[{data_item.question_id}] Step 2: Schema Metadata Response")
                # logger.info(f"{'='*80}")
                # logger.info(f"Schema Metadata Response (schema_metadata_response):")
                # logger.info(pprint.pformat(schema_metadata_response, width=120, depth=10))
                
                # Format and store schema metadata
                data_item.schema_metadata = self._format_schema_metadata(schema_metadata_response)
                
                # Print formatted schema_metadata
                logger.info(f"\n[{data_item.question_id}] Formatted schema_metadata:")
                logger.info(pprint.pformat(data_item.schema_metadata, width=120, depth=10))
                
                # Format and store join relationships
                if hasattr(schema_metadata_response, 'joins') and schema_metadata_response.joins:
                    data_item.join_relationships = schema_metadata_response.joins
                else:
                    data_item.join_relationships = []
            else:
                logger.warning(f"[{data_item.question_id}] No linked schema found, skipping schema metadata retrieval")
                data_item.schema_metadata = {}
                data_item.join_relationships = []
            
            # # Print formatted schema_metadata
            logger.info(f"Join relationships:")
            logger.info(pprint.pformat(data_item.join_relationships, width=120, depth=10))
            # ======================================================================
            # Step 3: SQL Guidance Retrieval
            # ======================================================================
            logger.debug(f"[{data_item.question_id}] Step 3: SQL guidance retrieval")
            
            # Use reversed_linking_sql_candidates for guidance retrieval
            if data_item.reversed_linking_sql_candidates and len(data_item.reversed_linking_sql_candidates) > 0:
                guidance_response = self._caf_system.retrieve_sql_guidance(
                    generated_sqls=data_item.reversed_linking_sql_candidates,
                    top_k=5
                )
                
                # Format and store guidance items
                data_item.sql_guidance_items = self._format_guidance_items(guidance_response)

                # Print guidance_response
                logger.info(f"\n{'='*80}")
                logger.info(f"[{data_item.question_id}] Step 3: Guidance Response")
                logger.info(f"{'='*80}")
                logger.info(f"Guidance Items:")
                logger.info(pprint.pformat(data_item.sql_guidance_items, width=120, depth=10))

            else:
                logger.warning(f"[{data_item.question_id}] No SQL candidates found for guidance retrieval")
                data_item.sql_guidance_items = []
            
            
            
        except Exception as e:
            logger.error(f"Error retrieving augmented data for question {data_item.question_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to empty data
            data_item.encoding_mappings = {}
            data_item.schema_metadata = {}
            data_item.join_relationships = []
            data_item.sql_guidance_items = []
        
        # Record timing and costs
        end_time = time.time()
        data_item.augmented_data_retrieval_time = end_time - start_time
        data_item.augmented_data_retrieval_llm_cost = total_token_usage
        
        # Update total time and cost
        if data_item.total_time is not None:
            data_item.total_time += data_item.augmented_data_retrieval_time
        else:
            data_item.total_time = data_item.augmented_data_retrieval_time
        
        if data_item.total_llm_cost is not None:
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.augmented_data_retrieval_llm_cost["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.augmented_data_retrieval_llm_cost["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.augmented_data_retrieval_llm_cost["total_tokens"],
            }
        else:
            data_item.total_llm_cost = data_item.augmented_data_retrieval_llm_cost
    
    def run(self):
        """Run augmented data retrieval for all data items"""
        logger.info("Starting augmented data retrieval...")
        
        if not self._caf_system:
            logger.warning("CAF system not available. Skipping augmented data retrieval.")
            # Still need to initialize empty fields for all items
            for data_item in self._dataset:
                data_item.encoding_mappings = {}
                data_item.schema_metadata = {}
                data_item.join_relationships = []
                data_item.sql_guidance_items = []
                data_item.augmented_data_retrieval_time = 0.0
                data_item.augmented_data_retrieval_llm_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self.save_result()
            return
        
        # Process all data items (using single thread for CAF thread safety)
        all_futures = []
        for data_item in self._dataset:
            future = self._thread_pool_executor.submit(self._retrieve_augmented_data, data_item)
            all_futures.append(future)
        
        for idx, future in tqdm(enumerate(as_completed(all_futures), start=1), total=len(all_futures), desc="Retrieving Augmented Data"):
            future.result()
            if idx % 20 == 0:
                logger.info(f"Augmented data retrieval {idx} / {len(all_futures)} completed")
                self.save_result()
        
        logger.info("Augmented data retrieval completed")
        self.save_result()
        self._clean_up()
    
    def save_result(self):
        """Save the dataset with augmented data"""
        save_dataset(self._dataset, config.augmented_data_retrieval_config.save_path)
    
    def _clean_up(self):
        """Clean up resources"""
        if self._thread_pool_executor is not None:
            self._thread_pool_executor.shutdown(wait=True)
            self._thread_pool_executor = None
        
        if self._caf_system is not None:
            try:
                self._caf_system.cleanup()
            except:
                pass
            self._caf_system = None
        
        self._dataset = None
