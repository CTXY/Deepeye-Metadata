from app.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from app.config import config, LLMConfig
from app.llm import LLM
from app.db_utils import filter_used_database_schema
from .linkers import DirectLinker, ReversedLinker, ValueLinker
from .utils import merge_schema_linking_results
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from app.logger import logger
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import copy

# Import CAF system
try:
    import caf
    CAF_AVAILABLE = True
except ImportError:
    logger.warning("CAF system not available. Schema linking will not use CAF metadata.")
    CAF_AVAILABLE = False


class SchemaLinkingRunner:
    
    _llm: LLM = None
    _dataset: BaseDataset = None
    _thread_pool_executor: ThreadPoolExecutor = None
    
    _direct_linker: DirectLinker = None
    _reversed_linker: ReversedLinker = None
    _value_linker: ValueLinker = None
    
    _caf_system = None
    
    def __init__(self):
        self._llm = LLM(config.schema_linking_config.llm)
        if Path(config.schema_linking_config.save_path).exists():
            logger.info(f"Resuming dataset from {config.schema_linking_config.save_path}")
            self._dataset = load_dataset(config.schema_linking_config.save_path)
        else:
            logger.info(f"Loading dataset from {config.value_retrieval_config.save_path}")
            self._dataset = load_dataset(config.value_retrieval_config.save_path)
        # self._dataset = load_dataset(config.value_retrieval_config.save_path)
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=config.schema_linking_config.n_parallel)
        self._direct_linker = DirectLinker()
        self._reversed_linker = ReversedLinker()
        self._value_linker = ValueLinker()
        
        # Initialize CAF system if configured
        if config.schema_linking_config.use_caf_metadata and CAF_AVAILABLE:
            try:
                caf_config_path = Path("config/caf_config.yaml")
                if not caf_config_path.exists():
                    logger.warning(f"CAF config not found at {caf_config_path}. Schema linking will not use CAF metadata.")
                    self._caf_system = None
                else:
                    self._caf_system = caf.initialize(config_path=str(caf_config_path))
                    logger.info("CAF system initialized successfully for schema linking")
            except Exception as e:
                logger.error(f"Failed to initialize CAF system: {e}")
                self._caf_system = None
        else:
            self._caf_system = None
    
    def _extract_relevant_schema_from_database_schema(self, database_schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract relevant schema from database_schema_after_value_retrieval.
        
        Args:
            database_schema: The database schema dictionary
            
        Returns:
            Dict mapping table_name to list of column_names, suitable for CAF get_schema_metadata
        """
        relevant_schema = {}
        if not database_schema or 'tables' not in database_schema:
            return relevant_schema
        
        for table_name, table_dict in database_schema['tables'].items():
            if 'columns' in table_dict:
                column_names = list(table_dict['columns'].keys())
                relevant_schema[table_name] = column_names
        
        return relevant_schema
    
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
        
        return metadata_dict
    
    def _format_join_relationships(self, join_relationships: List[Any]) -> List[Dict[str, Any]]:
        """
        Format join relationships into serializable format.
        
        Args:
            join_relationships: List of JoinRelationship objects or dicts from CAF
        
        Returns:
            List of dictionaries containing join relationship information
        """
        formatted_joins = []
        
        for join in join_relationships:
            # Handle both object and dict formats
            if isinstance(join, dict):
                join_dict = join.copy()
            else:
                # JoinRelationship object
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
    
    def _get_caf_metadata(self, data_item: DataItem) -> tuple[Optional[Dict[str, Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        """
        Get CAF metadata for schema linking.
        
        Args:
            data_item: The data item containing database schema
            
        Returns:
            Tuple of (schema_metadata, join_relationships) or (None, None) if unavailable
        """
        if not self._caf_system:
            return None, None
        
        try:
            # Bind database
            self._caf_system.bind_database(data_item.database_id)
            
            # Extract relevant schema from database_schema_after_value_retrieval
            relevant_schema = self._extract_relevant_schema_from_database_schema(data_item.database_schema_after_value_retrieval)
            
            if not relevant_schema:
                logger.warning(f"[{data_item.question_id}] No relevant schema found, skipping CAF metadata retrieval")
                return None, None
            
            # Get schema metadata
            schema_metadata_response = self._caf_system.get_schema_metadata(
                relevant_schema=relevant_schema,
                include_joins=True,
                include_fields=['long_description', 'short_description', 'data_type', 'semantic_type']
            )
            
            # Format schema metadata
            schema_metadata = self._format_schema_metadata(schema_metadata_response)
            
            # Format join relationships
            if hasattr(schema_metadata_response, 'joins') and schema_metadata_response.joins:
                join_relationships = self._format_join_relationships(schema_metadata_response.joins)
            else:
                join_relationships = []
            
            logger.debug(f"[{data_item.question_id}] Retrieved CAF metadata: {len([k for k in schema_metadata.keys() if not k.startswith('__')])} columns, {len(join_relationships)} joins")
            
            return schema_metadata, join_relationships
            
        except Exception as e:
            logger.error(f"Error retrieving CAF metadata for question {data_item.question_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
    
    def _link_tables_and_columns(self, data_item: DataItem) -> None:
        start_time = time.time()
        
        # Track token usage for this specific data item
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Get CAF metadata if enabled
        schema_metadata, join_relationships = self._get_caf_metadata(data_item)
        
        # Direct linking
        direct_linked_tables_and_columns, direct_tokens = self._direct_linker.link(
            data_item, 
            self._llm, 
            config.schema_linking_config.direct_linking_sampling_budget,
            schema_metadata=schema_metadata,
            join_relationships=join_relationships
        )
        total_token_usage["prompt_tokens"] += direct_tokens["prompt_tokens"]
        total_token_usage["completion_tokens"] += direct_tokens["completion_tokens"]
        total_token_usage["total_tokens"] += direct_tokens["total_tokens"]
        
        # Reversed linking
        reversed_linked_tables_and_columns, reversed_tokens, reversed_sql_candidates = self._reversed_linker.link(
            data_item, 
            self._llm, 
            config.schema_linking_config.reversed_linking_sampling_budget,
            schema_metadata=schema_metadata,
            join_relationships=join_relationships
        )
        total_token_usage["prompt_tokens"] += reversed_tokens["prompt_tokens"]
        total_token_usage["completion_tokens"] += reversed_tokens["completion_tokens"]
        total_token_usage["total_tokens"] += reversed_tokens["total_tokens"]
        data_item.reversed_linking_sql_candidates = reversed_sql_candidates
        
        # Value linking (no LLM calls, so no tokens)
        value_linked_tables_and_columns, value_tokens = self._value_linker.link(
            data_item, 
            self._llm,
            schema_metadata=schema_metadata,
            join_relationships=join_relationships
        )
        total_token_usage["prompt_tokens"] += value_tokens["prompt_tokens"]
        total_token_usage["completion_tokens"] += value_tokens["completion_tokens"]
        total_token_usage["total_tokens"] += value_tokens["total_tokens"]
        
        merged_linked_tables_and_columns = merge_schema_linking_results([direct_linked_tables_and_columns, reversed_linked_tables_and_columns, value_linked_tables_and_columns])
        data_item.direct_linked_tables_and_columns = direct_linked_tables_and_columns
        data_item.reversed_linked_tables_and_columns = reversed_linked_tables_and_columns
        data_item.value_linked_tables_and_columns = value_linked_tables_and_columns
        data_item.final_linked_tables_and_columns = merged_linked_tables_and_columns
        data_item.database_schema_after_schema_linking = filter_used_database_schema(data_item.database_schema_after_value_retrieval, merged_linked_tables_and_columns)
        
        end_time = time.time()
        data_item.schema_linking_time = end_time - start_time
        data_item.schema_linking_llm_cost = total_token_usage
        data_item.total_time += data_item.schema_linking_time
        data_item.total_llm_cost = {
            "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.schema_linking_llm_cost["prompt_tokens"],
            "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.schema_linking_llm_cost["completion_tokens"],
            "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.schema_linking_llm_cost["total_tokens"],
        }
        
    def run(self):
        all_futures = []
        for data_item in self._dataset:
            if hasattr(data_item, "final_linked_tables_and_columns") and data_item.final_linked_tables_and_columns is not None:
                logger.info(f"Skipping data item {data_item.question_id} because it has already been linked")
                continue
            future = self._thread_pool_executor.submit(self._link_tables_and_columns, data_item)
            all_futures.append(future)
        for idx, future in tqdm(enumerate(as_completed(all_futures), start=1), total=len(all_futures), desc="Linking tables and columns"):
            future.result()
            if idx % 20 == 0:
                logger.info(f"Linking tables and columns {idx} / {len(all_futures)} completed")
                self.save_result()
        logger.info("Linking tables and columns completed")
        logger.info("Evaluating schema linking recall...")
        self._eval_schema_linking_recall()        
        logger.info("Schema linking recall evaluated")
        self.save_result()
        self._clean_up()
        
    
    def save_result(self):
        save_dataset(self._dataset, config.schema_linking_config.save_path)
        
    def _eval_schema_linking_recall(self):
        for data_item in self._dataset:
            gold_tables_and_columns = self._reversed_linker._extract_tables_and_columns(data_item.gold_sql, data_item.database_schema_after_value_retrieval)
            
            # eval direct linking recall
            direct_linking_table_recall = 0
            direct_linking_column_recall = 0
            for table_name, columns in data_item.direct_linked_tables_and_columns.items():
                if table_name in gold_tables_and_columns:
                    direct_linking_table_recall += 1
                    for column_name in columns:
                        if column_name in gold_tables_and_columns[table_name]:
                            direct_linking_column_recall += 1
            direct_linking_table_recall /= len(gold_tables_and_columns.keys()) if len(gold_tables_and_columns.keys()) > 0 else 1
            direct_linking_column_recall /= sum(len(columns) for columns in gold_tables_and_columns.values()) if sum(len(columns) for columns in gold_tables_and_columns.values()) > 0 else 1
            
            # eval reversed linking recall
            reversed_linking_table_recall = 0
            reversed_linking_column_recall = 0
            for table_name, columns in data_item.reversed_linked_tables_and_columns.items():
                if table_name in gold_tables_and_columns:
                    reversed_linking_table_recall += 1
                    for column_name in columns:
                        if column_name in gold_tables_and_columns[table_name]:
                            reversed_linking_column_recall += 1
            reversed_linking_table_recall /= len(gold_tables_and_columns.keys()) if len(gold_tables_and_columns.keys()) > 0 else 1
            reversed_linking_column_recall /= sum(len(columns) for columns in gold_tables_and_columns.values()) if sum(len(columns) for columns in gold_tables_and_columns.values()) > 0 else 1
            
            # eval value linking recall
            value_linking_table_recall = 0
            value_linking_column_recall = 0
            for table_name, columns in data_item.value_linked_tables_and_columns.items():
                if table_name in gold_tables_and_columns:
                    value_linking_table_recall += 1
                    for column_name in columns:
                        if column_name in gold_tables_and_columns[table_name]:
                            value_linking_column_recall += 1
            value_linking_table_recall /= len(gold_tables_and_columns.keys()) if len(gold_tables_and_columns.keys()) > 0 else 1
            value_linking_column_recall /= sum(len(columns) for columns in gold_tables_and_columns.values()) if sum(len(columns) for columns in gold_tables_and_columns.values()) > 0 else 1
            
            # eval final linking recall
            final_linking_table_recall = 0
            final_linking_column_recall = 0
            for table_name, columns in data_item.final_linked_tables_and_columns.items():
                if table_name in gold_tables_and_columns:
                    final_linking_table_recall += 1
                    for column_name in columns:
                        if column_name in gold_tables_and_columns[table_name]:
                            final_linking_column_recall += 1
            final_linking_table_recall /= len(gold_tables_and_columns.keys()) if len(gold_tables_and_columns.keys()) > 0 else 1
            final_linking_column_recall /= sum(len(columns) for columns in gold_tables_and_columns.values()) if sum(len(columns) for columns in gold_tables_and_columns.values()) > 0 else 1
            
            data_item.direct_linking_recall = {"table_recall": direct_linking_table_recall, "column_recall": direct_linking_column_recall}
            data_item.reversed_linking_recall = {"table_recall": reversed_linking_table_recall, "column_recall": reversed_linking_column_recall}
            data_item.value_linking_recall = {"table_recall": value_linking_table_recall, "column_recall": value_linking_column_recall}
            data_item.final_linking_recall = {"table_recall": final_linking_table_recall, "column_recall": final_linking_column_recall}
    
    def _clean_up(self):
        if self._thread_pool_executor is not None:
            self._thread_pool_executor.shutdown(wait=True)
            self._thread_pool_executor = None
        if self._caf_system is not None:
            try:
                self._caf_system.cleanup()
            except:
                pass
            self._caf_system = None
        self._llm = None
        self._dataset = None
        self._direct_linker = None
        self._reversed_linker = None
        self._value_linker = None