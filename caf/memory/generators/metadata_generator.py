# Semantic Memory Metadata Generator
# Automatic metadata generation system with DDL analysis and LLM analysis

import logging
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import time
from datetime import datetime

from ..stores.semantic import SemanticMemoryStore
from ..types import VERSIONED_FIELDS, DatabaseMetadata, TableMetadata, ColumnMetadata, RelationshipMetadata, TermDefinition
from .ddl_analyzer import DDLAnalyzer
from .llm_analyzer import LLMAnalyzer
from .data_profiler import DataProfiler
from .schema_extractor import DatabaseSchemaExtractor
from .join_path_discovery import JoinPathDiscovery
from .semantic.requirement_registry import get_default_registry
from .semantic.requirement_system import GeneratorType

logger = logging.getLogger(__name__)

class GenerationPlan:
    """Plan for what metadata needs to be generated"""
    
    def __init__(self):
        self.needs_ddl_analysis = False
        self.needs_profiling = False
        self.needs_llm_analysis = False
        self.needs_join_path_discovery = False
        
        # DDL analysis needs
        self.ddl_needs = {
            'database': False,
            'tables': [],  # List of table names
            'columns': {},  # {table_name: [column_names]}
            'relationships': False
        }
        
        # Profiling needs
        self.profiling_needs = {
            'tables': [],  # List of table names needing stats
            'columns': {}  # {table_name: [column_names]}
        }
        
        # LLM analysis needs
        self.llm_needs = {
            'database': [],  # List[str]: fields needing LLM analysis for the database
            'tables': {},    # Dict[str, List[str]]: {table_name: [missing_fields]}
            'columns': {},   # Dict[str, Dict[str, List[str]]]: {table_name: {column_name: [missing_fields]}}
            'relationships': []  # List[Dict[str, Any]]: relationships needing business_meaning
        }
        
    def has_any_needs(self) -> bool:
        """Check if there are any generation needs"""
        return (self.needs_ddl_analysis or 
                self.needs_profiling or 
                self.needs_llm_analysis or
                self.needs_join_path_discovery)
    
class MetadataGenerationResult:
    """Result of metadata generation process"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.database_id: Optional[str] = None
        
        # Generation plan
        self.generation_plan: Optional[GenerationPlan] = None
        
        # Statistics
        self.ddl_generated = {'database': 0, 'table': 0, 'column': 0, 'relationship': 0}
        self.profiling_generated = {'database': 0, 'table': 0, 'column': 0, 'relationship': 0}
        self.llm_generated = {'database': 0, 'table': 0, 'column': 0, 'relationship': 0}
        self.conflicts_resolved = 0
        
        # Error tracking
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Detailed logs
        self.generation_logs: List[Dict[str, Any]] = []
    
    def add_log(self, level: str, message: str, metadata_type: str = None, source: str = None):
        """Add log entry"""
        self.generation_logs.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'metadata_type': metadata_type,
            'source': source
        })
    
    def finish(self):
        """Mark generation as finished"""
        self.end_time = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of generation results"""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            'database_id': self.database_id,
            'duration_seconds': duration,
            'generation_counts': {
                'ddl_analysis': self.ddl_generated,
                'data_profiling': self.profiling_generated,
                'llm_analysis': self.llm_generated
            },
            'conflicts_resolved': self.conflicts_resolved,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'error_messages': self.errors,
            'warning_messages': self.warnings
        }

class MetadataGenerator:
    """
    Main metadata generation system
    
    Coordinates automatic metadata generation using:
    1. DDL Analysis (highest priority)
    2. Data Profiling (highest priority) 
    3. LLM Analysis (only when no existing information)
    
    Features:
    - Source priority management
    - Conflict detection and resolution
    - Batch generation for performance
    - One-click generation interface
    """

    METADATA_MODELS = {
        'database': DatabaseMetadata,
        'table': TableMetadata,
        'column': ColumnMetadata,
        'relationship': RelationshipMetadata,
        'term': TermDefinition,
    }

    
    def __init__(self, semantic_store: SemanticMemoryStore, config: Dict[str, Any] = None, memory_config: Dict[str, Any] = None):
        self.semantic_store = semantic_store
        self.config = config or {}
        self.memory_config = memory_config or {}
        
        # Initialize analyzers
        self.ddl_analyzer = DDLAnalyzer(config.get('ddl', {}))
        self.data_profiler = DataProfiler(config.get('profiling', {}))
        self.llm_analyzer = LLMAnalyzer(config.get('llm', {}))
        self.schema_extractor = DatabaseSchemaExtractor()
        
        # Join path discovery config: try semantic.relationship_discovery.join_path_discovery first,
        # fallback to config.join_path_discovery for backward compatibility
        join_path_config = (
            self.memory_config.get('semantic', {})
            .get('relationship_discovery', {})
            .get('join_path_discovery', {})
        ) or config.get('join_path_discovery', {})
        self.join_path_discovery = JoinPathDiscovery(join_path_config)
        
        # Generation settings
        self.force_regenerate = config.get('force_regenerate', False)
        self.enable_ddl = config.get('enable_ddl_analysis', False)
        self.enable_llm_analysis = config.get('enable_llm_analysis', False)
        self.enable_profiling = config.get('enable_profiling', False)
        self.enable_join_path_discovery = config.get('enable_join_path_discovery', False)
        
        # Initialize requirement registry for unified requirement management
        self.requirement_registry = get_default_registry()
        
        logger.info("MetadataGenerator initialized")
    
    def generate_database_metadata(self, database_path: str) -> MetadataGenerationResult:
        """
        One-click metadata generation for entire database with intelligent pre-checking
        
        Args:
            database_path: Path to database file
            
        Returns:
            MetadataGenerationResult with detailed statistics
        """
        result = MetadataGenerationResult()
        
        try:
            # Validate database path
            db_path = Path(database_path)
            if not db_path.exists():
                raise FileNotFoundError(f"Database file not found: {database_path}")
            
            # Extract actual database_id from database file
            database_id = self.schema_extractor.extract_database_id(database_path)
            result.database_id = database_id
            result.add_log('info', f'Starting metadata generation for database: {database_id}')
            logger.info(f"Starting metadata generation for database: {database_id}")
            
            self.semantic_store.bind_database(database_id)
            
            # Step 0: Create generation plan (pre-check what needs to be generated)
            logger.info("Step 0: Creating generation plan...")
            generation_plan = self._create_generation_plan(database_path, database_id)
            result.generation_plan = generation_plan
            
            if not generation_plan.has_any_needs():
                logger.info("No metadata generation needed - all data already exists")
                result.add_log('info', 'No metadata generation needed - all data already exists')
                result.finish()
                return result
            
            # Step 1: DDL Analysis (highest priority) - only if needed
            if generation_plan.needs_ddl_analysis and self.enable_ddl:
                logger.info("Step 1: Analyzing database DDL...")
                self._generate_from_ddl(database_path, database_id, generation_plan, result)
            else:
                result.add_log('info', 'DDL analysis skipped - all DDL metadata already exists')
            
            # Step 2: Data Profiling (highest priority) - only if needed
            if generation_plan.needs_profiling and self.enable_profiling:
                logger.info("Step 2: Performing data profiling...")
                self._generate_from_profiling(database_path, database_id, generation_plan, result)
            elif not self.enable_profiling:
                result.add_log('info', 'Data profiling disabled in configuration')
            else:
                result.add_log('info', 'Data profiling skipped - all profiling data already exists')
            
            # Step 3: Join Path Discovery (discover non-FK join relationships) - BEFORE LLM
            # This is necessary because LLM table analysis needs similar_columns.json from join path discovery
            if generation_plan.needs_join_path_discovery and self.enable_join_path_discovery:
                logger.info("Step 3: Discovering join paths...")
                self._generate_from_join_path_discovery(database_path, database_id, generation_plan, result)
            elif not self.enable_join_path_discovery:
                result.add_log('info', 'Join path discovery disabled in configuration')
            else:
                result.add_log('info', 'Join path discovery skipped - all join path relationships already exist')
            
            # Step 4: LLM Analysis (after Join Path Discovery)
            # This ensures LLM has access to:
            # 1. similar_columns.json for table analysis (unique columns identification)
            # 2. All discovered relationships (FK from DDL + non-FK from Join Path)
            if self.enable_llm_analysis:
                logger.info("Step 4: Performing LLM analysis with complete context...")
                llm_needs_dynamic = self._check_llm_needs_dynamic(database_path, database_id)
                
                if llm_needs_dynamic and any(llm_needs_dynamic.values()):
                    logger.info("Generating missing metadata with LLM...")
                    self._generate_from_llm(database_path, database_id, llm_needs_dynamic, result)
                else:
                    result.add_log('info', 'LLM analysis skipped - all LLM metadata already exists')
            else:
                result.add_log('info', 'LLM analysis disabled in configuration')

            result.finish()
            logger.info(f"Metadata generation completed for {database_id}")
            
        except Exception as e:
            error_msg = f"Metadata generation failed: {str(e)}"
            result.errors.append(error_msg)
            result.add_log('error', error_msg)
            logger.error(error_msg, exc_info=True)
            result.finish()
        
        return result
    
    def _create_generation_plan(self, database_path: str, database_id: str) -> GenerationPlan:
        """
        Create generation plan using unified requirement system
        
        This method uses the RequirementRegistry to check what metadata needs to be generated.
        The requirement system handles:
        - Checking for missing fields (None, NaN, empty string)
        - Force regenerate logic with source priority awareness
        - Grouping requirements by generator type
        """
        plan = GenerationPlan()
        
        try:
            # Get actual database structure for comparison
            actual_structure = self._get_actual_database_structure(database_path)
            
            # Use requirement registry to check needs
            needs = self.requirement_registry.check_needs(
                dataframes=self.semantic_store.dataframes,
                database_id=database_id,
                actual_structure=actual_structure,
                force_config=self.force_regenerate
            )
            print('------------Needs--------------------')
            print(needs)
            print('-'*100)
            
            # Convert unified format to generator-specific format
            # DDL needs
            if GeneratorType.DDL in needs and needs[GeneratorType.DDL]:
                plan.needs_ddl_analysis = True
                plan.ddl_needs = self._convert_to_simple_format(needs[GeneratorType.DDL])
            
            # Profiling needs
            if GeneratorType.PROFILING in needs and needs[GeneratorType.PROFILING]:
                plan.needs_profiling = True
                plan.profiling_needs = self._convert_to_simple_format(needs[GeneratorType.PROFILING])
            
            # LLM needs (use unified format directly)
            if GeneratorType.LLM in needs and needs[GeneratorType.LLM]:
                plan.needs_llm_analysis = True
                plan.llm_needs = needs[GeneratorType.LLM]
            
            # Join path discovery needs
            if GeneratorType.JOIN_PATH in needs and needs[GeneratorType.JOIN_PATH]:
                plan.needs_join_path_discovery = True

            logger.info(f"Generation plan created: DDL={plan.needs_ddl_analysis}, "
                       f"Profiling={plan.needs_profiling}, LLM={plan.needs_llm_analysis}, "
                       f"JoinPathDiscovery={plan.needs_join_path_discovery} (config: {'enabled' if self.enable_join_path_discovery else 'disabled'})")
            
        except Exception as e:
            logger.warning(f"Error creating generation plan with requirement system: {e}")
            logger.warning("Falling back to legacy check methods")
            # Default to full generation if all checks fail
            plan.needs_ddl_analysis = True
            plan.needs_profiling = True
            plan.needs_llm_analysis = True
        
        return plan
    
    def _convert_to_simple_format(self, unified_needs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert unified needs format to simple format for DDL/Profiling analyzers
        
        Unified format (from requirement_system):
        {
            'database': ['field1', 'field2'],
            'tables': {'table1': ['field1'], 'table2': ['field1']},
            'columns': {'table1': {'col1': ['field1'], 'col2': ['field1']}},
            'relationships': True or [list]
        }
        
        Simple format (for DDL/Profiling):
        {
            'database': True,  # Just need to know if database metadata is needed
            'tables': ['table1', 'table2'],  # List of table names
            'columns': {'table1': ['col1', 'col2']},  # Dict of table -> column names
            'relationships': True  # Boolean flag
        }
        """
        converted = {}
        
        # Database: Convert list of fields to boolean
        if 'database' in unified_needs and unified_needs['database']:
            converted['database'] = True
        
        # Tables: Extract keys (table names) from dict
        if 'tables' in unified_needs:
            tables = unified_needs['tables']
            if isinstance(tables, dict):
                converted['tables'] = list(tables.keys())
            elif isinstance(tables, list):
                converted['tables'] = tables  # Backward compatibility
        
        # Columns: Convert nested dict to simple dict
        if 'columns' in unified_needs:
            columns = unified_needs['columns']
            if isinstance(columns, dict):
                converted['columns'] = {}
                for table_name, cols_dict in columns.items():
                    if isinstance(cols_dict, dict):
                        # Extract column names from {col: [fields]} dict
                        converted['columns'][table_name] = list(cols_dict.keys())
                    elif isinstance(cols_dict, list):
                        # Already in simple format
                        converted['columns'][table_name] = cols_dict
        
        # Relationships: Convert to boolean or pass through list
        if 'relationships' in unified_needs:
            rels = unified_needs['relationships']
            if isinstance(rels, list):
                # For profiling, pass the list (contains relationship details for cardinality)
                converted['relationships'] = rels
            elif rels is True:
                converted['relationships'] = True
        
        return converted
    
    def _get_actual_database_structure(self, database_path: str) -> Dict[str, Any]:
        """Get actual database structure for comparison using schema extractor"""
        try:
            return self.schema_extractor.extract_actual_schema(database_path)
        except Exception as e:
            logger.warning(f"Failed to get actual database structure: {e}")
            return {'tables': [], 'columns': {}}
        
    def _has_database_metadata(self, database_id: str) -> bool:
        """Check if database metadata exists"""
        if 'database' not in self.semantic_store.dataframes:
            return False
        return not self.semantic_store.dataframes['database'].empty
    
    def _has_relationship_metadata(self, database_id: str) -> bool:
        """Check if relationship metadata exists"""
        if 'relationship' not in self.semantic_store.dataframes:
            return False
        return not self.semantic_store.dataframes['relationship'].empty
    
    def _needs_database_description(self, database_id: str) -> bool:
        """Check if database description needs LLM analysis"""
        if 'database' not in self.semantic_store.dataframes:
            return True
        
        db_df = self.semantic_store.dataframes['database']
        if db_df.empty:
            return True
        
        return pd.isnull(db_df.iloc[0].get('description'))
    
    def _needs_database_domain(self, database_id: str) -> bool:
        """Check if database domain needs LLM analysis"""
        if 'database' not in self.semantic_store.dataframes:
            return True
        
        db_df = self.semantic_store.dataframes['database']
        if db_df.empty:
            return True
        
        return pd.isnull(db_df.iloc[0].get('domain'))
    
    def _save_metadata(
        self,
        metadata_type: str,
        source: str,
        data: Dict[str, Any],
        **identifiers: Any
    ):
        versioned_field_defs = VERSIONED_FIELDS.get(metadata_type, [])
        non_versioned_data = {}
        
        if 'database_id' not in identifiers:
            logger.error(f"Programming error: database_id missing for _save_metadata")
            return

        # 1. 分离版本化和非版本化字段
        versioned_fields_to_save = []
        for field, value in data.items():
            if value is None:
                continue
            if field in versioned_field_defs:
                # Collect versioned fields to save AFTER base row is created
                version_kwargs = {}
                if metadata_type == 'table':
                    if 'table_name' in identifiers:
                        version_kwargs['table_name'] = identifiers['table_name']
                elif metadata_type == 'column':
                    if 'table_name' in identifiers:
                        version_kwargs['table_name'] = identifiers['table_name']
                    if 'column_name' in identifiers:
                        version_kwargs['column_name'] = identifiers['column_name']
                elif metadata_type == 'relationship':
                    # Use composite key as identifier (similar to semantic.py implementation)
                    if 'source_table' in identifiers and 'source_columns' in identifiers and 'target_table' in identifiers and 'target_columns' in identifiers:
                        rel_id = f"{identifiers['source_table']}.{identifiers['source_columns']}->{identifiers['target_table']}.{identifiers['target_columns']}"
                        version_kwargs['table_name'] = rel_id  # Reuse table_name parameter
                elif metadata_type == 'term':
                    if 'term_name' in identifiers:
                        version_kwargs['term_name'] = identifiers['term_name']
                
                versioned_fields_to_save.append({
                    'field': field,
                    'value': value,
                    'kwargs': version_kwargs
                })
            else:
                non_versioned_data[field] = value

        # 2. CRITICAL: Create base row FIRST before saving versioned fields
        # This ensures _update_main_table can find the row when updating versioned fields
        model_class = self.METADATA_MODELS.get(metadata_type)
        if not model_class:
            logger.warning(f"No model class found for metadata_type: {metadata_type}")
            return
            
        try:
            full_data = {**identifiers, **non_versioned_data}
            # Debug: Log relationship full_data
            if metadata_type == 'relationship':
                logger.debug(f"Creating {metadata_type} with full_data: {full_data}")
            metadata_obj = model_class(**full_data)
            
            # 使用 getattr 动态调用 semantic_store 的方法
            add_method_name = f"add_{metadata_type}_metadata"
            add_method = getattr(self.semantic_store, add_method_name, None)
            
            if add_method:
                # All metadata types now support source parameter (including relationship)
                # This creates the base row in the main table
                add_method(metadata_obj, source)
                
                # 3. NOW save versioned fields - base row exists, so _update_main_table can find it
                for versioned_field in versioned_fields_to_save:
                    self.semantic_store._add_field_version(
                        metadata_type=metadata_type,
                        field_name=versioned_field['field'],
                        field_value=versioned_field['value'],
                        source=source,
                        **versioned_field['kwargs']
                    )
                
                # Log success with field summary
                versioned_count = len(versioned_fields_to_save)
                non_versioned_count = len(non_versioned_data)
                logger.debug(
                    f"✅ Saved {metadata_type} metadata from {source}: "
                    f"{versioned_count} versioned + {non_versioned_count} non-versioned fields"
                )
            else:
                logger.error(f"Semantic store has no method named {add_method_name}")
                
        except Exception as e:
            # Improved error message with full context
            logger.error(
                f"Failed to save {metadata_type} metadata from {source}. "
                f"Identifiers: {identifiers}, Fields: {list(data.keys())}. Error: {e}",
                exc_info=True
            )

    def _process_and_save_results(
        self,
        generated_results: Dict[str, Any],
        source: str,
        database_id: str,
        result_tracker: Dict[str, int],
        result_object: MetadataGenerationResult
    ):
        """
        [新增] 统一处理并保存来自任何分析器的结果。
        此方法包含标准的循环逻辑，用于解析分析结果并调用 _save_metadata。
        """
        # 1. 处理 Database 级元数据
        if 'database' in generated_results and generated_results['database']:
            self._save_metadata(
                'database', source, generated_results['database'],
                database_id=database_id
            )
            result_tracker['database'] += len(generated_results['database'])
            result_object.add_log('info', 'Processed database metadata', 'database', source)

        # 2. 处理 Table 级元数据
        if 'tables' in generated_results:
            for table_name, table_data in generated_results['tables'].items():
                self._save_metadata(
                    'table', source, table_data,
                    database_id=database_id, table_name=table_name
                )
                result_tracker['table'] += 1
                result_object.add_log('debug', f'Processed table: {table_name}', 'table', source)
        
        # 3. 处理 Column 级元数据
        if 'columns' in generated_results:
            for table_name, columns_dict in generated_results['columns'].items():
                for column_name, column_data in columns_dict.items():
                    self._save_metadata(
                        'column', source, column_data,
                        database_id=database_id, table_name=table_name, column_name=column_name
                    )
                    result_tracker['column'] += 1
                    result_object.add_log('debug', f'Processed column: {table_name}.{column_name}', 'column', source)

        # 4. 处理 Relationship 级元数据
        if 'relationships' in generated_results:
            for rel_data in generated_results['relationships']:
                # Relationship 的标识符是其自身内容，无需额外传递
                self._save_metadata('relationship', source, rel_data, database_id=database_id)
                result_tracker['relationship'] += 1
                result_object.add_log('debug', f'Processed relationship: {rel_data.get("source_table")} -> {rel_data.get("target_table")}', 'relationship', source)

    def _generate_from_ddl(self, database_path: str, database_id: str, 
                                   generation_plan: GenerationPlan, result: MetadataGenerationResult) -> None:
        """从DDL分析中生成元数据 (已简化)。"""
        try:
            ddl_results = self.ddl_analyzer.analyze_database(
                database_path, needs=generation_plan.ddl_needs
            )
            # 统一处理结果
            self._process_and_save_results(
                ddl_results, 'ddl_extract', database_id, result.ddl_generated, result
            )
            # 立即保存到文件，避免中途出错导致数据丢失
            self.semantic_store.save_all_metadata()
            result.add_log('info', 'DDL analysis metadata saved to files', 'database', 'ddl_extract')
            logger.debug(f"DDL analysis metadata saved for database: {database_id}")
        except Exception as e:
            error_msg = f"DDL analysis failed: {str(e)}"
            result.errors.append(error_msg)
            result.add_log('error', error_msg)
            logger.error(error_msg, exc_info=True)
    
    def _generate_from_profiling(self, database_path: str, database_id: str,
                                        generation_plan: GenerationPlan, result: MetadataGenerationResult) -> None:
        """从数据分析中生成元数据 (已简化)。"""
        try:
            # Get relationships from requirement system
            relationships = None
            profiling_rels = generation_plan.profiling_needs.get('relationships')
            
            if profiling_rels:
                if isinstance(profiling_rels, list):
                    # Use the relationships list directly from requirement system
                    relationships = profiling_rels
                    logger.info(f"Found {len(relationships)} relationships to calculate cardinality from requirement system")
                elif profiling_rels is True:
                    # Fallback: Query from semantic store if only boolean flag
                    relationships = self._get_existing_relationships_for_profiling(database_id)
                    if relationships:
                        logger.info(f"Found {len(relationships)} relationships to calculate cardinality from semantic store")
            
            profiling_results = self.data_profiler.profile_database(
                database_path,
                tables=generation_plan.profiling_needs.get('tables'),
                columns=generation_plan.profiling_needs.get('columns'),
                relationships=relationships
            )
            # 统一处理结果
            self._process_and_save_results(
                profiling_results, 'data_profiling', database_id, result.profiling_generated, result
            )
            # 立即保存到文件，避免中途出错导致数据丢失
            self.semantic_store.save_all_metadata()
            result.add_log('info', 'Data profiling metadata saved to files', 'database', 'data_profiling')
            logger.debug(f"Data profiling metadata saved for database: {database_id}")
        except Exception as e:
            error_msg = f"Data profiling failed: {str(e)}"
            result.errors.append(error_msg)
            result.add_log('error', error_msg)
            logger.error(error_msg, exc_info=True)
    
    def _get_existing_relationships_for_profiling(self, database_id: str) -> List[Dict[str, Any]]:
        """Get existing relationships from semantic store that need cardinality calculation"""
        relationships = []
        
        if 'relationship' not in self.semantic_store.dataframes:
            return relationships
        
        relationship_df = self.semantic_store.dataframes['relationship']
        if relationship_df.empty:
            return relationships
        
        # Filter by database_id and check if cardinality is missing
        db_relationships = relationship_df[relationship_df['database_id'] == database_id]
        
        for _, row in db_relationships.iterrows():
            # Check if cardinality is missing (None, NaN, or empty string)
            cardinality = row.get('cardinality')
            if pd.isna(cardinality) or cardinality is None or (isinstance(cardinality, str) and cardinality.strip() == ''):
                rel_dict = {
                    'source_table': row.get('source_table'),
                    'target_table': row.get('target_table'),
                    'source_columns': row.get('source_columns'),
                    'target_columns': row.get('target_columns'),
                    'relationship_type': row.get('relationship_type'),
                    'source': row.get('source', 'ddl_extract')
                }
                # Ensure columns are lists
                if not isinstance(rel_dict['source_columns'], list):
                    rel_dict['source_columns'] = [rel_dict['source_columns']] if rel_dict['source_columns'] else []
                if not isinstance(rel_dict['target_columns'], list):
                    rel_dict['target_columns'] = [rel_dict['target_columns']] if rel_dict['target_columns'] else []
                
                relationships.append(rel_dict)
        
        return relationships
    
    def _generate_from_llm(self, database_path: str, database_id: str,
                                   llm_needs: Dict[str, Any], result: MetadataGenerationResult) -> None:
        """
        从LLM分析中生成元数据
        
        Args:
            llm_needs: LLM needs in unified format (from requirement_system or dynamic check)
        """
        try:
            context = self._prepare_llm_context(database_id)
            llm_results = self.llm_analyzer.analyze_database(
                context=context, needs=llm_needs
            )

            self._process_and_save_results(
                llm_results, 'llm_analysis', database_id, result.llm_generated, result
            )
            # 立即保存到文件，避免中途出错导致数据丢失
            self.semantic_store.save_all_metadata()
            result.add_log('info', 'LLM analysis metadata saved to files', 'database', 'llm_analysis')
            logger.debug(f"LLM analysis metadata saved for database: {database_id}")
        except Exception as e:
            error_msg = f"LLM analysis failed: {str(e)}"
            result.errors.append(error_msg)
            result.add_log('error', error_msg)
            logger.error(error_msg, exc_info=True)
    
    def _generate_from_join_path_discovery(self, database_path: str, database_id: str,
                                                  generation_plan: GenerationPlan, result: MetadataGenerationResult) -> None:
        """从连接路径发现中生成关系元数据"""
        try:
            # Get column metadata from semantic store
            column_df = None
            if 'column' in self.semantic_store.dataframes and not self.semantic_store.dataframes['column'].empty:
                column_df = self.semantic_store.dataframes['column']
            else:
                logger.warning("No column metadata available for join path discovery")
                result.add_log('warning', 'Join path discovery skipped - no column metadata available')
                return
            
            # Discover join paths
            relationships = self.join_path_discovery.discover_join_paths(
                database_path=database_path,
                database_id=database_id,
                column_df=column_df,
                exclude_fk_relationships=True
            )
            
            if relationships:
                # Convert to the format expected by _process_and_save_results
                join_path_results = {
                    'relationships': relationships
                }
                
                join_path_tracker = {'database': 0, 'table': 0, 'column': 0, 'relationship': 0}
                
                # 统一处理结果
                self._process_and_save_results(
                    join_path_results, 'join_path_discovery', database_id, join_path_tracker, result
                )
                
                # 立即保存到文件，避免中途出错导致数据丢失
                self.semantic_store.save_all_metadata()
                result.add_log('info', f'Discovered {len(relationships)} join paths', 'relationship', 'join_path_discovery')
                result.add_log('info', 'Join path discovery metadata saved to files', 'relationship', 'join_path_discovery')
                logger.debug(f"Join path discovery metadata saved for database: {database_id}")
            else:
                result.add_log('info', 'No join paths discovered', 'relationship', 'join_path_discovery')
                
        except Exception as e:
            error_msg = f"Join path discovery failed: {str(e)}"
            result.errors.append(error_msg)
            result.add_log('error', error_msg)
            logger.error(error_msg, exc_info=True)

    def _check_llm_needs_dynamic(self, database_path: str, database_id: str) -> Dict[str, Any]:
        """
        Dynamically check LLM needs after DDL/Profiling/JOIN_PATH completion
        
        This is necessary because:
        1. Relationship metadata depends on DDL/JOIN_PATH discovery results
        2. Table analysis needs similar_columns.json from JOIN_PATH
        
        By checking after all prior generators have run, we ensure LLM has complete context.
        
        Args:
            database_path: Path to database file (needed for structure extraction)
            database_id: Database identifier
            
        Returns:
            LLM needs in unified format (same as requirement_system output)
        """
        try:
            # Get actual database structure for comparison
            actual_structure = self._get_actual_database_structure(database_path)
            
            # Use requirement registry to check only LLM needs
            all_needs = self.requirement_registry.check_needs(
                dataframes=self.semantic_store.dataframes,
                database_id=database_id,
                actual_structure=actual_structure,
                force_config=self.force_regenerate
            )
            
            # Extract only LLM needs
            from .semantic.requirement_system import GeneratorType
            llm_needs = all_needs.get(GeneratorType.LLM, {})
            
            logger.debug(f"Dynamic LLM needs check found: {llm_needs}")
            return llm_needs
            
        except Exception as e:
            logger.warning(f"Error in dynamic LLM needs check: {e}")
            # Return empty needs on error (conservative approach)
            return {}
    
    def _prepare_llm_context(self, database_id: str) -> Dict[str, Any]:
        """
        Prepare context information for LLM analysis
        
        NOTE: This method reads from self.semantic_store.dataframes, which contains
        the latest in-memory data from all previously completed generators (DDL, Profiling).
        This ensures LLM analysis uses the most up-to-date metadata.
        """

        context = {
            'database_id': database_id,
            'tables': {},
            'columns': {},
            'relationships': []
        }
        
        # Populate database-level description and domain if available
        if 'database' in self.semantic_store.dataframes and not self.semantic_store.dataframes['database'].empty:
            db_df = self.semantic_store.dataframes['database']

            db_row = db_df.iloc[0]
            if 'description' in db_row and pd.notna(db_row['description']):
                context['description'] = db_row['description']
            if 'domain' in db_row and pd.notna(db_row['domain']):
                context['domain'] = db_row['domain']

        # Populate table information
        if 'table' in self.semantic_store.dataframes and not self.semantic_store.dataframes['table'].empty:
            table_df = self.semantic_store.dataframes['table']
            for _, row in table_df.iterrows():
                table_name = row['table_name']
                context['tables'][table_name] = {
                    'table_name': table_name,
                    'primary_keys': row.get('primary_keys', []),
                    'foreign_keys': row.get('foreign_keys', []),
                    'sample_data': row.get('sample_data', {}),
                    'row_count': row.get('row_count', 0),
                    'column_count': row.get('column_count', 0),
                    'table_role': row.get('table_role', None),
                    'description': row.get('description', None),
                    'row_definition': row.get('row_definition', None),
                    'fk_in_degree': None if pd.isna(row.get('fk_in_degree', 0)) else int(row.get('fk_in_degree', 0)),
                    'fk_out_degree': None if pd.isna(row.get('fk_out_degree', 0)) else int(row.get('fk_out_degree', 0)),
                    'columns': []
                }
                
        
        # Populate column information with all necessary stats for intelligent prompting
        if 'column' in self.semantic_store.dataframes and not self.semantic_store.dataframes['column'].empty:
            column_df = self.semantic_store.dataframes['column']
            for _, row in column_df.iterrows():
                table_name = row['table_name']
                column_name = row['column_name']
                
                if table_name not in context['columns']:
                    context['columns'][table_name] = {}
                
                # For LLM, we only need a small set of representative values.
                # Use keys of top_k_values as value examples.
                top_k = row.get('top_k_values') if 'top_k_values' in row else None
                example_values = list(top_k.keys()) if isinstance(top_k, dict) else None

                # Include ALL column metadata fields for mechanical description generation
                # Convert NaN/None to None for consistent handling
                def safe_get(key):
                    val = row.get(key)
                    return None if pd.isna(val) else val
                
                context['columns'][table_name][column_name] = {
                    'column_name': column_name,
                    'data_type': safe_get('data_type'),
                    'semantic_type': safe_get('semantic_type'),
                    'null_count': safe_get('null_count'),
                    'distinct_count': safe_get('distinct_count'),
                    'min_value': safe_get('min_value'),
                    'max_value': safe_get('max_value'),
                    'min_length': safe_get('min_length'),
                    'max_length': safe_get('max_length'),
                    'avg_length': safe_get('avg_length'),
                    'fixed_prefix': safe_get('fixed_prefix'),
                    'fixed_suffix': safe_get('fixed_suffix'),
                    'top_k_values': top_k,
                    'pattern_description': safe_get('pattern_description'),
                    'description': safe_get('description'),
                    'distinct_values': example_values,  # Note: This is derived from top_k_values, not from the removed distinct_values field
                }
                
                if table_name in context['tables']:
                    context['tables'][table_name]['columns'].append(column_name)
        
        # Populate relationship information (for relationship semantics analysis)
        if 'relationship' in self.semantic_store.dataframes and not self.semantic_store.dataframes['relationship'].empty:
            relationship_df = self.semantic_store.dataframes['relationship']
            db_relationships = relationship_df[relationship_df['database_id'] == database_id]
            for _, rel_row in db_relationships.iterrows():
                context['relationships'].append({
                    'source_table': rel_row.get('source_table'),
                    'target_table': rel_row.get('target_table'),
                    'source_columns': rel_row.get('source_columns', []),
                    'target_columns': rel_row.get('target_columns', []),
                    'cardinality': rel_row.get('cardinality'),
                    'relationship_type': rel_row.get('relationship_type'),
                    'business_meaning': rel_row.get('business_meaning'),
                    'source': rel_row.get('source', 'ddl_extract')
                })

        return context
        