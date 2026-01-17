# Metadata Requirement System - Unified requirement management for metadata generation
# 
# This system provides a centralized way to:
# 1. Define what metadata fields need to be generated
# 2. Check which fields are missing/empty
# 3. Track which generator should generate each field
# 4. Handle force_regenerate logic with source priority awareness

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field

from caf.memory.types import SOURCE_PRIORITY, VERSIONED_FIELDS

logger = logging.getLogger(__name__)


class GeneratorType(Enum):
    """Types of metadata generators"""
    DDL = "ddl"
    PROFILING = "profiling"
    LLM = "llm"
    JOIN_PATH = "join_path"


# Generator dependency graph
# Format: {generator: [dependencies]} - dependencies must run first
GENERATOR_DEPENDENCIES = {
    GeneratorType.DDL: [],  # No dependencies
    GeneratorType.PROFILING: [GeneratorType.DDL],  # Needs DDL for table/column structure
    GeneratorType.JOIN_PATH: [GeneratorType.DDL, GeneratorType.PROFILING],  # Needs column metadata for similarity analysis
    GeneratorType.LLM: [GeneratorType.DDL, GeneratorType.PROFILING, GeneratorType.JOIN_PATH]  # Needs all prior data including similar_columns.json
}


@dataclass
class MetadataRequirement:
    """
    Defines a single metadata field requirement
    
    This class encapsulates:
    - What field needs to be generated
    - Which generator should generate it
    - How to check if it's missing
    - How to check if it should be regenerated (force_regenerate logic)
    """
    # Field identification
    metadata_type: str  # 'database', 'table', 'column', 'relationship'
    field_name: str
    
    # Generator information
    generator: GeneratorType
    source: str  # Source name for version control (e.g., 'ddl_extract', 'llm_analysis')
    
    # Priority information
    source_priority: int  # From SOURCE_PRIORITY
    
    # Check functions
    check_missing_fn: Callable[[pd.DataFrame, str, Optional[str], Optional[str]], bool]
    """Check if field is missing. Args: (df, database_id, table_name, column_name) -> bool"""
    
    # Force regenerate logic
    should_force_regenerate_fn: Optional[Callable[[pd.DataFrame, str, Optional[str], Optional[str], Dict[str, Any]], bool]] = None
    """
    Check if field should be force regenerated.
    Args: (df, database_id, table_name, column_name, force_config) -> bool
    
    force_config can be:
    - True: Force regenerate all fields from this generator
    - False: Don't force regenerate
    - List[str]: Force regenerate specific fields using generic format:
      * ['table.description'] - matches description field for ALL tables
      * ['column.description'] - matches description field for ALL columns
      * ['table.description', 'column.description'] - matches both
    - Dict[str, Any]: More complex rules (future extension)
    
    Default behavior: Only regenerate if current source has lower priority than this requirement's source
    """
    
    # Metadata
    description: str = ""  # Human-readable description
    is_versioned: bool = False  # Whether this field supports versioning
    
    def __post_init__(self):
        """Initialize derived fields"""
        # Check if field is versioned
        self.is_versioned = (
            self.metadata_type in VERSIONED_FIELDS and
            self.field_name in VERSIONED_FIELDS[self.metadata_type]
        )
        
        # Set default should_force_regenerate_fn if not provided
        if self.should_force_regenerate_fn is None:
            self.should_force_regenerate_fn = self._default_force_regenerate_check
    
    def _default_force_regenerate_check(
        self,
        df: pd.DataFrame,
        database_id: str,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
        force_config: Any = False
    ) -> bool:
        """
        Default force regenerate logic:
        - If force_config is True: Always regenerate
        - If force_config is a list: Check if this field is in the list
        - Otherwise: Only regenerate if current source has lower priority
        """
        # Check if force_config explicitly requests regeneration
        if force_config is True:
            return True
        
        if isinstance(force_config, list):
            # Check if this field is in the force list
            field_path = self._get_field_path(table_name, column_name)
            return field_path in force_config
        
        # Default: Only regenerate if current source has lower priority
        # This respects human-generated (manual) data
        current_source = self._get_current_source(df, database_id, table_name, column_name)
        current_priority = SOURCE_PRIORITY.get(current_source, 0)
        
        # Only regenerate if current priority is lower than this requirement's priority
        # This means:
        # - If current is 'manual' (priority 3) and this is 'llm_analysis' (priority 1), DON'T regenerate
        # - If current is 'llm_analysis' (priority 1) and this is 'ddl_extract' (priority 4), DO regenerate
        return current_priority < self.source_priority
    
    def _get_field_path(self, table_name: Optional[str], column_name: Optional[str]) -> str:
        """
        Get field path for force_config list matching.
        
        Always returns generic format (without table/column names) to match force_config list:
        - database.{field_name}
        - table.{field_name}
        - column.{field_name}
        - {metadata_type}.{field_name}
        
        This allows force_config like ['table.description', 'column.description'] to match
        all tables/columns, not just specific ones.
        """
        return f"{self.metadata_type}.{self.field_name}"
    
    def _get_current_source(
        self,
        df: pd.DataFrame,
        database_id: str,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None
    ) -> str:
        """
        Get current source for this field from field_versions
        
        Note: This method requires access to field_versions dataframe.
        In practice, this will be called from RequirementRegistry.check_needs()
        which has access to all dataframes including field_versions.
        """
        # This will be implemented when integrated with semantic store
        # For now, return 'unknown' - the actual implementation will query field_versions
        return 'unknown'
    
    def check_needs_generation(
        self,
        df: pd.DataFrame,
        database_id: str,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
        force_config: Any = False
    ) -> bool:
        """
        Check if this field needs to be generated
        
        Returns True if:
        1. Field is missing (check_missing_fn returns True), OR
        2. Force regenerate is requested (should_force_regenerate_fn returns True)
        """
        # Check if missing
        is_missing = self.check_missing_fn(df, database_id, table_name, column_name)
        
        if is_missing:
            return True
        
        # Check if should force regenerate
        if force_config:
            return self.should_force_regenerate_fn(
                df, database_id, table_name, column_name, force_config
            )
        
        return False


class RequirementRegistry:
    """
    Central registry for all metadata requirements
    
    This class:
    1. Registers all metadata requirements
    2. Provides methods to check what needs to be generated
    3. Groups requirements by generator type
    4. Provides debugging and inspection capabilities
    """
    
    def __init__(self):
        self.requirements: List[MetadataRequirement] = []
        self._by_generator: Dict[GeneratorType, List[MetadataRequirement]] = {
            gen: [] for gen in GeneratorType
        }
        self._by_metadata_type: Dict[str, List[MetadataRequirement]] = {}
    
    def register(self, requirement: MetadataRequirement) -> None:
        """Register a metadata requirement"""
        self.requirements.append(requirement)
        
        # Index by generator
        self._by_generator[requirement.generator].append(requirement)
        
        # Index by metadata type
        if requirement.metadata_type not in self._by_metadata_type:
            self._by_metadata_type[requirement.metadata_type] = []
        self._by_metadata_type[requirement.metadata_type].append(requirement)
        
        logger.debug(f"Registered requirement: {requirement.metadata_type}.{requirement.field_name} "
                    f"(generator: {requirement.generator.value}, source: {requirement.source})")
    
    def get_requirements(
        self,
        generator: Optional[GeneratorType] = None,
        metadata_type: Optional[str] = None
    ) -> List[MetadataRequirement]:
        """Get requirements filtered by generator and/or metadata type"""
        result = self.requirements
        
        if generator:
            result = [r for r in result if r.generator == generator]
        
        if metadata_type:
            result = [r for r in result if r.metadata_type == metadata_type]
        
        return result
    
    def _get_current_source(
        self,
        dataframes: Dict[str, pd.DataFrame],
        database_id: str,
        metadata_type: str,
        field_name: str,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None
    ) -> str:
        """Get current source for a field from field_versions dataframe"""
        field_versions_df = dataframes.get('field_versions', pd.DataFrame())
        if field_versions_df.empty:
            return 'unknown'
        
        # Build filter mask
        mask = (
            (field_versions_df['database_id'] == database_id) &
            (field_versions_df['metadata_type'] == metadata_type) &
            (field_versions_df['field_name'] == field_name)
        )
        
        if table_name and 'table_name' in field_versions_df.columns:
            mask &= (field_versions_df['table_name'] == table_name)
        if column_name and 'column_name' in field_versions_df.columns:
            mask &= (field_versions_df['column_name'] == column_name)
        
        matching = field_versions_df[mask]
        if matching.empty:
            return 'unknown'
        
        # Find highest priority source
        best_source = 'unknown'
        best_priority = 0
        for _, row in matching.iterrows():
            source = row.get('source', 'unknown')
            priority = SOURCE_PRIORITY.get(source, 0)
            if priority > best_priority:
                best_priority = priority
                best_source = source
        
        return best_source
    
    def check_needs(
        self,
        dataframes: Dict[str, pd.DataFrame],
        database_id: str,
        actual_structure: Dict[str, Any],
        force_config: Any = False
    ) -> Dict[GeneratorType, Dict[str, Any]]:
        """
        Check what metadata needs to be generated
        
        Args:
            dataframes: Dict of dataframes from semantic store
            database_id: Database ID
            actual_structure: Actual database structure from schema extractor
            force_config: Force regenerate configuration
        
        Returns:
            Dict mapping generator type to needs dict
        """
        needs = {
            gen: {} for gen in GeneratorType
        }
        
        # Group requirements by generator
        for generator in GeneratorType:
            generator_needs = self._check_generator_needs(
                generator, dataframes, database_id, actual_structure, force_config
            )
            if generator_needs:
                needs[generator] = generator_needs
        
        return needs
    
    def _check_generator_needs(
        self,
        generator: GeneratorType,
        dataframes: Dict[str, pd.DataFrame],
        database_id: str,
        actual_structure: Dict[str, Any],
        force_config: Any
    ) -> Dict[str, Any]:
        """Check needs for a specific generator"""
        requirements = self._by_generator[generator]
        if not requirements:
            return {}
        
        needs = {}
        
        # Process each requirement
        for req in requirements:
            # Check if should force regenerate (considering source priority)
            should_force = False
            if force_config:
                # Get current source
                current_source = self._get_current_source(
                    dataframes, database_id, req.metadata_type, req.field_name
                )
                current_priority = SOURCE_PRIORITY.get(current_source, 0)
                
                # Check force_config
                if force_config is True:
                    # Only regenerate if current source has lower priority
                    should_force = current_priority < req.source_priority
                elif isinstance(force_config, list):
                    # Check if field is in force list
                    field_path = req._get_field_path(None, None)
                    should_force = field_path in force_config
            
            if req.metadata_type == 'database':
                df = dataframes.get('database', pd.DataFrame())
                is_missing = req.check_missing_fn(df, database_id)
                
                if is_missing or should_force:
                    if 'database' not in needs:
                        needs['database'] = []
                    # Always append field_name to list for consistency
                    if req.field_name not in needs['database']:
                        needs['database'].append(req.field_name)
            
            elif req.metadata_type == 'table':
                # Get actual tables from database structure
                actual_tables = set(actual_structure.get('tables', []))
                
                # Get existing tables from dataframe (for checking missing fields)
                table_df = dataframes.get('table', pd.DataFrame())
                
                for table_name in actual_tables:
                    # Check if should force regenerate for this specific table
                    should_force_table = False
                    if force_config:
                        current_source = self._get_current_source(
                            dataframes, database_id, req.metadata_type, req.field_name, table_name=table_name
                        )
                        current_priority = SOURCE_PRIORITY.get(current_source, 0)
                        
                        if force_config is True:
                            should_force_table = current_priority < req.source_priority
                        elif isinstance(force_config, list):
                            field_path = req._get_field_path(table_name, None)
                            should_force_table = field_path in force_config
                    
                    is_missing = req.check_missing_fn(table_df, database_id, table_name=table_name)
                    
                    if is_missing or should_force_table:
                        if 'tables' not in needs:
                            needs['tables'] = {}
                        
                        # Always use dict format: {table_name: [field_names]}
                        if table_name not in needs['tables']:
                            needs['tables'][table_name] = []
                        if req.field_name not in needs['tables'][table_name]:
                            needs['tables'][table_name].append(req.field_name)
            
            elif req.metadata_type == 'column':
                # Get actual columns
                actual_columns = actual_structure.get('columns', {})
                
                # Get existing columns from dataframe
                column_df = dataframes.get('column', pd.DataFrame())
                
                # Check all columns
                for table_name, column_names in actual_columns.items():
                    for column_name in column_names:
                        # Check if should force regenerate for this specific column
                        should_force_column = False
                        if force_config:
                            current_source = self._get_current_source(
                                dataframes, database_id, req.metadata_type, req.field_name,
                                table_name=table_name, column_name=column_name
                            )
                            current_priority = SOURCE_PRIORITY.get(current_source, 0)
                            
                            if force_config is True:
                                should_force_column = current_priority < req.source_priority
                            elif isinstance(force_config, list):
                                field_path = req._get_field_path(table_name, column_name)
                                should_force_column = field_path in force_config
                        
                        is_missing = req.check_missing_fn(
                            column_df, database_id, table_name=table_name, column_name=column_name
                        )
                        
                        if is_missing or should_force_column:
                            if 'columns' not in needs:
                                needs['columns'] = {}
                            
                            # Always use nested dict format: {table: {col: [fields]}}
                            if table_name not in needs['columns']:
                                needs['columns'][table_name] = {}
                            if column_name not in needs['columns'][table_name]:
                                needs['columns'][table_name][column_name] = []
                            if req.field_name not in needs['columns'][table_name][column_name]:
                                needs['columns'][table_name][column_name].append(req.field_name)
            
            elif req.metadata_type == 'relationship':
                # Get existing relationships from dataframe
                relationship_df = dataframes.get('relationship', pd.DataFrame())
                
                if relationship_df.empty:
                    # No relationships exist yet - need to discover them first (DDL or JOIN_PATH)
                    if req.generator == GeneratorType.DDL or req.generator == GeneratorType.JOIN_PATH:
                        needs['relationships'] = True
                    continue
                
                # Filter relationships for this database
                db_relationships = relationship_df[relationship_df['database_id'] == database_id] if not relationship_df.empty and 'database_id' in relationship_df.columns else pd.DataFrame()
                
                if db_relationships.empty:
                    # No relationships for this database
                    if req.generator == GeneratorType.DDL or req.generator == GeneratorType.JOIN_PATH:
                        needs['relationships'] = True
                    continue
                
                # Check each relationship for missing fields
                for _, rel_row in db_relationships.iterrows():
                    # Check if should force regenerate for this specific relationship
                    should_force_rel = False
                    if force_config:
                        current_source = self._get_current_source(
                            dataframes, database_id, req.metadata_type, req.field_name,
                            table_name=f"{rel_row.get('source_table')}.{rel_row.get('source_columns')}->{rel_row.get('target_table')}.{rel_row.get('target_columns')}"
                        )
                        current_priority = SOURCE_PRIORITY.get(current_source, 0)
                        
                        if force_config is True:
                            should_force_rel = current_priority < req.source_priority
                        elif isinstance(force_config, list):
                            field_path = req._get_field_path(None, None)
                            should_force_rel = field_path in force_config
                    
                    # Check if field is missing
                    field_value = rel_row.get(req.field_name)
                    is_missing = pd.isna(field_value) or field_value is None or (isinstance(field_value, str) and field_value.strip() == '')
                    
                    if is_missing or should_force_rel:
                        if 'relationships' not in needs:
                            if generator == GeneratorType.LLM:
                                needs['relationships'] = []  # List of relationship dicts for LLM
                            else:
                                needs['relationships'] = True  # Boolean for DDL/Profiling
                        
                        # For LLM, track specific relationships and fields needed
                        if generator == GeneratorType.LLM:
                            rel_dict = {
                                'source_table': rel_row.get('source_table'),
                                'target_table': rel_row.get('target_table'),
                                'source_columns': rel_row.get('source_columns'),
                                'target_columns': rel_row.get('target_columns'),
                                'fields_needed': [req.field_name]
                            }
                            
                            # Check if this relationship is already in the list
                            existing_rel = None
                            for r in needs['relationships']:
                                if (r['source_table'] == rel_dict['source_table'] and
                                    r['target_table'] == rel_dict['target_table'] and
                                    str(r['source_columns']) == str(rel_dict['source_columns']) and
                                    str(r['target_columns']) == str(rel_dict['target_columns'])):
                                    existing_rel = r
                                    break
                            
                            if existing_rel:
                                # Add field to existing relationship
                                if req.field_name not in existing_rel['fields_needed']:
                                    existing_rel['fields_needed'].append(req.field_name)
                            else:
                                # Add new relationship
                                needs['relationships'].append(rel_dict)
        
        return needs
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all registered requirements"""
        summary = {
            'total_requirements': len(self.requirements),
            'by_generator': {
                gen.value: len(reqs) for gen, reqs in self._by_generator.items()
            },
            'by_metadata_type': {
                mt: len(reqs) for mt, reqs in self._by_metadata_type.items()
            },
            'requirements': [
                {
                    'metadata_type': req.metadata_type,
                    'field_name': req.field_name,
                    'generator': req.generator.value,
                    'source': req.source,
                    'source_priority': req.source_priority,
                    'is_versioned': req.is_versioned,
                    'description': req.description
                }
                for req in self.requirements
            ]
        }
        return summary


# ============================================================================
# Helper Functions for Common Check Patterns
# ============================================================================

def make_is_null_check(field_name: str) -> Callable:
    """Create a check function that checks if field is null/NaN/empty"""
    def check_fn(df: pd.DataFrame, database_id: str, 
                 table_name: Optional[str] = None, 
                 column_name: Optional[str] = None) -> bool:
        if df.empty or field_name not in df.columns:
            return True
        
        # Build filter mask
        mask = pd.Series(True, index=df.index)
        if 'database_id' in df.columns:
            mask &= (df['database_id'] == database_id)
        
        if table_name and 'table_name' in df.columns:
            mask &= (df['table_name'] == table_name)
        
        if column_name and 'column_name' in df.columns:
            mask &= (df['column_name'] == column_name)
        
        matching_rows = df[mask]
        if matching_rows.empty:
            return True
        
        # Check if field is null/NaN/empty
        for _, row in matching_rows.iterrows():
            value = row.get(field_name)
            if value is None:
                return True
            if pd.isna(value):
                return True
            if isinstance(value, str) and value.strip() == '':
                return True
        
        return False
    
    return check_fn


def make_existence_check() -> Callable:
    """Create a check function that checks if row exists"""
    def check_fn(df: pd.DataFrame, database_id: str,
                 table_name: Optional[str] = None,
                 column_name: Optional[str] = None) -> bool:
        if df.empty:
            return True
        
        # Build filter mask
        mask = pd.Series(True, index=df.index)
        if 'database_id' in df.columns:
            mask &= (df['database_id'] == database_id)
        
        if table_name and 'table_name' in df.columns:
            mask &= (df['table_name'] == table_name)
        
        if column_name and 'column_name' in df.columns:
            mask &= (df['column_name'] == column_name)
        
        matching_rows = df[mask]
        return matching_rows.empty
    
    return check_fn

