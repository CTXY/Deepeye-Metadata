# Requirement Registry Initialization
# This module initializes all metadata requirements

import logging
import pandas as pd
from typing import Optional
from .requirement_system import (
    RequirementRegistry, MetadataRequirement, GeneratorType,
    make_is_null_check, make_existence_check
)
from caf.memory.types import SOURCE_PRIORITY

logger = logging.getLogger(__name__)


def create_default_registry() -> RequirementRegistry:
    """
    Create and initialize the default requirement registry with all metadata requirements
    
    This function registers all known metadata requirements:
    - DDL requirements (database, table, column structure)
    - Profiling requirements (statistics, sample data)
    - LLM requirements (descriptions, semantic analysis)
    """
    registry = RequirementRegistry()
    
    # ========================================================================
    # DDL Requirements (Highest Priority: 4)
    # ========================================================================
    
    # Database-level DDL
    registry.register(MetadataRequirement(
        metadata_type='database',
        field_name='database_id',  # Just existence check
        generator=GeneratorType.DDL,
        source='ddl_extract',
        source_priority=SOURCE_PRIORITY['ddl_extract'],
        check_missing_fn=make_existence_check(),
        description='Database identifier'
    ))
    
    # Table-level DDL
    registry.register(MetadataRequirement(
        metadata_type='table',
        field_name='column_count',
        generator=GeneratorType.DDL,
        source='ddl_extract',
        source_priority=SOURCE_PRIORITY['ddl_extract'],
        check_missing_fn=make_is_null_check('column_count'),
        description='Number of columns in table'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='table',
        field_name='primary_keys',
        generator=GeneratorType.DDL,
        source='ddl_extract',
        source_priority=SOURCE_PRIORITY['ddl_extract'],
        check_missing_fn=make_is_null_check('primary_keys'),
        description='Primary key columns'
    ))
    
    # Column-level DDL
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='data_type',
        generator=GeneratorType.DDL,
        source='ddl_extract',
        source_priority=SOURCE_PRIORITY['ddl_extract'],
        check_missing_fn=make_is_null_check('data_type'),
        description='Database data type'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='is_nullable',
        generator=GeneratorType.DDL,
        source='ddl_extract',
        source_priority=SOURCE_PRIORITY['ddl_extract'],
        check_missing_fn=make_is_null_check('is_nullable'),
        description='Whether column allows NULL values'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='is_primary_key',
        generator=GeneratorType.DDL,
        source='ddl_extract',
        source_priority=SOURCE_PRIORITY['ddl_extract'],
        check_missing_fn=make_is_null_check('is_primary_key'),
        description='Whether column is part of primary key'
    ))
    
    # Relationship DDL
    registry.register(MetadataRequirement(
        metadata_type='relationship',
        field_name='relationship_type',  # Just existence check
        generator=GeneratorType.DDL,
        source='ddl_extract',
        source_priority=SOURCE_PRIORITY['ddl_extract'],
        check_missing_fn=make_existence_check(),
        description='Foreign key relationships'
    ))
    
    # ========================================================================
    # Profiling Requirements (Highest Priority: 4)
    # ========================================================================
    
    # Table-level profiling
    registry.register(MetadataRequirement(
        metadata_type='table',
        field_name='row_count',
        generator=GeneratorType.PROFILING,
        source='data_profiling',
        source_priority=SOURCE_PRIORITY['data_profiling'],
        check_missing_fn=make_is_null_check('row_count'),
        description='Number of rows in table'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='table',
        field_name='sample_data',
        generator=GeneratorType.PROFILING,
        source='data_profiling',
        source_priority=SOURCE_PRIORITY['data_profiling'],
        check_missing_fn=make_is_null_check('sample_data'),
        description='Sample data for table'
    ))
    
    # Column-level profiling
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='null_count',
        generator=GeneratorType.PROFILING,
        source='data_profiling',
        source_priority=SOURCE_PRIORITY['data_profiling'],
        check_missing_fn=make_is_null_check('null_count'),
        description='Number of NULL values'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='distinct_count',
        generator=GeneratorType.PROFILING,
        source='data_profiling',
        source_priority=SOURCE_PRIORITY['data_profiling'],
        check_missing_fn=make_is_null_check('distinct_count'),
        description='Number of distinct values'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='top_k_values',
        generator=GeneratorType.PROFILING,
        source='data_profiling',
        source_priority=SOURCE_PRIORITY['data_profiling'],
        check_missing_fn=lambda df, db_id, table_name, column_name: (
            make_is_null_check('top_k_values')(df, db_id, table_name, column_name) or
            (lambda: (
                # Check if top_k_values is empty dict
                df.empty or 'top_k_values' not in df.columns or
                any(
                    isinstance(row.get('top_k_values'), dict) and len(row.get('top_k_values', {})) == 0
                    for _, row in df.iterrows()
                    if (table_name is None or row.get('table_name') == table_name) and
                       (column_name is None or row.get('column_name') == column_name)
                )
            ))()
        ),
        description='Most common values'
    ))
    
    # ========================================================================
    # LLM Requirements (Lowest Priority: 1)
    # ========================================================================
    
    # Database-level LLM
    registry.register(MetadataRequirement(
        metadata_type='database',
        field_name='description',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('description'),
        description='Database description'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='database',
        field_name='domain',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('domain'),
        description='Business domain'
    ))
    
    # Table-level LLM - Basic description
    registry.register(MetadataRequirement(
        metadata_type='table',
        field_name='description',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('description'),
        description='Table description'
    ))
    
    # Table-level LLM - Deep semantic fields
    registry.register(MetadataRequirement(
        metadata_type='table',
        field_name='table_role',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('table_role'),
        description='Table role (Fact/Dimension/Bridge/Lookup)'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='table',
        field_name='row_definition',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('row_definition'),
        description='Definition of what each row represents'
    ))
    
    # Column-level LLM
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='description',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('description'),
        description='Column semantic description'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='pattern_description',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('pattern_description'),
        description='Pattern description for structured data'
    ))
    
    # encoding_mapping is not generated by LLM - should be provided manually by humans
    # registry.register(MetadataRequirement(
    #     metadata_type='column',
    #     field_name='encoding_mapping',
    #     generator=GeneratorType.LLM,
    #     source='llm_analysis',
    #     source_priority=SOURCE_PRIORITY['llm_analysis'],
    #     check_missing_fn=lambda df, db_id, table_name, column_name: (
    #         # Only check if semantic_type is Categorical
    #         make_is_null_check('encoding_mapping')(df, db_id, table_name, column_name) and
    #         # TODO: Also check semantic_type == 'Categorical'
    #         # This requires access to the row to check semantic_type
    #         True  # Simplified for now
    #     ),
    #     description='Value encoding mapping for categorical columns'
    # ))
    
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='short_description',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('short_description'),
        description='Short natural language description (for schema linking)'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='column',
        field_name='long_description',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('long_description'),
        description='Long natural language description (for SQL generation)'
    ))
    
    
    registry.register(MetadataRequirement(
        metadata_type='relationship',
        field_name='cardinality',
        generator=GeneratorType.PROFILING,
        source='data_profiling',
        source_priority=SOURCE_PRIORITY['data_profiling'],
        check_missing_fn=make_is_null_check('cardinality'),
        description='Relationship cardinality (1:1, 1:N, N:1, N:M, Lookup) computed from actual JOIN queries'
    ))
    
    registry.register(MetadataRequirement(
        metadata_type='relationship',
        field_name='business_meaning',
        generator=GeneratorType.LLM,
        source='llm_analysis',
        source_priority=SOURCE_PRIORITY['llm_analysis'],
        check_missing_fn=make_is_null_check('business_meaning'),
        description='Business meaning of relationship (Identifies, Belongs to, Measures, References)'
    ))
    
    # Relationship-level join path discovery
    # Check if there are any relationships with source='join_path_discovery'
    def check_join_path_relationships(df: pd.DataFrame, database_id: str, 
                                       table_name: Optional[str] = None, 
                                       column_name: Optional[str] = None) -> bool:
        """Check if join_path_discovery relationships exist for this database"""
        if df.empty or 'database_id' not in df.columns:
            return True  # Need to generate if no relationships exist
        
        # Filter by database_id
        mask = df['database_id'] == database_id
        
        # Check if any relationship has source='join_path_discovery'
        if 'source' in df.columns:
            mask = mask & (df['source'] == 'join_path_discovery')
        
        matching = df[mask]
        return matching.empty  # Need to generate if no join_path_discovery relationships found
    
    registry.register(MetadataRequirement(
        metadata_type='relationship',
        field_name='relationship_type',  # Just existence check for join_path_discovery relationships
        generator=GeneratorType.JOIN_PATH,
        source='join_path_discovery',
        source_priority=SOURCE_PRIORITY.get('join_path_discovery', 2),
        check_missing_fn=check_join_path_relationships,
        description='Join path relationships discovered from value overlap'
    ))
    
    logger.info(f"Initialized requirement registry with {len(registry.requirements)} requirements")
    return registry


# Global registry instance
_default_registry: Optional[RequirementRegistry] = None


def get_default_registry() -> RequirementRegistry:
    """Get or create the default requirement registry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry

