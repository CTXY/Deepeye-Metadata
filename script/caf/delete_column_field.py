#!/usr/bin/env python3
"""
Script to delete a specific field from metadata in SemanticMemoryStore

This script allows you to delete (clear) the value of a specific field for database,
table, column, or term metadata. Optionally, it can also remove historical versions of the field.

Usage:
    # Database-level metadata
    python script/caf/delete_column_field.py --database_id <db_id> --metadata_type database --field_name <field> [options]
    
    # Table-level metadata
    python script/caf/delete_column_field.py --database_id financial --metadata_type column --table_name district --column_name A8 --field_name description
        
    # Column-level metadata
    python script/caf/delete_column_field.py --database_id <db_id> --metadata_type column --table_name <table> --column_name <column> --field_name <field> [options]
    
    # Term-level metadata
    python script/caf/delete_column_field.py --database_id <db_id> --metadata_type term --term_name <term> --field_name <field> [options]

Examples:
    # Delete database description
    python script/caf/delete_column_field.py --database_id california_schools --metadata_type database --field_name description

    # Delete table description
    python script/caf/delete_column_field.py --database_id california_schools --metadata_type table --table_name frpm --field_name description

    # Delete column description
    python script/caf/delete_column_field.py --database_id california_schools --metadata_type column --table_name frpm --column_name Zip --field_name long_description

    # Delete term definition
    python script/caf/delete_column_field.py --database_id california_schools --metadata_type term --term_name "charter school" --field_name definition

    # Delete field and remove all historical versions
    python script/caf/delete_column_field.py --database_id california_schools --metadata_type column --table_name frpm --column_name Zip --field_name long_description --remove_versions

    # Delete without saving (dry run)
    python script/caf/delete_column_field.py --database_id california_schools --metadata_type table --table_name frpm --field_name description --no-save
"""

import sys
import argparse
import logging
from pathlib import Path

import caf
from caf.memory.types import MemoryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_field(
    database_id: str,
    metadata_type: str,
    field_name: str,
    table_name: str = None,
    column_name: str = None,
    term_name: str = None,
    remove_versions: bool = False,
    save: bool = True,
    config_path: str = "config/caf_config.yaml"
):
    """
    Delete a specific field from metadata
    
    Args:
        database_id: Database ID
        metadata_type: Type of metadata ('database', 'table', 'column', 'term')
        field_name: Field name to delete
        table_name: Table name (required for 'table' and 'column' types)
        column_name: Column name (required for 'column' type)
        term_name: Term name (required for 'term' type)
        remove_versions: Whether to remove historical versions
        save: Whether to save changes immediately
        config_path: Path to CAF config file
    """
    logger.info("="*60)
    logger.info("Delete Metadata Field")
    logger.info("="*60)
    logger.info(f"Database ID: {database_id}")
    logger.info(f"Metadata Type: {metadata_type}")
    logger.info(f"Field: {field_name}")
    
    if table_name:
        logger.info(f"Table: {table_name}")
    if column_name:
        logger.info(f"Column: {column_name}")
    if term_name:
        logger.info(f"Term: {term_name}")
    
    logger.info(f"Remove versions: {remove_versions}")
    logger.info(f"Save changes: {save}")
    logger.info("-"*60)
    
    # Validate metadata type
    if metadata_type not in ['database', 'table', 'column', 'term']:
        logger.error(f"Invalid metadata_type: {metadata_type}. Must be one of: database, table, column, term")
        sys.exit(1)
    
    # Validate required parameters based on metadata type
    if metadata_type == 'table' and not table_name:
        logger.error("table_name is required for table metadata")
        sys.exit(1)
    if metadata_type == 'column' and (not table_name or not column_name):
        logger.error("table_name and column_name are required for column metadata")
        sys.exit(1)
    if metadata_type == 'term' and not term_name:
        logger.error("term_name is required for term metadata")
        sys.exit(1)
    
    # Initialize CAF system
    logger.info("Initializing CAF system...")
    caf_system = caf.initialize(config_path=config_path)
    logger.info("✓ CAF system initialized")
    
    # Build description for logging
    if metadata_type == 'database':
        desc = f"database '{database_id}'"
    elif metadata_type == 'table':
        desc = f"table '{table_name}'"
    elif metadata_type == 'column':
        desc = f"column '{table_name}.{column_name}'"
    elif metadata_type == 'term':
        desc = f"term '{term_name}'"
    
    # Execute deletion via CAFSystem public API
    logger.info(f"Deleting field '{field_name}' for {desc}...")
    success = caf_system.delete_semantic_field(
        database_id=database_id,
        metadata_type=metadata_type,
        field_name=field_name,
        table_name=table_name,
        column_name=column_name,
        term_name=term_name,
        save=save,
        remove_versions=remove_versions
    )
    
    if success:
        logger.info("✓ Field deleted successfully")
        if not save:
            logger.warning("⚠ Changes not saved (--no-save flag used). Use --save to persist changes.")
    else:
        logger.error(f"✗ Failed to delete field. {desc.capitalize()} may not exist or field may not be in schema.")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("Operation completed")
    logger.info("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Delete a specific field from metadata in SemanticMemoryStore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete database description
  %(prog)s --database_id california_schools --metadata_type database --field_name description

  # Delete table description
  %(prog)s --database_id california_schools --metadata_type table --table_name frpm --field_name description

  # Delete column long_description
  %(prog)s --database_id california_schools --metadata_type column --table_name frpm --column_name Zip --field_name long_description

  # Delete term definition
  %(prog)s --database_id california_schools --metadata_type term --term_name "charter school" --field_name definition

  # Delete field and remove all historical versions
  %(prog)s --database_id california_schools --metadata_type column --table_name frpm --column_name Zip --field_name long_description --remove_versions

  # Dry run (don't save)
  %(prog)s --database_id california_schools --metadata_type table --table_name frpm --field_name description --no-save
        """
    )
    
    parser.add_argument(
        '--database_id',
        type=str,
        required=True,
        help='Database ID'
    )
    
    parser.add_argument(
        '--metadata_type',
        type=str,
        required=True,
        choices=['database', 'table', 'column', 'term'],
        help='Type of metadata: database, table, column, or term'
    )
    
    parser.add_argument(
        '--table_name',
        type=str,
        default=None,
        help='Table name (required for table and column metadata)'
    )
    
    parser.add_argument(
        '--column_name',
        type=str,
        default=None,
        help='Column name (required for column metadata)'
    )
    
    parser.add_argument(
        '--term_name',
        type=str,
        default=None,
        help='Term name (required for term metadata)'
    )
    
    parser.add_argument(
        '--field_name',
        type=str,
        required=True,
        help='Field name to delete (e.g., description, long_description, definition, domain)'
    )
    
    parser.add_argument(
        '--remove_versions',
        action='store_true',
        help='Also remove historical versions of this field from field_versions table'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save changes immediately (dry run mode)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/caf_config.yaml',
        help='Path to CAF config file (default: config/caf_config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute deletion
    delete_field(
        database_id=args.database_id,
        metadata_type=args.metadata_type,
        field_name=args.field_name,
        table_name=args.table_name,
        column_name=args.column_name,
        term_name=args.term_name,
        remove_versions=args.remove_versions,
        save=not args.no_save,
        config_path=args.config
    )


if __name__ == '__main__':
    main()
