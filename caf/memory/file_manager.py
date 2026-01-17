# File manager for Semantic Memory DataFrame persistence
# Based on the design document: docs/6_Memory_Design.md

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FileManager:
    """File manager - handles DataFrame persistence"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FileManager initialized with storage path: {storage_path}")
    
    def load_dataframes(self, database_id: str) -> Dict[str, pd.DataFrame]:
        """Load all metadata DataFrames for a database, filtering by database_id to prevent cross-database data mixing"""
        dataframes = {}
        metadata_types = ['database', 'table', 'column', 'relationship', 'term', 'field_versions']
        
        # Create database-specific directory
        db_dir = self.storage_path / database_id
        db_dir.mkdir(parents=True, exist_ok=True)
        
        for metadata_type in metadata_types:
            file_path = db_dir / f"{metadata_type}.pkl"
            if file_path.exists():
                try:
                    df = pd.read_pickle(file_path)
                    
                    # Filter DataFrame to only include rows for the current database_id
                    # This prevents cross-database data contamination from existing files
                    if 'database_id' in df.columns:
                        filtered_df = df[df['database_id'] == database_id].copy()
                        if len(filtered_df) < len(df):
                            logger.warning(
                                f"Filtered {metadata_type} DataFrame for {database_id}: "
                                f"removed {len(df) - len(filtered_df)} rows from other databases "
                                f"(total: {len(df)} -> filtered: {len(filtered_df)})"
                            )
                            # Auto-save the cleaned DataFrame to prevent future contamination
                            filtered_df.to_pickle(file_path)
                            logger.info(f"Auto-cleaned and saved {metadata_type} DataFrame for {database_id}")
                        dataframes[metadata_type] = filtered_df
                    else:
                        # For metadata types without database_id column, use as-is
                        dataframes[metadata_type] = df
                        logger.debug(f"Loaded {metadata_type} DataFrame for {database_id} (no database_id filter applied)")
                    
                    logger.debug(f"Loaded {metadata_type} DataFrame for {database_id} ({len(dataframes[metadata_type])} rows)")
                except Exception as e:
                    logger.warning(f"Failed to load {metadata_type} DataFrame for {database_id}: {e}")
                    dataframes[metadata_type] = pd.DataFrame()
            else:
                dataframes[metadata_type] = pd.DataFrame()
                logger.debug(f"No existing {metadata_type} DataFrame for {database_id}, created empty DataFrame")
        
        logger.info(f"Loaded {len([df for df in dataframes.values() if not df.empty])} non-empty DataFrames for {database_id}")
        return dataframes
    
    def save_dataframes(self, database_id: str, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Save all DataFrames for a database, filtering by database_id to prevent cross-database data mixing"""
        saved_count = 0
        
        # Create database-specific directory
        db_dir = self.storage_path / database_id
        db_dir.mkdir(parents=True, exist_ok=True)
        
        for metadata_type, df in dataframes.items():
            if not df.empty:
                # Filter DataFrame to only include rows for the current database_id
                # This prevents cross-database data contamination
                if 'database_id' in df.columns:
                    filtered_df = df[df['database_id'] == database_id].copy()
                else:
                    # For metadata types without database_id column, save as-is
                    # (though this should be rare)
                    filtered_df = df.copy()
                    logger.warning(f"DataFrame '{metadata_type}' does not have 'database_id' column, saving all rows")
                
                if not filtered_df.empty:
                    file_path = db_dir / f"{metadata_type}.pkl"
                    try:
                        filtered_df.to_pickle(file_path)
                        saved_count += 1
                        logger.debug(f"Saved {metadata_type} DataFrame for {database_id} ({len(filtered_df)} rows, filtered from {len(df)} total rows)")
                    except Exception as e:
                        logger.error(f"Failed to save {metadata_type} DataFrame for {database_id}: {e}")
                else:
                    logger.debug(f"Skipping empty filtered DataFrame for {metadata_type} (database_id: {database_id})")
        
        logger.info(f"Saved {saved_count} DataFrames for {database_id}")
    
    def delete_database_data(self, database_id: str) -> None:
        """Delete all DataFrames for a database"""
        db_dir = self.storage_path / database_id
        deleted_count = 0
        
        if db_dir.exists():
            metadata_types = ['database', 'table', 'column', 'relationship', 'term', 'field_versions']
            
            for metadata_type in metadata_types:
                file_path = db_dir / f"{metadata_type}.pkl"
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted {metadata_type} DataFrame for {database_id}")
                    except Exception as e:
                        logger.error(f"Failed to delete {metadata_type} DataFrame for {database_id}: {e}")
            
            # Remove the database directory if it's empty
            try:
                if not any(db_dir.iterdir()):
                    db_dir.rmdir()
                    logger.debug(f"Removed empty database directory: {db_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove database directory {db_dir}: {e}")
        
        logger.info(f"Deleted {deleted_count} DataFrame files for {database_id}")
    
    def list_databases(self) -> List[str]:
        """List all databases that have stored metadata"""
        databases = []
        
        for db_dir in self.storage_path.iterdir():
            if db_dir.is_dir():
                # Check if this directory contains any metadata files
                metadata_files = list(db_dir.glob("*.pkl"))
                if metadata_files:
                    databases.append(db_dir.name)
        
        return sorted(databases)
    
    def get_database_info(self, database_id: str) -> Dict[str, Any]:
        """Get information about stored DataFrames for a database"""
        info = {
            'database_id': database_id,
            'metadata_types': {},
            'total_files': 0,
            'total_rows': 0
        }
        
        db_dir = self.storage_path / database_id
        metadata_types = ['database', 'table', 'column', 'relationship', 'term', 'field_versions']
        
        for metadata_type in metadata_types:
            file_path = db_dir / f"{metadata_type}.pkl"
            if file_path.exists():
                try:
                    df = pd.read_pickle(file_path)
                    info['metadata_types'][metadata_type] = {
                        'rows': len(df),
                        'columns': len(df.columns) if not df.empty else 0,
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                    info['total_files'] += 1
                    info['total_rows'] += len(df)
                except Exception as e:
                    info['metadata_types'][metadata_type] = {'error': str(e)}
            else:
                info['metadata_types'][metadata_type] = None
        
        return info
