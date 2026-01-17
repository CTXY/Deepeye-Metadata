# Unified Path Configuration for CAF System
# This module provides centralized path management for the entire project

from pathlib import Path
from typing import Optional
import os

class PathConfig:
    """
    Centralized path configuration for the project
    
    This ensures all modules use consistent paths for:
    - Datasets (original data like BIRD)
    - Memory storage (semantic, episodic)
    - Database files
    """
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # ==================== Datasets (Original Data) ====================
    DATASETS_ROOT = PROJECT_ROOT / "dataset"
    BIRD_DATASET_PATH = DATASETS_ROOT / "bird"
    
    # ==================== Memory Storage (Unified) ====================
    MEMORY_ROOT = PROJECT_ROOT / "memory"
    
    # Semantic Memory - stores schema and domain knowledge
    SEMANTIC_MEMORY_PATH = MEMORY_ROOT / "semantic_memory"
    
    # Episodic Memory - stores historical interaction records
    EPISODIC_MEMORY_PATH = MEMORY_ROOT / "episodic_memory"
    
    
    # ==================== Database Files ====================
    DATA_ROOT = PROJECT_ROOT / "data"  # Keep for compatibility
    MEMORY_DB_PATH = DATA_ROOT / "memory.db"
    CAF_DB_PATH = DATA_ROOT / "caf.db"
    
    # ==================== Cache and Temporary Files ====================
    CACHE_ROOT = PROJECT_ROOT / "cache"
    
    # Cache subdirectories (following scheme A: unified cache structure)
    CACHE_EPISODIC = CACHE_ROOT / "episodic"
    CACHE_EPISODIC_EMBEDDINGS = CACHE_EPISODIC / "embeddings"
    CACHE_SEMANTIC = CACHE_ROOT / "semantic"
    CACHE_SEMANTIC_SPLADE = CACHE_SEMANTIC / "splade"
    CACHE_SEMANTIC_VECTOR = CACHE_SEMANTIC / "vector_indexes"
    CACHE_MODELS = CACHE_ROOT / "models"  # For embedding model downloads
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist"""
        directories = [
            cls.DATASETS_ROOT,
            cls.MEMORY_ROOT,
            cls.SEMANTIC_MEMORY_PATH,
            cls.EPISODIC_MEMORY_PATH,
            cls.DATA_ROOT,
            cls.CACHE_ROOT,
            cls.CACHE_EPISODIC,
            cls.CACHE_EPISODIC_EMBEDDINGS,
            cls.CACHE_SEMANTIC,
            cls.CACHE_SEMANTIC_SPLADE,
            cls.CACHE_SEMANTIC_VECTOR,
            cls.CACHE_MODELS,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_semantic_memory_path(cls, database_id: Optional[str] = None) -> Path:
        """Get semantic memory path for a specific database"""
        base_path = cls.SEMANTIC_MEMORY_PATH
        if database_id:
            return base_path / database_id
        return base_path
    
    @classmethod
    def get_episodic_memory_path(cls, database_id: Optional[str] = None) -> Path:
        """Get episodic memory path for a specific database"""
        base_path = cls.EPISODIC_MEMORY_PATH
        if database_id:
            return base_path / database_id
        return base_path
    
    
    @classmethod
    def to_absolute(cls, path: str) -> Path:
        """Convert a path string to absolute Path object"""
        p = Path(path)
        if p.is_absolute():
            return p
        return cls.PROJECT_ROOT / p
    
    @classmethod
    def get_database_mapping_path(cls) -> Path:
        """Get path to database mapping file

        Prefer a unified location under MEMORY_ROOT to avoid hardcoded
        absolute paths scattered across the codebase.
        """
        return cls.MEMORY_ROOT / "database_mapping.json"
    
    # ==================== Cache Path Methods ====================
    
    @classmethod
    def get_episodic_cache_path(cls) -> Path:
        """Get cache path for episodic memory embeddings"""
        return cls.CACHE_EPISODIC_EMBEDDINGS
    
    @classmethod
    def get_splade_cache_path(cls) -> Path:
        """Get cache path for SPLADE indexes"""
        return cls.CACHE_SEMANTIC_SPLADE
    
    @classmethod
    def get_vector_cache_path(cls) -> Path:
        """Get cache path for vector/FAISS indexes"""
        return cls.CACHE_SEMANTIC_VECTOR
    
    @classmethod
    def get_model_cache_path(cls) -> Path:
        """Get cache path for embedding models"""
        return cls.CACHE_MODELS


# Convenience functions for backward compatibility
def get_semantic_memory_path(database_id: Optional[str] = None) -> str:
    """Get semantic memory path as string"""
    return str(PathConfig.get_semantic_memory_path(database_id))


def get_episodic_memory_path(database_id: Optional[str] = None) -> str:
    """Get episodic memory path as string"""
    return str(PathConfig.get_episodic_memory_path(database_id))




def get_bird_dataset_path() -> str:
    """Get BIRD dataset path as string"""
    return str(PathConfig.BIRD_DATASET_PATH)


# Initialize directories on import
PathConfig.ensure_directories()





