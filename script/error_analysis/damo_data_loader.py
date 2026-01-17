"""
Data loader for DAMO format results

This loader handles the format from gpt_4o_mini_damo_wo_memory_results_on_train_set.json
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .config import Config
from .models import ProcessedSample

logger = logging.getLogger(__name__)


class DAMODataLoader:
    """
    Load and process data from DAMO format results
    
    Format:
    {
      "results": [
        {
          "question_id": 0,
          "db_id": "movie_platform",
          "question": "...",
          "evidence": "...",
          "ground_truth_sql": "...",
          "generated_sql": "...",
          "execution_success": true,
          "ex_correct": false,  // false = incorrect
          ...
        }
      ]
    }
    """
    
    def __init__(self, 
                 results_path: Path,
                 database_mapping_path: Path = None):
        """
        Initialize DAMO data loader
        
        Args:
            results_path: Path to DAMO results JSON file
            database_mapping_path: Path to database mapping file (db_path -> db_id)
        """
        self.results_path = results_path
        
        # Use default mapping path if not provided
        if database_mapping_path is None:
            database_mapping_path = Config.PROJECT_ROOT / "memory/database_mapping.json"
        
        self.database_mapping_path = database_mapping_path
        self.db_id_to_path = {}  # Reverse mapping: db_id -> db_path
    
    def load_data(self) -> List[ProcessedSample]:
        """
        Load and process DAMO results
        
        Returns:
            List of ProcessedSample objects (only incorrect ones)
        """
        logger.info("Loading DAMO format data...")
        
        # Load database mapping
        self._load_database_mapping()
        
        # Load results
        results_data = self._load_results()
        
        # Filter and convert to ProcessedSample
        samples = self._process_results(results_data)
        
        logger.info(f"Loaded {len(samples)} incorrect samples from DAMO results")
        return samples
    
    def _load_database_mapping(self):
        """Load and reverse database mapping"""
        logger.info(f"Loading database mapping from {self.database_mapping_path}")
        
        if not self.database_mapping_path.exists():
            logger.warning(f"Database mapping file not found: {self.database_mapping_path}")
            logger.warning("Will try to infer database paths from db_id")
            return
        
        with open(self.database_mapping_path, 'r', encoding='utf-8') as f:
            path_to_id_mapping = json.load(f)
        
        # Reverse mapping: db_id -> db_path
        self.db_id_to_path = {
            db_id: db_path 
            for db_path, db_id in path_to_id_mapping.items()
        }
        
        logger.info(f"Loaded mapping for {len(self.db_id_to_path)} databases")
    
    def _load_results(self) -> Dict[str, Any]:
        """Load DAMO results JSON file"""
        logger.info(f"Loading results from {self.results_path}")
        
        with open(self.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'results' not in data:
            raise ValueError("Invalid DAMO format: missing 'results' key")
        
        total_results = len(data['results'])
        logger.info(f"Found {total_results} total results")
        
        return data
    
    def _process_results(self, data: Dict[str, Any]) -> List[ProcessedSample]:
        """
        Process results and convert to ProcessedSample objects
        
        Filters:
        - Only incorrect results (ex_correct == false)
        - Only results with valid database paths
        """
        samples = []
        
        total_count = 0
        incorrect_count = 0
        missing_db_count = 0
        
        for result in data['results']:
            total_count += 1
            
            # Only process incorrect results
            if result.get('ex_correct', True):
                continue
            
            incorrect_count += 1
            
            # Get database path
            db_id = result.get('db_id', '')
            db_path = self._resolve_database_path(db_id)
            
            if not db_path:
                missing_db_count += 1
                logger.warning(f"Could not resolve database path for db_id: {db_id} (question_id: {result.get('question_id')})")
                continue
            
            # Create ProcessedSample
            sample = ProcessedSample(
                question_id=result['question_id'],
                nlq=result['question'],
                evidence=result.get('evidence', ''),
                incorrect_sql=result['generated_sql'],
                correct_sql=result['ground_truth_sql'],
                db_path=db_path
            )
            
            samples.append(sample)
        
        logger.info(f"Statistics:")
        logger.info(f"  Total results: {total_count}")
        logger.info(f"  Incorrect results: {incorrect_count}")
        logger.info(f"  Missing database paths: {missing_db_count}")
        logger.info(f"  Processed samples: {len(samples)}")
        
        return samples
    
    def _resolve_database_path(self, db_id: str) -> str:
        """
        Resolve db_id to database file path
        
        Args:
            db_id: Database identifier (e.g., "movie_platform")
        
        Returns:
            Database file path, or empty string if not found
        """
        if not db_id:
            return ""
        
        # Try direct lookup
        if db_id in self.db_id_to_path:
            return self.db_id_to_path[db_id]
        
        # Try to construct path (fallback)
        # Assume train_databases by default for DAMO results
        possible_paths = [
            Config.PROJECT_ROOT / f"dataset/bird/databases/train_databases/{db_id}/{db_id}.sqlite",
            Config.PROJECT_ROOT / f"dataset/bird/databases/dev_databases/{db_id}/{db_id}.sqlite",
            Config.PROJECT_ROOT / f"dataset/bird/train/train_databases/{db_id}/{db_id}.sqlite",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.debug(f"Inferred database path for {db_id}: {path}")
                return str(path)
        
        return ""
    
    @staticmethod
    def count_incorrect(results_path: Path) -> Dict[str, int]:
        """
        Quick count of incorrect results in file
        
        Returns:
            Dict with 'total', 'incorrect', 'correct' counts
        """
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        total = len(results)
        incorrect = sum(1 for r in results if not r.get('ex_correct', True))
        correct = total - incorrect
        
        return {
            'total': total,
            'incorrect': incorrect,
            'correct': correct
        }















