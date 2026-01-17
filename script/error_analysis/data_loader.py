"""
Data loading utilities
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import Config
from .models import ProcessedSample

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and merge data from multiple sources"""
    
    def __init__(self, 
                 bird_dev_path: Path = Config.BIRD_DEV_JSON,
                 incorrect_results_path: Path = Config.INCORRECT_RESULTS_JSON):
        self.bird_dev_path = bird_dev_path
        self.incorrect_results_path = incorrect_results_path
        
        self.nlq_map: Dict[int, Dict[str, Any]] = {}
        self.incorrect_samples: List[Dict[str, Any]] = []
    
    def load_data(self) -> List[ProcessedSample]:
        """
        Load and merge data from bird dev.json and incorrect results
        
        Returns:
            List of ProcessedSample objects
        """
        logger.info("Loading data...")
        
        # Load bird dev.json (contains NLQ)
        self._load_bird_dev()
        
        # Load incorrect results
        self._load_incorrect_results()
        
        # Merge data
        samples = self._merge_data()
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def _load_bird_dev(self):
        """Load bird dev.json file"""
        logger.info(f"Loading NLQ data from {self.bird_dev_path}")
        
        with open(self.bird_dev_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            question_id = item['question_id']
            self.nlq_map[question_id] = {
                'question': item['question'],
                'evidence': item.get('evidence', ''),
                'db_id': item.get('db_id', ''),
                'difficulty': item.get('difficulty', '')
            }
        
        logger.info(f"Loaded {len(self.nlq_map)} NLQ entries")
    
    def _load_incorrect_results(self):
        """Load incorrect results JSON"""
        logger.info(f"Loading incorrect results from {self.incorrect_results_path}")
        
        with open(self.incorrect_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.incorrect_samples = data.get('incorrect_sqls', [])
        logger.info(f"Loaded {len(self.incorrect_samples)} incorrect samples")
    
    def _merge_data(self) -> List[ProcessedSample]:
        """Merge NLQ and incorrect SQL data"""
        samples = []
        
        for item in self.incorrect_samples:
            question_id = item['question_id']
            
            # Get corresponding NLQ
            nlq_data = self.nlq_map.get(question_id)
            if not nlq_data:
                logger.warning(f"No NLQ found for question_id {question_id}, skipping")
                continue
            
            # Create ProcessedSample
            sample = ProcessedSample(
                question_id=question_id,
                nlq=nlq_data['question'],
                evidence=nlq_data.get('evidence', ''),
                incorrect_sql=item['pred_sql'],
                correct_sql=item['gold_sql'],
                db_path=item['db_path']
            )
            
            samples.append(sample)
        
        return samples

