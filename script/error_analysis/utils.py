"""
Utility functions for error analysis
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


def save_jsonl(data: list, output_path: Path):
    """
    Save list of objects to JSONL file
    
    Args:
        data: List of objects (should be JSON serializable)
        output_path: Output file path
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Convert Pydantic models to dict if needed
                if hasattr(item, 'model_dump'):
                    item_dict = item.model_dump()
                elif hasattr(item, 'dict'):
                    item_dict = item.dict()
                else:
                    item_dict = item
                
                json.dump(item_dict, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(data)} items to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSONL file: {e}")


def load_jsonl(input_path: Path) -> list:
    """
    Load JSONL file into list of dicts
    
    Args:
        input_path: Input file path
    
    Returns:
        List of dicts
    """
    data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} items from {input_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSONL file: {e}")
        return []


def save_json(data: Any, output_path: Path, indent: int = 2):
    """
    Save object to JSON file
    
    Args:
        data: Object to save
        output_path: Output file path
        indent: JSON indentation
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Pydantic models to dict if needed
        if hasattr(data, 'model_dump'):
            data_dict = data.model_dump()
        elif hasattr(data, 'dict'):
            data_dict = data.dict()
        else:
            data_dict = data
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=indent)
        
        logger.info(f"Saved JSON to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file: {e}")


def load_json(input_path: Path) -> Any:
    """
    Load JSON file
    
    Args:
        input_path: Input file path
    
    Returns:
        Loaded object
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON from {input_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        return None


def get_timestamp() -> str:
    """Get current timestamp as ISO format string"""
    return datetime.utcnow().isoformat() + 'Z'


def setup_logging(log_level: str = "INFO", log_file: Path = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def format_sample_summary(sample: Dict[str, Any]) -> str:
    """
    Format a sample for logging/display
    
    Args:
        sample: ProcessedSample dict
    
    Returns:
        Formatted string
    """
    lines = [
        f"Question ID: {sample.get('question_id')}",
        f"NLQ: {sample.get('nlq', '')[:100]}...",
        f"Status: {sample.get('processing_status')}",
    ]
    
    if sample.get('is_pure_schema_error') is not None:
        lines.append(f"Pure Schema Error: {sample.get('is_pure_schema_error')}")
        lines.append(f"Schema Overlap: {sample.get('schema_overlap_score', 0):.3f}")
    
    if sample.get('verification_passed') is not None:
        lines.append(f"Verification Passed: {sample.get('verification_passed')}")
    
    return '\n'.join(lines)















