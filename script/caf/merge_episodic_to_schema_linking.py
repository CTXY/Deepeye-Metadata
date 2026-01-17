#!/usr/bin/env python3
"""
Temporary script to merge episodic data from episodic_retrieval output 
into schema_linking output.

This script:
1. Loads episodic data from workspace/episodic_retrieval/bird/sub_dev.pkl
2. Loads schema_linking data from workspace/schema_linking/bird/sub_dev.pkl
3. Merges episodic fields (episodic_cases, episodic_hint, combined_hint, etc.) 
   from episodic data into schema_linking data
4. Saves the merged data back to workspace/episodic_retrieval/bird/sub_dev.pkl
"""

import sys
from pathlib import Path
import pickle
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.dataset import load_dataset, save_dataset
from app.logger import logger


def merge_episodic_data():
    """Merge episodic data into schema_linking data"""
    
    # Paths
    episodic_path = project_root / "workspace/episodic_retrieval/bird/sub_dev.pkl"
    schema_linking_path = project_root / "workspace/schema_linking/bird/sub_dev.pkl"
    output_path = project_root / "workspace/episodic_retrieval/bird/sub_dev.pkl"
    
    # Check if files exist
    if not episodic_path.exists():
        logger.error(f"Episodic data file not found: {episodic_path}")
        return False
    
    if not schema_linking_path.exists():
        logger.error(f"Schema linking data file not found: {schema_linking_path}")
        return False
    
    logger.info(f"Loading episodic data from: {episodic_path}")
    episodic_dataset = load_dataset(str(episodic_path))
    
    logger.info(f"Loading schema linking data from: {schema_linking_path}")
    schema_linking_dataset = load_dataset(str(schema_linking_path))
    
    # Create a mapping of question_id -> episodic data item
    episodic_map: Dict[int, Any] = {}
    for item in episodic_dataset:
        episodic_map[item.question_id] = item
    
    logger.info(f"Found {len(episodic_map)} items in episodic data")
    logger.info(f"Found {len(schema_linking_dataset)} items in schema linking data")
    
    # Merge episodic fields into schema_linking data
    merged_count = 0
    missing_count = 0
    
    episodic_fields = [
        'episodic_cases',
        'episodic_hint',
        'combined_hint',
        'episodic_retrieval_time',
        'episodic_retrieval_llm_cost'
    ]
    
    for schema_item in schema_linking_dataset:
        question_id = schema_item.question_id
        
        if question_id in episodic_map:
            episodic_item = episodic_map[question_id]
            
            # Copy episodic fields from episodic_item to schema_item
            for field in episodic_fields:
                if hasattr(episodic_item, field):
                    value = getattr(episodic_item, field)
                    setattr(schema_item, field, value)
            
            # Update combined_hint if episodic_hint exists
            # combined_hint should be evidence + episodic_hint
            if hasattr(episodic_item, 'episodic_hint') and episodic_item.episodic_hint:
                if schema_item.evidence:
                    schema_item.combined_hint = f"{schema_item.evidence}\n\n{episodic_item.episodic_hint}"
                else:
                    schema_item.combined_hint = episodic_item.episodic_hint
            elif not hasattr(schema_item, 'combined_hint') or not schema_item.combined_hint:
                # Fallback to evidence if no episodic_hint
                schema_item.combined_hint = schema_item.evidence or ""
            
            # Update total_time and total_llm_cost if episodic data exists
            if hasattr(episodic_item, 'episodic_retrieval_time') and episodic_item.episodic_retrieval_time:
                if schema_item.total_time is not None:
                    # Subtract old episodic time if it was already added
                    # (in case of re-running, we want to replace not add)
                    schema_item.total_time = schema_item.total_time
                else:
                    schema_item.total_time = 0.0
                schema_item.total_time += episodic_item.episodic_retrieval_time
            
            if hasattr(episodic_item, 'episodic_retrieval_llm_cost') and episodic_item.episodic_retrieval_llm_cost:
                if schema_item.total_llm_cost is not None:
                    # Update total_llm_cost
                    cost = episodic_item.episodic_retrieval_llm_cost
                    schema_item.total_llm_cost = {
                        "prompt_tokens": schema_item.total_llm_cost.get("prompt_tokens", 0) + cost.get("prompt_tokens", 0),
                        "completion_tokens": schema_item.total_llm_cost.get("completion_tokens", 0) + cost.get("completion_tokens", 0),
                        "total_tokens": schema_item.total_llm_cost.get("total_tokens", 0) + cost.get("total_tokens", 0),
                    }
                else:
                    schema_item.total_llm_cost = episodic_item.episodic_retrieval_llm_cost
            
            merged_count += 1
        else:
            # No episodic data for this question_id, set defaults
            schema_item.episodic_cases = []
            schema_item.episodic_hint = ""
            schema_item.combined_hint = schema_item.evidence or ""
            schema_item.episodic_retrieval_time = 0.0
            schema_item.episodic_retrieval_llm_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            missing_count += 1
    
    logger.info(f"Merged episodic data for {merged_count} items")
    logger.info(f"Missing episodic data for {missing_count} items (set to defaults)")
    
    # Save merged data
    logger.info(f"Saving merged data to: {output_path}")
    save_dataset(schema_linking_dataset, str(output_path))
    
    logger.info("Merge completed successfully!")
    return True


if __name__ == "__main__":
    success = merge_episodic_data()
    sys.exit(0 if success else 1)





