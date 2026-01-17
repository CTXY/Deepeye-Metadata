# Retrieval Candidate for episodic search results

from dataclasses import dataclass
from caf.memory.types import EpisodicRecord

@dataclass
class RetrievalCandidate:
    """Container for retrieval candidate with multiple scores"""
    record: EpisodicRecord
    nlq_similarity: float = 0.0
    sql_skeleton_similarity: float = 0.0
    final_score: float = 0.0
    retrieval_type: str = "unknown"  # "same_db" or "cross_db"
    explanation: str = ""  # Optional explanation for the score
