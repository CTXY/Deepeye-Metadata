# Matchers Module - Specialized matching components

from .bm25_matcher import BM25Matcher
from .value_matcher import ValueMatcher

# Conditionally import SPLADE matcher (requires PyTorch)
try:
    from .splade_matcher import SPLADEMatcher
    __all__ = [
        'BM25Matcher',
        'SPLADEMatcher', 
        'ValueMatcher'
    ]
except ImportError:
    SPLADEMatcher = None
    __all__ = [
        'BM25Matcher',
        'ValueMatcher'
    ]
