# Search Engines for CAF Memory System

from .semantic import SemanticSearchEngine
from .episodic import EpisodicSearchEngine

# Register search engines
from ..base import SearchEngineRegistry

# Register available search engines
SearchEngineRegistry.register('semantic', SemanticSearchEngine)
SearchEngineRegistry.register('episodic', EpisodicSearchEngine)

__all__ = ['SemanticSearchEngine', 'EpisodicSearchEngine', 'SearchEngineRegistry']