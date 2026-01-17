from app.dataset import DataItem
from app.llm import LLM
from app.logger import logger
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

class BaseSchemaLinker(ABC):

    @abstractmethod
    def link(
        self, 
        data_item: DataItem, 
        llm: LLM, 
        sampling_budget: int = 1,
        schema_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        join_relationships: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[Dict[str, List[str]], Dict[str, int]]:
        pass
