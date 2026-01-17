# Query Analyzer - Extract entities and constraints from natural language queries

import logging
from typing import List, Optional
from dataclasses import dataclass

from .dependency_parser import StanzaDependencyParser, HybridDependencyParser
from caf.llm.client import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class EntityGroup:
    """Entity group with base entity and constraints for unified retrieval"""
    full_phrase: str
    base_phrase: str  # Base entity phrase (without nummod)
    constraints: List[str]  # List of constraint phrases
    
    # Backward compatibility properties
    @property
    def base_entity(self) -> str:
        """Backward compatibility: return base_phrase"""
        return self.base_phrase
    
    @property
    def filters(self) -> List[str]:
        """Backward compatibility: return constraints"""
        return self.constraints


@dataclass
class IntentAnalysis:
    """Simplified intent analysis containing only entity groups"""
    entity_groups: List[EntityGroup]


class QueryAnalyzer:
    """
    Query Analyzer - Extract entities and constraints from natural language queries
    
    This class provides functionality to:
    1. Analyze natural language queries using dependency parsing
    2. Extract entity groups (base entities and their constraints)
    3. Extract search terms from intent analysis
    """
    
    def __init__(self, 
                 llm_client: BaseLLMClient,
                 config: Optional[dict] = None):
        """
        Initialize Query Analyzer
        
        Args:
            llm_client: LLM client for hybrid dependency parsing
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize dependency parser
        # First initialize Stanza parser
        stanza_parser = StanzaDependencyParser(
            lang=self.config.get('dependency_parser_lang', 'en'),
            use_gpu=self.config.get('dependency_parser_use_gpu', False)
        )
        
        # Then initialize Hybrid parser with LLM client
        self.dependency_parser = HybridDependencyParser(
            stanza_parser=stanza_parser,
            llm_client=llm_client,
            model=self.config.get('dependency_parser_model', 'gpt-4o-mini'),
            temperature=self.config.get('dependency_parser_temperature', 0.1),
            max_tokens=self.config.get('dependency_parser_max_tokens', 2000)
        )
        
        logger.debug("QueryAnalyzer initialized")
    
    def analyze(self, user_question: str) -> IntentAnalysis:
        """
        Analyze query using dependency parsing to extract entities and constraints
        
        Args:
            user_question: Natural language question
            
        Returns:
            IntentAnalysis object containing entity groups
        """
        if not self.dependency_parser:
            raise ValueError("Dependency parser not initialized")
            
        logger.debug(f"Starting dependency parsing for query: {user_question[:100]}...")
        
        # Use HybridDependencyParser to parse the question
        parsing_result = self.dependency_parser.parse_question(user_question)
        
        # Convert ParsingResult to IntentAnalysis
        entity_groups = []
        
        if parsing_result.core_entities:
            for entity in parsing_result.core_entities:
                full_phrase = entity.get('full_phrase', entity.get('word', '')).strip()
                base_phrase = entity.get('base_phrase', entity.get('word', '')).strip()
                
                if not full_phrase or not base_phrase:
                    continue
                
                # Extract constraints as list of strings
                constraints = []
                if entity.get('constraints'):
                    for constraint in entity['constraints']:
                        constraint_phrase = constraint.get('full_phrase', constraint.get('word', '')).strip()
                        if constraint_phrase:
                            constraints.append(constraint_phrase)
                
                entity_groups.append(EntityGroup(
                    full_phrase=full_phrase,
                    base_phrase=base_phrase,
                    constraints=constraints
                ))
        
        logger.debug(f"Extracted {len(entity_groups)} entity groups from dependency parsing")
        
        return IntentAnalysis(
            entity_groups=entity_groups
        )
    
    def extract_search_terms(self, intent_analysis: IntentAnalysis) -> List[str]:
        """
        Extract all search terms from IntentAnalysis
        
        Args:
            intent_analysis: IntentAnalysis object
            
        Returns:
            List of search terms (base_phrase + all constraints)
        """
        terms = []
        for group in intent_analysis.entity_groups:
            # Add base phrase
            if group.base_phrase:
                terms.append(group.base_phrase)
            # Add all constraints
            terms.extend(group.constraints)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term and term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        logger.debug(f"Extracted {len(unique_terms)} unique search terms")
        return unique_terms
