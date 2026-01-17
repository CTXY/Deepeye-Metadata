"""
Dependency Parser Module for Entity and Constraint Extraction

This module provides dependency parsing functionality to extract entities and their constraints
from natural language questions using Stanford Stanza and optional LLM refinement.
"""

import logging
import os
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    logging.warning("Stanza not available. Please install: pip install stanza")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Please install: pip install openai")

logger = logging.getLogger(__name__)


@dataclass
class DependencyRelation:
    """Represents a single dependency relation"""
    word: str
    head: int
    deprel: str
    pos: str
    lemma: str
    upos: str
    feats: Optional[str] = None


@dataclass
class ParsingResult:
    """Complete parsing result for a question"""
    question: str
    sentences: List[Dict[str, Any]]
    dependency_relations: List[List[DependencyRelation]]
    constituency_trees: Optional[List[str]] = None
    core_entities: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[List[Dict[str, Any]]] = None


class StanzaDependencyParser:
    """
    Dependency and Constituency Parser using Stanford Stanza
    
    This class provides methods to:
    1. Perform dependency parsing to identify head-dependent relationships
    2. Perform constituency parsing to identify phrase structures
    3. Extract core entities and their constraints from questions
    """
    
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        """
        Initialize the Stanza parser
        
        Args:
            lang: Language code (default: 'en' for English)
            use_gpu: Whether to use GPU acceleration
        """
        if not STANZA_AVAILABLE:
            raise ImportError("Stanza is not available. Please install: pip install stanza")
        
        self.lang = lang
        self.use_gpu = use_gpu
        
        logger.info(f"Initializing Stanza parser for language: {lang}")
        
        # Download models if needed (will auto-download if not present)
        try:
            # Initialize pipeline with both dependency and constituency parsing
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,pos,lemma,depparse,constituency',
                use_gpu=use_gpu,
                verbose=False
            )
            logger.info("Stanza pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Stanza pipeline: {e}")
            logger.info("Attempting to download models...")
            stanza.download(lang)
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,pos,lemma,depparse,constituency',
                use_gpu=use_gpu,
                verbose=False
            )
            logger.info("Stanza pipeline initialized successfully after download")
    
    def parse_question(self, question: str) -> ParsingResult:
        """
        Parse a question and extract syntactic structures
        
        Args:
            question: Natural language question string
            
        Returns:
            ParsingResult containing dependency relations, constituency trees, and extracted entities
        """
        logger.debug(f"Parsing question: {question[:100]}...")
        
        # Process the question
        doc = self.nlp(question)
        
        # Extract dependency relations
        dependency_relations = []
        sentences_data = []
        constituency_trees = []
        
        for sent in doc.sentences:
            # Dependency parsing
            sent_deps = []
            for word in sent.words:
                dep_rel = DependencyRelation(
                    word=word.text,
                    head=word.head,
                    deprel=word.deprel,
                    pos=word.xpos if hasattr(word, 'xpos') else word.pos,
                    lemma=word.lemma,
                    upos=word.upos,
                    feats=word.feats if hasattr(word, 'feats') else None
                )
                sent_deps.append(dep_rel)
            
            dependency_relations.append(sent_deps)
            
            # Sentence data
            sent_data = {
                'text': sent.text,
                'words': [word.text for word in sent.words],
                'dependencies': [
                    {
                        'word': word.text,
                        'head': word.head,
                        'head_text': sent.words[word.head - 1].text if word.head > 0 else 'ROOT',
                        'deprel': word.deprel,
                        'pos': word.xpos if hasattr(word, 'xpos') else word.pos,
                        'upos': word.upos,
                        'lemma': word.lemma
                    }
                    for word in sent.words
                ]
            }
            sentences_data.append(sent_data)
            
            # Constituency parsing
            if hasattr(sent, 'constituency') and sent.constituency:
                tree_str = str(sent.constituency)
                constituency_trees.append(tree_str)
            else:
                constituency_trees.append(None)
        
        # Extract core entities and constraints
        core_entities, constraints = self._extract_entities_and_constraints(doc)
        
        result = ParsingResult(
            question=question,
            sentences=sentences_data,
            dependency_relations=dependency_relations,
            constituency_trees=constituency_trees,
            core_entities=core_entities,
            constraints=constraints
        )
        
        return result
    
    def _extract_entities_and_constraints(self, doc) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract core entities and their constraints from parsed document
        
        This method identifies:
        - Core entities (nouns that are likely table/column names)
        - Constraints (prepositional phrases, relative clauses, etc.)
        
        Args:
            doc: Stanza Document object
            
        Returns:
            Tuple of (core_entities, constraints)
        """
        core_entities = []
        constraints = []
        
        for sent in doc.sentences:
            # Find root of the sentence
            root = None
            for word in sent.words:
                if word.head == 0:  # Root has head = 0
                    root = word
                    break
            
            # Identify core entities (nouns that are subjects, objects, or root)
            for word in sent.words:
                # Check if this is a potential core entity
                if word.upos in ['NOUN', 'PROPN']:
                    entity_info = {
                        'word': word.text,
                        'lemma': word.lemma,
                        'pos': word.upos,
                        'deprel': word.deprel,
                        'head': word.head,
                        'head_text': sent.words[word.head - 1].text if word.head > 0 else 'ROOT',
                        'constraints': []
                    }
                    
                    # Find constraints (children that are prepositional phrases, relative clauses, etc.)
                    # Also handle compound nouns properly
                    for child in sent.words:
                        if child.head == word.id:
                            # Skip compound relations as they're part of the entity itself
                            if child.deprel == 'compound':
                                continue
                            
                            # Skip nummod as it's part of the entity name itself
                            if child.deprel == 'nummod':
                                continue
                            
                            # Include prepositional phrases, noun modifiers, relative clauses, etc.
                            if child.deprel in ['prep', 'nmod', 'acl', 'acl:relcl', 'amod', 'det', 'case']:
                                # For prepositional phrases, extract the full PP including the object
                                if child.deprel == 'prep' or child.deprel == 'case':
                                    # Find the object of the preposition
                                    full_pp = self._extract_prepositional_phrase(sent, child)
                                    constraint_info = {
                                        'word': child.text,
                                        'deprel': child.deprel,
                                        'pos': child.upos,
                                        'lemma': child.lemma,
                                        'full_phrase': full_pp
                                    }
                                else:
                                    constraint_info = {
                                        'word': child.text,
                                        'deprel': child.deprel,
                                        'pos': child.upos,
                                        'lemma': child.lemma,
                                        'full_phrase': self._extract_phrase(sent, child)
                                    }
                                entity_info['constraints'].append(constraint_info)
                    
                    # Include compound nouns and numeric modifiers as part of the entity name
                    compound_parts = []
                    nummod_parts = []
                    for child in sent.words:
                        if child.head == word.id:
                            if child.deprel == 'compound':
                                compound_parts.append(child.text)
                            elif child.deprel == 'nummod':
                                # Numeric modifiers should be included in the entity name
                                nummod_parts.append(child.text)
                    
                    # Build full phrase: compound parts + main word + numeric modifiers
                    full_phrase_parts = compound_parts + [word.text] + nummod_parts
                    entity_info['full_phrase'] = ' '.join(full_phrase_parts)
                    
                    # Also store the base phrase without nummod for reference
                    if compound_parts:
                        entity_info['base_phrase'] = ' '.join(compound_parts + [word.text])
                    else:
                        entity_info['base_phrase'] = word.text
                    
                    if entity_info['constraints'] or word.deprel in ['nsubj', 'obj', 'dobj', 'nsubjpass', 'root']:
                        core_entities.append(entity_info)
            
            # Extract verb phrases and their constraints (e.g., "pay for transactions")
            for word in sent.words:
                # Identify verbs (especially root verbs)
                if word.upos == 'VERB' and (word.deprel == 'root' or word.head == 0):
                    verb_info = {
                        'verb': word.text,
                        'lemma': word.lemma,
                        'deprel': word.deprel,
                        'constraints': []
                    }
                    
                    # Find all modifiers of the verb (prepositional phrases, obliques, etc.)
                    for child in sent.words:
                        if child.head == word.id:
                            if child.deprel in ['prep', 'obl', 'case', 'advmod']:
                                if child.deprel == 'prep' or child.deprel == 'case':
                                    full_pp = self._extract_prepositional_phrase(sent, child)
                                    verb_info['constraints'].append({
                                        'type': child.deprel,
                                        'full_phrase': full_pp,
                                        'modifies': 'verb'
                                    })
                                else:
                                    verb_info['constraints'].append({
                                        'type': child.deprel,
                                        'full_phrase': self._extract_phrase(sent, child),
                                        'modifies': 'verb'
                                    })
                    
                    if verb_info['constraints']:
                        constraints.append({
                            'word': word.text,
                            'deprel': 'verb_phrase',
                            'pos': 'VERB',
                            'head': 0,
                            'head_text': 'ROOT',
                            'full_phrase': f"{word.text} {' '.join([c['full_phrase'] for c in verb_info['constraints']])}",
                            'verb_constraints': verb_info['constraints']
                        })
            
            # Extract standalone constraints (prepositional phrases, etc.)
            for word in sent.words:
                if word.deprel in ['prep', 'obl', 'advmod'] and word.upos != 'VERB':
                    # Check if this constraint is already captured in verb constraints
                    is_verb_constraint = False
                    for constraint in constraints:
                        if constraint.get('verb_constraints'):
                            for vc in constraint['verb_constraints']:
                                if word.text in vc.get('full_phrase', ''):
                                    is_verb_constraint = True
                                    break
                        if is_verb_constraint:
                            break
                    
                    if not is_verb_constraint:
                        constraint_info = {
                            'word': word.text,
                            'deprel': word.deprel,
                            'pos': word.upos,
                            'head': word.head,
                            'head_text': sent.words[word.head - 1].text if word.head > 0 else 'ROOT',
                            'full_phrase': self._extract_phrase(sent, word)
                        }
                        constraints.append(constraint_info)
        
        return core_entities, constraints
    
    def _extract_prepositional_phrase(self, sent, prep_word) -> str:
        """
        Extract the full prepositional phrase including the preposition and its object
        
        Args:
            sent: Stanza Sentence object
            prep_word: Preposition word (or case marker)
            
        Returns:
            Full prepositional phrase string
        """
        phrase_parts = [prep_word.text]
        
        # Find the object of the preposition (pobj)
        for word in sent.words:
            if word.head == prep_word.id and word.deprel in ['pobj', 'obj']:
                # Get the full phrase for the object (including compound nouns)
                obj_phrase = self._extract_phrase(sent, word)
                phrase_parts.append(obj_phrase)
                break
        
        return ' '.join(phrase_parts)
    
    def _extract_phrase(self, sent, word) -> str:
        """
        Extract the full phrase starting from a word, including compound nouns
        
        Args:
            sent: Stanza Sentence object
            word: Stanza Word object
            
        Returns:
            Full phrase string
        """
        # Collect all words in the phrase subtree
        phrase_words = []
        visited = set()
        
        def collect_subtree(w):
            """Recursively collect all words in the subtree"""
            if w.id in visited:
                return
            visited.add(w.id)
            
            # First, collect compound nouns (they should come before the head)
            compound_parts = []
            for child in sent.words:
                if child.head == w.id and child.deprel == 'compound':
                    collect_subtree(child)
                    compound_parts.append((child.id, child.text))
            
            # Add compound parts first, then the word itself
            phrase_words.extend(compound_parts)
            phrase_words.append((w.id, w.text))
            
            # Collect all other children (dependents) except compounds (already handled)
            for child in sent.words:
                if child.head == w.id and child.deprel != 'compound':
                    collect_subtree(child)
        
        # Start from the given word
        collect_subtree(word)
        
        # Sort by word id to maintain order
        phrase_words.sort(key=lambda x: x[0])
        
        return ' '.join([w[1] for w in phrase_words])


class HybridDependencyParser:
    """
    Hybrid Dependency Parser: Stanza + LLM Refinement
    
    This class combines Stanza's accurate syntactic parsing with LLM's semantic understanding:
    1. First uses Stanza to get accurate dependency structure
    2. Then uses LLM to refine, adjust, and supplement the results based on semantic understanding
    """
    
    def __init__(self,
                 stanza_parser: StanzaDependencyParser,
                 llm_client=None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """
        Initialize the hybrid dependency parser
        
        Args:
            stanza_parser: Pre-initialized StanzaDependencyParser instance
            llm_client: Optional LLM client (if provided, will use it instead of OpenAI client)
            api_key: OpenAI API key (if None, will try to get from environment)
            base_url: Custom base URL for OpenAI API (optional)
            model: Model name to use (default: gpt-4o-mini)
            temperature: Temperature for LLM generation (default: 0.1)
            max_tokens: Maximum tokens for response (default: 2000)
        """
        self.stanza_parser = stanza_parser
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_client = llm_client
        
        # If llm_client is provided, use it; otherwise initialize OpenAI client
        if llm_client is None:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI is not available. Please install: pip install openai")
            
            # Get API key from parameter or environment
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            # Initialize OpenAI client
            client_kwargs = {'api_key': api_key}
            if base_url:
                client_kwargs['base_url'] = base_url
            
            self.client = OpenAI(**client_kwargs)
            self.use_openai_client = True
        else:
            self.client = None
            self.use_openai_client = False
        
        logger.info(f"Hybrid Dependency Parser initialized with model: {model}")
    
    def parse_question(self, question: str) -> ParsingResult:
        """
        Parse a question using hybrid approach: Stanza first, then LLM refinement
        
        Args:
            question: Natural language question string
            
        Returns:
            ParsingResult containing refined core entities and constraints
        """
        logger.debug(f"Parsing question with hybrid approach: {question[:100]}...")
        
        # Step 1: Use Stanza to get initial parsing
        logger.debug("Step 1: Running Stanza dependency parsing...")
        stanza_result = self.stanza_parser.parse_question(question)
        
        # Step 2: Format Stanza results for LLM
        stanza_summary = self._format_stanza_results(stanza_result)
        
        # Step 3: Use LLM to refine and adjust
        logger.debug("Step 2: Using LLM to refine and adjust results...")
        prompt = self._construct_refinement_prompt(question, stanza_summary)
        
        # Call LLM
        try:
            if self.use_openai_client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in linguistic dependency parsing. Your task is to refine and adjust dependency parsing results based on semantic understanding."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                llm_response = response.choices[0].message.content
            else:
                # Use provided LLM client
                llm_response = self.llm_client.call_with_messages(
                    messages=[
                        {"role": "system", "content": "You are an expert in linguistic dependency parsing. Your task is to refine and adjust dependency parsing results based on semantic understanding."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
            
            logger.debug(f"LLM response: {llm_response[:500]}...")
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            # Fallback to Stanza results if LLM fails
            logger.warning("Falling back to Stanza results only")
            return stanza_result
        
        # Step 4: Parse LLM response and merge with Stanza results
        refined_entities = self._parse_llm_refinement(llm_response, stanza_result)
        
        # Create refined result
        result = ParsingResult(
            question=question,
            sentences=stanza_result.sentences,
            dependency_relations=stanza_result.dependency_relations,
            constituency_trees=stanza_result.constituency_trees,
            core_entities=refined_entities,
            constraints=stanza_result.constraints  # Keep original constraints for now
        )
        
        return result
    
    def _format_stanza_results(self, result: ParsingResult) -> str:
        """
        Format Stanza parsing results into a readable summary for LLM
        
        Args:
            result: ParsingResult from Stanza
            
        Returns:
            Formatted string summary
        """
        lines = ["Stanza Dependency Parsing Results:", "=" * 60]
        
        if result.core_entities:
            lines.append("\nCore Entities:")
            for entity in result.core_entities:
                entity_name = entity.get('full_phrase', entity['word'])
                lines.append(f"  - {entity_name} ({entity['pos']}, {entity['deprel']})")
                if entity.get('constraints'):
                    lines.append("    Constraints:")
                    for constraint in entity['constraints']:
                        lines.append(f"      * {constraint['full_phrase']} ({constraint['deprel']})")
                lines.append("")
        
        if result.constraints:
            lines.append("\nStandalone Constraints:")
            for constraint in result.constraints:
                if not constraint.get('verb_constraints'):
                    lines.append(f"  - {constraint.get('full_phrase', constraint.get('word', ''))} (modifies: {constraint.get('head_text', '')}, {constraint.get('deprel', '')})")
        
        return "\n".join(lines)
    
    def _construct_refinement_prompt(self, question: str, stanza_summary: str) -> str:
        """
        Construct prompt for LLM to refine Stanza results
        
        Args:
            question: Original question
            stanza_summary: Formatted Stanza results
            
        Returns:
            Prompt string
        """
        # Use the same prompt as in the original dependency_parsing.py
        # (This is a simplified version - the full prompt is very long)
        prompt = f"""You are given a question and its dependency parsing results from Stanza (a syntactic parser). Your task is to review, refine, and adjust these results based on semantic understanding.

Question: "{question}"

Stanza Parsing Results:
{stanza_summary}

Please review the Stanza results and:
1. Identify any missing constraints for entities
2. Correct any incorrect constraint assignments
3. Add constraints that should be associated with entities based on semantic understanding
4. Remove or adjust constraints that don't make semantic sense
5. Ensure each entity has ALL its complete constraint phrases
6. Identify cases where a constraint should modify multiple parallel entities
7. Correct cases where constraints are not properly attached to the right entities

CRITICAL REQUIREMENTS:
- Each constraint MUST be a COMPLETE modifying phrase (not fragments)
- Extract entities layer by layer (hierarchical extraction)
- Each constraint must be a continuous text fragment from the original question
- If an entity has no constraints, use an empty array []

Output your refined analysis in JSON format:
{{
  "entities": [
    {{
      "entity": "entity_name",
      "constraints": [
        "complete modifying phrase 1",
        "complete modifying phrase 2"
      ]
    }}
  ]
}}

Now review and refine the Stanza results:"""
        
        return prompt
    
    def _parse_llm_refinement(self, response: str, stanza_result: ParsingResult) -> List[Dict[str, Any]]:
        """
        Parse LLM refinement response and convert to entity format
        
        Args:
            response: LLM response string
            stanza_result: Original Stanza result (for reference)
            
        Returns:
            List of refined entity dictionaries
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in LLM response")
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Convert to entity format
            refined_entities = []
            if 'entities' in data:
                for entity_data in data['entities']:
                    entity_name = entity_data.get('entity', '').strip()
                    constraints = entity_data.get('constraints', [])
                    
                    if entity_name:
                        entity_info = {
                            'word': entity_name,
                            'lemma': entity_name.lower(),
                            'pos': 'NOUN',
                            'deprel': '',
                            'head': 0,
                            'head_text': 'ROOT',
                            'full_phrase': entity_name,
                            'base_phrase': entity_name,
                            'constraints': []
                        }
                        
                        # Add constraints as separate items
                        if isinstance(constraints, str):
                            constraints = [constraints] if constraints.strip() else []
                        elif not isinstance(constraints, list):
                            constraints = []
                        
                        # Add each constraint separately
                        for constraint_str in constraints:
                            constraint_str = constraint_str.strip()
                            if constraint_str:
                                constraint_info = {
                                    'word': constraint_str,
                                    'deprel': 'constraint',
                                    'pos': '',
                                    'lemma': constraint_str.lower(),
                                    'full_phrase': constraint_str
                                }
                                entity_info['constraints'].append(constraint_info)
                        
                        refined_entities.append(entity_info)
            
            logger.debug(f"LLM refined {len(refined_entities)} entities")
            return refined_entities
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response was: {response[:500]}")
            # Fallback to Stanza results
            logger.warning("Falling back to Stanza entities")
            return stanza_result.core_entities or []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response was: {response[:500]}")
            return stanza_result.core_entities or []
















