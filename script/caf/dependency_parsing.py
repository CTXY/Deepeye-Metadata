#!/usr/bin/env python3
"""
Dependency Parsing using Stanford Stanza or Hybrid approach (Stanza + LLM refinement)

This script provides two methods to perform dependency parsing on natural language questions:
1. Stanza (default): Accurate syntactic parsing using Stanford NLP
2. Hybrid: Combines Stanza's syntactic accuracy with LLM's semantic understanding

It extracts:
- Core entities (nouns with their POS tags and dependency relations)
- Constraints for each entity (determiners, modifiers, prepositional phrases, etc.)

Usage:
    # Using Stanza (default):
    python script/caf/dependency_parsing.py --question "How much, in total, did client number 617 pay for all of the transactions in 1998?"
    python script/caf/dependency_parsing.py --question "How much, in total, did client number 617 pay for all of the transactions in 1998?" --output output.json
    
    # Using Hybrid approach (Stanza + LLM refinement) - RECOMMENDED:
    python script/caf/dependency_parsing.py --question "How much, in total, did client number 617 pay for all of the transactions in 1998?" --use-hybrid --api-key "sk-6F6WwG6R2FW7Mvq12889E46c67B04d549bE13f7465F212Ba" --base-url 'https://vip.yi-zhan.top/v1'
    python script/caf/dependency_parsing.py --input questions.txt --output results.jsonl --use-hybrid --api-key "sk-..."
    
    # With custom model:
    python script/caf/dependency_parsing.py --question "..." --use-hybrid --model gpt-4 --temperature 0.1 --api-key "sk-..."
"""

import argparse
import json
import logging
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import stanza
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
class ConstituencyNode:
    """Represents a node in the constituency parse tree"""
    label: str
    text: str
    children: List['ConstituencyNode']


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
        logger.info(f"Parsing question: {question[:100]}...")
        
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
    
    def print_dependencies(self, result: ParsingResult):
        """
        Print dependency relations in a readable format
        
        Args:
            result: ParsingResult object
        """
        print(f"\n{'='*80}")
        print(f"Question: {result.question}")
        print(f"{'='*80}\n")
        
        for i, (sent, deps) in enumerate(zip(result.sentences, result.dependency_relations)):
            print(f"Sentence {i+1}: {sent['text']}")
            print(f"\nDependency Relations:")
            print(f"{'Word':<20} {'Head':<20} {'Relation':<15} {'POS':<10} {'Lemma':<15}")
            print("-" * 80)
            
            for idx, dep in enumerate(deps):
                head_text = sent['dependencies'][idx]['head_text']
                print(f"{dep.word:<20} {head_text:<20} {dep.deprel:<15} {dep.upos:<10} {dep.lemma:<15}")
            
            print()
    
    def print_constituency_tree(self, result: ParsingResult):
        """
        Print constituency parse trees
        
        Args:
            result: ParsingResult object
        """
        print(f"\n{'='*80}")
        print(f"Constituency Parse Trees")
        print(f"{'='*80}\n")
        
        for i, tree_str in enumerate(result.constituency_trees):
            if tree_str:
                print(f"Sentence {i+1} Tree:")
                print(tree_str)
                print()
    
    def print_entities_and_constraints(self, result: ParsingResult):
        """
        Print extracted core entities and constraints
        
        Args:
            result: ParsingResult object
        """
        print(f"\n{'='*80}")
        print(f"Core Entities and Constraints")
        print(f"{'='*80}\n")
        
        if result.core_entities:
            print("Core Entities:")
            for entity in result.core_entities:
                # Show full phrase if available (includes compound nouns and numeric modifiers)
                entity_name = entity.get('full_phrase', entity['word'])
                print(f"  - {entity_name} ({entity['pos']}, {entity['deprel']})")
                if entity.get('constraints'):
                    print("    Constraints:")
                    for constraint in entity['constraints']:
                        print(f"      * {constraint['full_phrase']} ({constraint['deprel']})")
                print()
        
        # Separate verb phrases from other constraints
        verb_constraints = []
        other_constraints = []
        
        if result.constraints:
            for constraint in result.constraints:
                if constraint.get('verb_constraints') or constraint.get('deprel') == 'verb_phrase':
                    verb_constraints.append(constraint)
                else:
                    other_constraints.append(constraint)
        
        if verb_constraints:
            print("Verb Phrases and Their Constraints:")
            for constraint in verb_constraints:
                verb_text = constraint.get('word', constraint.get('full_phrase', ''))
                print(f"  - Verb: {verb_text}")
                if constraint.get('verb_constraints'):
                    for vc in constraint['verb_constraints']:
                        print(f"    * {vc['full_phrase']} ({vc['type']})")
                print()
        
        if other_constraints:
            print("Other Constraints:")
            for constraint in other_constraints:
                print(f"  - {constraint['full_phrase']} (modifies: {constraint['head_text']}, {constraint['deprel']})")
            print()
    
    def to_dict(self, result: ParsingResult) -> Dict[str, Any]:
        """
        Convert ParsingResult to dictionary for JSON serialization
        
        Args:
            result: ParsingResult object
            
        Returns:
            Dictionary representation
        """
        return {
            'question': result.question,
            'sentences': result.sentences,
            'dependency_relations': [
                [asdict(dep) for dep in sent_deps]
                for sent_deps in result.dependency_relations
            ],
            'constituency_trees': result.constituency_trees,
            'core_entities': result.core_entities,
            'constraints': result.constraints
        }


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
        logger.info(f"Parsing question with hybrid approach: {question[:100]}...")
        
        # Step 1: Use Stanza to get initial parsing
        logger.info("Step 1: Running Stanza dependency parsing...")
        stanza_result = self.stanza_parser.parse_question(question)
        
        # Step 2: Format Stanza results for LLM
        stanza_summary = self._format_stanza_results(stanza_result)
        
        # Step 3: Use LLM to refine and adjust
        logger.info("Step 2: Using LLM to refine and adjust results...")
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
6. Identify cases where a constraint should modify multiple parallel entities (e.g., "ID and district" both modified by the same constraint)
7. Correct cases where constraints are not properly attached to the right entities

================================================================================
CRITICAL REQUIREMENTS FOR CONSTRAINTS:
================================================================================

1. **Complete Modifying Phrases (完整修饰短语)**:
   - Each constraint MUST be a COMPLETE modifying phrase that contains the FULL semantic relationship to the entity
   - Do NOT split constraints into smaller fragments
   - Each constraint should be a continuous text fragment from the original question
   - Example CORRECT: For "zip code", use "of all the charter schools in Fresno County Office of Education" (complete phrase)
   - Example WRONG: "of all the charter schools" and "in Fresno County Office of Education" (split fragments - DO NOT DO THIS)

2. **Hierarchical Extraction (逐层提取)**:
   - Extract entities and constraints layer by layer, going deeper into nested structures
   - Layer 1: Extract top-level entities with their COMPLETE modifying phrases
   - Layer 2: Extract entities that appear within the constraints of Layer 1, with their own COMPLETE modifying phrases
   - Continue this process for deeper layers
   - Each entity at each layer should have its own complete constraint phrases
   - This allows gradual decomposition: each layer extracts entities with their complete modifying phrases

3. **Constraint Format**:
   - Each constraint must be a continuous text fragment from the original question
   - Each constraint should semantically modify the entity
   - Multiple constraints for the same entity should be provided as separate items in the array
   - If an entity has no constraints, use an empty array []

Example 1 - Hierarchical Extraction:
Question: Please list the zip code of all the charter schools in Fresno County Office of Education.

Expected Output:
{{
  "entities": [
    {{
      "entity": "zip code",
      "constraints": [
        "of all the charter schools in Fresno County Office of Education"
      ]
    }},
    {{
      "entity": "charter schools",
      "constraints": [
        "in Fresno County Office of Education"
      ]
    }},
    {{
      "entity": "County Office",
      "constraints": [
        "of Education",
        "in Fresno County"
      ]
    }},
    {{
      "entity": "Education",
      "constraints": [
        "of Education"
      ]
    }}
  ]
}}

Key Points (关键点):
✓ "zip code" has constraint "of all the charter schools in Fresno County Office of Education" 
  → This is a COMPLETE modifying phrase (NOT split into "of all the charter schools" and "in Fresno County Office of Education")
✓ "charter schools" appears within the constraint of "zip code", so we extract it as a separate entity with its own COMPLETE constraint "in Fresno County Office of Education"
✓ "County Office" appears within the constraint of "charter schools", so we extract it with its COMPLETE constraints "of Education" and "in Fresno County"
✓ "Education" appears within the constraint of "County Office", so we extract it with its COMPLETE constraint "of Education"
✓ This demonstrates hierarchical extraction (逐层提取): each layer extracts entities with their complete modifying phrases, allowing gradual decomposition (逐步向下拆分)

Example 2 - Parallel Entities with Shared Constraints:
Question: List all ID and district for clients that can only have the right to issue permanent orders or apply for loans.

Expected Output:
{{
  "entities": [
    {{
      "entity": "ID",
      "constraints": [
        "all",
        "for clients that can only have the right to issue permanent orders or apply for loans"
      ]
    }},
    {{
      "entity": "district",
      "constraints": [
        "all",
        "for clients that can only have the right to issue permanent orders or apply for loans"
      ]
    }},
    {{
      "entity": "clients",
      "constraints": [
        "that can only have the right to issue permanent orders or apply for loans"
      ]
    }},
    {{
      "entity": "orders",
      "constraints": [
        "permanent"
      ]
    }},
    {{
      "entity": "loans",
      "constraints": []
    }}
  ]
}}

Key Points (关键点):
✓ "ID" and "district" are parallel entities that share the same constraints: "all" and "for clients that can only have the right to issue permanent orders or apply for loans"
  → Each parallel entity should have the shared constraints listed separately (both entities get the same COMPLETE constraints)
✓ "clients" appears within the constraint of "ID" and "district", so we extract it as a separate entity with its own COMPLETE constraint "that can only have the right to issue permanent orders or apply for loans"
✓ "orders" has constraint "permanent" (complete modifying phrase)
✓ "loans" has no constraints, so use empty array []
✓ This demonstrates how parallel entities share constraints and how nested entities are extracted with their own complete constraints

Example 3 - Verb Phrase Constraints:
Question: How much, in total, did client number 617 pay for all of the transactions in 1998?

Expected Output:
{{
  "entities": [
    {{
      "entity": "client",
      "constraints": [
        "number 617"
      ]
    }},
    {{
      "entity": "transactions",
      "constraints": [
        "client number 617 pay for",
        "all of the transactions in 1998",
        "in 1998"
      ]
    }}
  ]
}}

Output your refined analysis in the following JSON format:
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

Important guidelines:
- **Keep complete phrases**: Each constraint must be a complete modifying phrase from the original question, NOT fragments
- **Hierarchical extraction**: Extract entities layer by layer - entities that appear in constraints should be extracted in the next layer with their own constraints
- **Multiple constraints per entity**: An entity can have multiple constraints, each as a separate complete phrase
- **Shared constraints**: If multiple parallel entities share the same constraint, include it for each entity
- **Empty constraints**: If an entity has no constraints, use an empty array []
- **Text continuity**: Constraints must be continuous text fragments from the original question
- **Semantic correctness**: Think about what the question is REALLY asking about and what constraints each entity should have semantically
- **Verb phrase constraints**: If a verb phrase semantically modifies an entity, include it as a constraint (e.g., "client number 617 pay for" modifies "transactions")
- **Do NOT split**: Do NOT split a single complete modifying phrase into multiple smaller constraints

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
                        
                        # Add constraints as separate items (support both list and string for backward compatibility)
                        if isinstance(constraints, str):
                            # Backward compatibility: if constraints is a string, treat as single constraint
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
            
            logger.info(f"LLM refined {len(refined_entities)} entities")
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
    
    def print_entities_and_constraints(self, result: ParsingResult):
        """Print extracted entities and their constraints in Stanza format"""
        print(f"\n{'='*80}")
        print(f"Question: {result.question}")
        print(f"{'='*80}\n")
        
        if result.core_entities:
            print("Core Entities:")
            for entity in result.core_entities:
                entity_name = entity.get('full_phrase', entity['word'])
                pos = entity.get('pos', 'NOUN')
                deprel = entity.get('deprel', '')
                print(f"  - {entity_name} ({pos}, {deprel})")
                
                # Print constraints in Stanza format (each constraint on a separate line with *)
                if entity.get('constraints'):
                    print("    Constraints:")
                    for constraint in entity['constraints']:
                        constraint_phrase = constraint.get('full_phrase', constraint.get('word', ''))
                        if constraint_phrase:
                            print(f"      * {constraint_phrase}")
                print()
        else:
            print("No entities found.\n")
    
    def to_dict(self, result: ParsingResult) -> Dict[str, Any]:
        """Convert ParsingResult to dictionary for JSON serialization"""
        return {
            'question': result.question,
            'sentences': result.sentences,
            'dependency_relations': [
                [asdict(dep) for dep in sent_deps]
                for sent_deps in result.dependency_relations
            ],
            'constituency_trees': result.constituency_trees,
            'core_entities': result.core_entities,
            'constraints': result.constraints
        }


def main():
    """Main function to run dependency parsing"""
    parser = argparse.ArgumentParser(
        description="Perform dependency and constituency parsing on questions using Stanford Stanza or Hybrid approach (Stanza + LLM refinement)"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Single question to parse"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file containing questions (one per line or JSON format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (JSON or JSONL format)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration"
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed parsing results to console"
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid approach: Stanza first, then LLM refinement (recommended)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Custom OpenAI API base URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM generation (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.question and not args.input:
        parser.error("Either --question or --input must be provided")
    
    # Initialize parser
    if args.use_hybrid:
        # Hybrid approach: Stanza + LLM
        stanza_parser = StanzaDependencyParser(lang=args.lang, use_gpu=args.use_gpu)
        parser_obj = HybridDependencyParser(
            stanza_parser=stanza_parser,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature
        )
    else:
        parser_obj = StanzaDependencyParser(lang=args.lang, use_gpu=args.use_gpu)
    
    # Process questions
    results = []
    
    if args.question:
        # Single question
        result = parser_obj.parse_question(args.question)
        results.append(result)
        
        # For single question, print detailed results if:
        # 1. No output file specified (default behavior)
        # 2. Verbose flag is set
        should_print = not args.output or args.verbose
        if should_print:
            if args.use_hybrid:
                parser_obj.print_entities_and_constraints(result)
            else:
                parser_obj.print_dependencies(result)
                parser_obj.print_constituency_tree(result)
                parser_obj.print_entities_and_constraints(result)
    
    elif args.input:
        # Batch processing from file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return
        
        questions = []
        if args.input.endswith('.json'):
            # JSON format
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions = [item.get('question', item) if isinstance(item, dict) else str(item) for item in data]
                elif isinstance(data, dict):
                    questions = [data.get('question', str(data))]
        else:
            # Plain text, one question per line
            with open(args.input, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(questions)} questions from {args.input}")
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            try:
                result = parser_obj.parse_question(question)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                continue
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == "jsonl":
            # JSONL format (one result per line)
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(parser_obj.to_dict(result), f, ensure_ascii=False)
                    f.write('\n')
        else:
            # JSON format (single array)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [parser_obj.to_dict(result) for result in results],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        
        logger.info(f"Results saved to {args.output}")
        # If output file is specified, still print details if verbose is enabled
        if args.verbose:
            for result in results:
                if args.use_hybrid:
                    parser_obj.print_entities_and_constraints(result)
                else:
                    parser_obj.print_dependencies(result)
                    parser_obj.print_constituency_tree(result)
                    parser_obj.print_entities_and_constraints(result)
    elif args.input and not args.verbose:
        # For batch processing without verbose, print summary
        print(f"\nProcessed {len(results)} question(s)")
        for i, result in enumerate(results):
            print(f"\nQuestion {i+1}: {result.question}")
            print(f"  Sentences: {len(result.sentences)}")
            print(f"  Core entities: {len(result.core_entities) if result.core_entities else 0}")
            print(f"  Constraints: {len(result.constraints) if result.constraints else 0}")
    elif args.input and args.verbose:
        # For batch processing with verbose, print details for all
        for result in results:
            if args.use_hybrid:
                parser_obj.print_entities_and_constraints(result)
            else:
                parser_obj.print_dependencies(result)
                parser_obj.print_constituency_tree(result)
                parser_obj.print_entities_and_constraints(result)


if __name__ == "__main__":
    main()

