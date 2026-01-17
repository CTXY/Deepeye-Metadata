# LLM Analyzer - Generate semantic metadata using LLM analysis
# Lower priority source, only used when no existing information

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import unified LLM client system
from caf.llm.client import LLMConfig, create_llm_client, BaseLLMClient
from caf.prompt import PromptFactory

logger = logging.getLogger(__name__)

class LLMProviderError(Exception):
    """Error in LLM provider communication"""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"LLM Provider ({provider}): {message}")

class LLMAnalyzer:
    """
    LLM Analyzer - Generate semantic metadata using Large Language Models
    
    This analyzer uses LLMs to generate:
    - Database descriptions and domain identification
    - Table descriptions based on schema
    - Column descriptions, formats, and encoding mappings
    - Pattern descriptions in natural language
    - Relationship business meanings
    
    Priority: Lowest (llm_analysis) - only used when no existing information
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.provider = self.config.get('provider', 'openai')
        self.model_name = self.config.get('model_name', 'gpt-4o-mini')
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 4000) # Increased for batch analysis
        
        self._init_llm_client()
        logger.debug(f"LLMAnalyzer initialized with {self.provider} model: {self.model_name}")
    
    def _init_llm_client(self):
        """Initialize LLM client using unified client system"""
        try:
            # Create LLM config from the old config format
            llm_config = LLMConfig(
                provider=self.provider,
                model_name=self.model_name,
                api_key=self.config.get('api_key'),
                base_url=self.config.get('base_url'),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.config.get('timeout', 60)
            )
            
            # Create the client using the unified system
            self.llm_client: BaseLLMClient = create_llm_client(llm_config)
            logger.debug(f"Initialized unified LLM client for {self.provider} with model {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    def analyze_database(self, context: Dict[str, Any], needs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform LLM analysis on the database based on a generation plan.
        Uses batching to improve efficiency.
        """
        results = {'database': {}, 'tables': {}, 'columns': {}}
        
        # 1. Analyze Database level 
        if needs.get('database'):
            try:
                db_properties = self._analyze_database_properties(context)
                results['database'].update({k: v for k, v in db_properties.items() if k in needs['database']})
            except Exception as e:
                logger.warning(f"LLM failed to generate database properties: {e}")

        # 2. Analyze Table level 
        if needs.get('tables'):
            
            tables_needing_semantics = self._extract_table_semantic_needs(needs['tables'])
     
            print('------------Tables Needing Semantics--------------------')
            print(tables_needing_semantics)
            if tables_needing_semantics:
                logger.info(f"Analyzing deep semantics for {len(tables_needing_semantics)} tables")
                try:
                    for table_name, missing_fields in tables_needing_semantics.items():
                        semantic_data = self._analyze_single_table_semantics(table_name, context)
                        if semantic_data:

                            filtered_data = {}
                            
                            for field in missing_fields:
                                if field in semantic_data:
                                    filtered_data[field] = semantic_data[field]
                            
                            results['tables'].setdefault(table_name, {}).update(filtered_data)
                            
                except Exception as e:
                    logger.warning(f"LLM failed during deep table semantics analysis: {e}")

        # 3. Analyze Column level 
        if needs.get('columns'):
            for table_name, columns_dict in needs['columns'].items():
                if not columns_dict: 
                    continue
                # Check if any column needs description or pattern_description
                columns_needing_analysis = {
                    col: fields for col, fields in columns_dict.items()
                    if 'description' in fields or 'pattern_description' in fields
                }
                if not columns_needing_analysis:
                    continue
                try:
                    column_analyses = self._batch_analyze_columns_for_table(table_name, columns_needing_analysis, context)
                    if column_analyses:
                        results['columns'].setdefault(table_name, {}).update(column_analyses)
                        print('=============== column_analyses ===============')
                        print(results['columns'])
                        print('=============== column_analyses ===============')
                        # Update context with the newly generated column metadata
                        for col_name, col_data in column_analyses.items():
                            if table_name not in context['columns']:
                                context['columns'][table_name] = {}
                            if col_name not in context['columns'][table_name]:
                                context['columns'][table_name][col_name] = {}
                            context['columns'][table_name][col_name].update(col_data)
                except Exception as e:
                    logger.warning(f"LLM failed during batch column analysis for table '{table_name}': {e}")

        # 4. Generate natural language descriptions (short/long) for columns
        #    This happens AFTER all other column metadata has been generated
        if needs.get('columns'):
            for table_name, columns_dict in needs['columns'].items():
                if not columns_dict: continue
                # Check if any column needs short_description or long_description
                columns_needing_nl_desc = {
                    col: fields for col, fields in columns_dict.items()
                    if 'short_description' in fields or 'long_description' in fields
                }
                if columns_needing_nl_desc:
                    try:
                        nl_descriptions = self._batch_generate_natural_descriptions(
                            table_name, columns_needing_nl_desc, context
                        )
                        if nl_descriptions:
                            results['columns'].setdefault(table_name, {}).update(nl_descriptions)
                    except Exception as e:
                        logger.warning(f"LLM failed to generate natural descriptions for table '{table_name}': {e}")
        
        # 5. Generate business meaning for relationships (after table descriptions are ready)
        if needs.get('relationships'):
            try:
                # Extract relationships that need business_meaning
                # needs['relationships'] is a list of relationship dicts with 'fields_needed'
                if isinstance(needs['relationships'], list):
                    # Use the list directly - it already contains only relationships that need generation
                    relationships_needing_meaning = needs['relationships']
                    logger.info(f"Found {len(relationships_needing_meaning)} relationships needing field generation")
                elif needs['relationships'] is True:
                    # Backward compatibility: Extract all relationships from context that have missing/empty business_meaning
                    relationships_needing_meaning = []
                    context_relationships = context.get('relationships', [])
                    for rel in context_relationships:
                        business_meaning = rel.get('business_meaning')
                        # Check if business_meaning is missing, None, NaN, or empty string
                        if (business_meaning is None or 
                            pd.isna(business_meaning) or 
                            (isinstance(business_meaning, str) and business_meaning.strip() == '')):
                            relationships_needing_meaning.append(rel)
                    logger.info(f"Found {len(relationships_needing_meaning)} relationships needing business_meaning out of {len(context_relationships)} total relationships")
                else:
                    relationships_needing_meaning = []
                
                if relationships_needing_meaning:
                    relationship_meanings = self._analyze_relationship_semantics(context, relationships_needing_meaning)
                    if relationship_meanings:
                        results['relationships'] = relationship_meanings
                else:
                    logger.info("No relationships need business_meaning generation")
            except Exception as e:
                logger.warning(f"LLM failed to generate relationship business meanings: {e}")
        
        logger.info("LLM analysis based on generation plan completed.")
        return results


    def _parse_json_from_response(self, response_text: str) -> Optional[Any]:
        """Extracts and parses JSON from a string, handling markdown code blocks."""
        try:
            text = response_text.strip()
            
            # Try to find JSON in markdown code blocks first
            import re
            json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_block_pattern, text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Find the first '{' or '[' to start parsing from
            json_start = -1
            for i, char in enumerate(text):
                if char in ['{', '[']:
                    json_start = i
                    break
            
            if json_start == -1: return None

            # Find the matching closing bracket by counting brackets
            bracket_stack = []
            json_end = -1
            opening = text[json_start]
            closing = '}' if opening == '{' else ']'
            
            for i in range(json_start, len(text)):
                char = text[i]
                if char == opening:
                    bracket_stack.append(char)
                elif char == closing:
                    bracket_stack.pop()
                    if len(bracket_stack) == 0:
                        json_end = i
                        break

            if json_end == -1: 
                # Fallback to rfind if bracket matching fails
                json_end = text.rfind(closing)
                if json_end == -1: return None
            
            clean_text = text[json_start : json_end + 1]
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}. Response body: {response_text[:500]}...")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error parsing JSON: {e}. Response body: {response_text[:500]}...")
            return None

    def _create_sample_data_markdown(
        self, 
        sample_data: Optional[Dict[str, List]], 
        max_cols: int = 10, 
        max_rows: int = 5,
        columns: Optional[List[str]] = None,
        cell_max_length: int = 80
    ) -> str:
        """
        Creates a markdown table from sample data for use in prompts.
        
        Args:
            sample_data: Dictionary mapping column names to lists of values
            max_cols: Maximum number of columns to include (when columns=None)
            max_rows: Maximum number of rows to include
            columns: Specific columns to include (if None, uses first max_cols columns)
            cell_max_length: Maximum length for each cell value
            
        Returns:
            Markdown-formatted table string
        """
        if not sample_data or not isinstance(sample_data, dict):
            return "No sample data available"

        # Use specified columns or first max_cols columns
        if columns:
            available_columns = [col for col in columns if col in sample_data and sample_data[col]]
            if not available_columns:
                return "No sample data available for the specified columns"
            sampled_columns = available_columns
        else:
            sampled_columns = list(sample_data.keys())[:max_cols]
            if not sampled_columns:
                return "No sample data available"

        def _format_cell(v: Any) -> str:
            s = str(v).replace('\n', ' ').replace('|', '\\|')
            return s[:cell_max_length]

        header = "| " + " | ".join(sampled_columns) + " |"
        separator = "| " + " | ".join(["---"] * len(sampled_columns)) + " |"
        
        # Get the maximum number of rows across all columns
        max_available_rows = max(len(sample_data.get(c, [])) for c in sampled_columns)
        num_rows = min(max_rows, max_available_rows)

        rows = []
        for i in range(num_rows):
            row_cells = []
            for col in sampled_columns:
                col_data = sample_data.get(col, [])
                if i < len(col_data):
                    row_cells.append(_format_cell(col_data[i]))
                else:
                    row_cells.append("")
            rows.append("| " + " | ".join(row_cells) + " |")
        
        # Add notes for omitted rows/columns
        notes = []
        if max_available_rows > num_rows:
            notes.append(f"*(...and {max_available_rows - num_rows} more rows)*")
        if columns is None and len(sample_data) > len(sampled_columns):
            notes.append(f"*(...and {len(sample_data) - len(sampled_columns)} more columns)*")

        result = "\n".join([header, separator] + rows)
        if notes:
            result += "\n" + "\n".join(notes)
        
        return result
    
    def _check_column_has_sufficient_data(self, sample_data: Optional[Dict[str, List]], column_name: str, min_non_null_ratio: float = 0.1) -> bool:
        """Check if a column has sufficient non-null data in sample_data."""
        if not sample_data or not isinstance(sample_data, dict):
            return False
        
        if column_name not in sample_data:
            return False
        
        col_data = sample_data[column_name]
        if not col_data or len(col_data) == 0:
            return False
        
        # Count non-null values
        non_null_count = sum(1 for v in col_data if v is not None and str(v).strip() != '' and str(v).lower() != 'null')
        total_count = len(col_data)
        
        if total_count == 0:
            return False
        
        non_null_ratio = non_null_count / total_count
        return non_null_ratio >= min_non_null_ratio
    
        
    def _clean_regex_pattern(self, regex: str) -> str:
        """
        Clean and simplify regex pattern for better readability.
        
        Removes redundant word boundaries (\b) when using ^ and $ anchors,
        as they are unnecessary for full string matching with fullmatch().
        
        Examples:
            ^\b[0-9]{5}\b$ -> ^[0-9]{5}$
            ^\b[0-9]{4}-[0-9]{4}\b$ -> ^[0-9]{4}-[0-9]{4}$
        
        Args:
            regex: Raw regex pattern from LLM
            
        Returns:
            Cleaned regex pattern
        """
        if not regex:
            return regex
        
        fixed_regex = regex.replace('\x08', r'\b')
        
        if fixed_regex.startswith(r'^\b') and fixed_regex.endswith(r'\b$'):
            middle_part = fixed_regex[3:-3]
            fixed_regex = f'^{middle_part}$'
        
        return fixed_regex
    
    def _validate_regex_pattern(self, regex: str, values: List[Any]) -> bool:
        """Validates if all non-null values in a list match the given regex pattern."""
        if not regex or not values: return False
        try:
            # Clean regex pattern (remove redundant \b, fix escape sequences)
            cleaned_regex = self._clean_regex_pattern(regex)
            pattern = re.compile(cleaned_regex)
            return all(pattern.fullmatch(str(v)) for v in values if v is not None)
        except re.error:
            logger.warning(f"Invalid regex pattern provided by LLM: '{regex}'")
            return False
    
    def _validate_pattern_description(self, pattern_description: str, template: Optional[str], 
                                     regex: Optional[str], values: List[Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates pattern_description by checking:
        1. If template is provided, verify it matches the actual values
        2. If regex is provided, verify it matches all values
        3. Check if pattern_description accurately describes the pattern
        
        Returns: (is_valid, error_message)
        """
        if not pattern_description:
            return False, "Missing pattern_description"
        
        # If no values available, we can only validate that description exists
        if not values:
            # Allow validation to pass if we have description but no values to check against
            # This is acceptable when only description is provided
            return True, None
        
        # Validate template if provided
        if template:
            try:
                sample_values = [str(v) for v in values[:20] if v is not None]
                if not sample_values:
                    return False, "No sample values to validate template against"
                
                # Extract fixed parts from template (parts outside {placeholders})
                import re as re_module
                # Split template by placeholders
                fixed_parts = re_module.split(r'\{[^}]+\}', template)
                fixed_parts = [p for p in fixed_parts if p]  # Remove empty strings
                
                # Check if template has placeholders
                placeholders = template.count('{')
                
                # If template has no placeholders (pure pattern like 'XXXXXXXXXXXXXX'),
                # validate by checking length consistency instead of fixed parts
                if placeholders == 0:
                    # For pure pattern templates, check if all values have the same length as template
                    template_length = len(template)
                    lengths = [len(v) for v in sample_values]
                    matching_lengths = sum(1 for l in lengths if l == template_length)
                    match_ratio = matching_lengths / len(sample_values) if sample_values else 0
                    if match_ratio < 0.7:  # Less than 70% match
                        return False, f"Template '{template}' length doesn't match most values (only {match_ratio:.1%} match, expected length {template_length})"
                else:
                    # Template has placeholders, check fixed parts
                    if fixed_parts:
                        # Check if at least some fixed parts appear in most values
                        matching_count = 0
                        for val in sample_values:
                            if all(part in val for part in fixed_parts):
                                matching_count += 1
                        
                        match_ratio = matching_count / len(sample_values) if sample_values else 0
                        if match_ratio < 0.7:  # Less than 70% match
                            return False, f"Template '{template}' fixed parts don't match most values (only {match_ratio:.1%} match)"
                    
                    # Check length consistency if template has placeholders
                    lengths = [len(v) for v in sample_values]
                    length_variance = max(lengths) - min(lengths) if lengths else 0
                    # Allow some variance but not too much
                    avg_length = sum(lengths) / len(lengths) if lengths else 0
                    if avg_length > 0 and length_variance / avg_length > 0.5:  # More than 50% variance
                        logger.debug(f"Template '{template}' has high length variance: {length_variance}")
                
            except Exception as e:
                logger.debug(f"Template validation error: {e}")
        
        # Validate regex if provided
        if regex:
            if not self._validate_regex_pattern(regex, values):
                return False, f"Regex '{regex}' doesn't match all values"
        
        # Basic sanity check: pattern_description should not be too generic
        generic_patterns = ["text", "string", "alphanumeric", "mixed", "various", "different"]
        desc_lower = pattern_description.lower()
        if any(gp in desc_lower for gp in generic_patterns) and len(pattern_description.split()) < 5:
            # Too generic, might not be useful
            logger.debug(f"Pattern description seems too generic: {pattern_description}")
        
        return True, None

    def _extract_pattern_data(self, result_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract pattern data from LLM response fields (handles both nested and flat formats)."""
        if 'pattern' in result_fields:
            return result_fields['pattern']
        elif 'template' in result_fields or 'regex' in result_fields or 'fmt_desc' in result_fields:
            return {
                'template': result_fields.get('template'),
                'regex': result_fields.get('regex'),
                'fmt_desc': result_fields.get('fmt_desc')
            }
        return None

    def _process_description_field(self, result_fields: Dict[str, Any], cleaned_analysis: Dict[str, Any]) -> None:
        """Process and extract description field from LLM response."""
        if 'description' in result_fields:
            desc_value = result_fields['description']
            if desc_value and str(desc_value).lower() not in ["null", "none", ""]:
                cleaned_analysis['description'] = desc_value

    def _process_pattern_field_simple(
        self, 
        pattern_data: Dict[str, Any], 
        cleaned_analysis: Dict[str, Any],
        table_name: str,
        column_name: str
    ) -> None:
        """Process pattern field in simple mode (for top_k_values analysis, no validation)."""
        desc = pattern_data.get('fmt_desc') or pattern_data.get('description')
        if desc and str(desc).strip():
            cleaned_analysis['pattern_description'] = desc
            logger.info(f"✅ Pattern description saved for {table_name}.{column_name}: {desc[:80]}...")

    def _process_pattern_field_with_validation(
        self,
        pattern_data: Dict[str, Any],
        cleaned_analysis: Dict[str, Any],
        table_name: str,
        column_name: str,
        distinct_values: List[Any]
    ) -> None:
        """Process pattern field with full validation (for batch analysis with sample data)."""
        regex = pattern_data.get('regex')
        desc = pattern_data.get('fmt_desc') or pattern_data.get('description')
        template = pattern_data.get('template')
        
        # Normalize None values (handle string "null" or "None")
        if regex and str(regex).lower() in ['null', 'none', '']:
            regex = None
        if template and str(template).lower() in ['null', 'none', '']:
            template = None
        
        # Clean regex pattern (remove redundant \b, fix escape sequences)
        if regex:
            regex = self._clean_regex_pattern(regex)
        
        if not desc:
            logger.warning(f"Pattern for {table_name}.{column_name} missing description, skipping")
            return
        
        # Case 1: Has description with regex/template (full pattern)
        if regex or template:
            # Validate pattern with template and regex if both provided
            if regex and template:
                is_valid, error_msg = self._validate_pattern_description(
                    desc, template, regex, distinct_values
                )
                
                if is_valid:
                    # Build pattern_description with regex/template information
                    pattern_desc_parts = [desc]
                    if regex:
                        pattern_desc_parts.append(f"Regex: {regex}")
                    if template:
                        pattern_desc_parts.append(f"Template: {template}")
                    cleaned_analysis['pattern_description'] = " | ".join(pattern_desc_parts)
                    
                    validation_details = []
                    if template:
                        validation_details.append(f"template='{template}'")
                    if regex:
                        validation_details.append(f"regex='{regex}'")
                    logger.info(
                        f"✅ Pattern validated for {table_name}.{column_name}: "
                        f"{', '.join(validation_details)}. Description: {desc[:80]}..."
                    )
                else:
                    logger.warning(
                        f"⚠️ Pattern validation failed for {table_name}.{column_name}: {error_msg}\n"
                        f"   Template: {template}\n   Regex: {regex}\n   Description: {desc[:100]}...\n"
                        f"   Sample values: {[str(v) for v in distinct_values[:5]] if distinct_values else 'none'}\n"
                        f"   Saving description only (no regex/template)"
                    )
                    if desc and str(desc).strip():
                        cleaned_analysis['pattern_description'] = desc
            elif regex:
                # Only regex provided, validate it
                if distinct_values and self._validate_regex_pattern(regex, distinct_values):
                    # Build pattern_description with regex information
                    cleaned_analysis['pattern_description'] = f"{desc} | Regex: {regex}"
                    logger.info(
                        f"✅ Pattern validated for {table_name}.{column_name}: "
                        f"regex='{regex}'. Description: {desc[:80]}..."
                    )
                else:
                    logger.warning(f"⚠️ Regex validation failed for {table_name}.{column_name}, saving description only")
                    if desc and str(desc).strip():
                        cleaned_analysis['pattern_description'] = desc
            else:
                # Only template provided, save description with template
                if desc and str(desc).strip():
                    cleaned_analysis['pattern_description'] = f"{desc} | Template: {template}"
                    logger.info(
                        f"✅ Pattern saved for {table_name}.{column_name}: "
                        f"template='{template}' (no regex). Description: {desc[:80]}..."
                    )
        else:
            # Case 2: Only description provided (no unified pattern)
            if desc and str(desc).strip():
                cleaned_analysis['pattern_description'] = desc
                logger.info(
                    f"✅ Pattern description saved for {table_name}.{column_name} "
                    f"(no unified pattern, description only): {desc[:80]}..."
                )
            else:
                logger.warning(f"Pattern description too short for {table_name}.{column_name}: {desc}")

    def _build_column_analysis_schema(self, fields_needed: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """Build instructions and output schema for column analysis based on needed fields."""
        instructions = []
        output_schema = {}
        
        if 'description' in fields_needed:
            instructions.append("- Analyze the semantic meaning of the column by combining the table context, sample data, and column values. Provide a concise, clear, one-sentence description that explains what the data represents, its purpose within the table context, and its business meaning. The description must be brief and insightful, capturing the essence in a single sentence.")
            output_schema["description"] = "String (One concise sentence summarizing the column's semantic meaning)"
        
        if 'pattern_description' in fields_needed:
            instructions.append("- Analyze the syntactic structure (regex/format). Ignore nulls.")
            output_schema["pattern"] = {
                "template": "String or null",
                "regex": "String or null",
                "fmt_desc": "String (Format description)"
            }
        
        return instructions, output_schema

    # ===================================================================================
    # Core Analysis Functions (Database, Table, Column, Relationship)
    # ===================================================================================

    def _analyze_database_properties(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generates database description and domain in a single LLM call."""
        database_id = context.get('database_id', 'unknown')
        tables = context.get('tables', {})
        
        table_info = []
        for name, data in list(tables.items())[:20]: # Limit to first 20 tables for prompt brevity
            cols = data.get('columns', [])
            col_preview = ', '.join(cols[:10]) + ('...' if len(cols) > 10 else '') # Limit to first 10 columns for prompt brevity
            table_info.append(f"- {name} ({len(cols)} columns): {col_preview}")

        prompt = PromptFactory.format_database_properties_prompt(database_id, table_info)

        response_text = self._call_llm(prompt)
        analysis = self._parse_json_from_response(response_text)
        
        if isinstance(analysis, dict):
            logger.info(f"Generated database properties for {database_id}: {analysis}")
            return {
                "description": analysis.get("description"),
                "domain": analysis.get("domain")
            }
        return {}

    # ========== Deep Table Semantics Analysis ==========
    
    def _get_unique_columns_for_table(self, database_id: str, table_name: str, all_columns: List[str]) -> List[str]:
        """
        Get unique columns for a table by checking against similar_columns.json.
        
        Args:
            database_id: Database identifier
            table_name: Table name
            all_columns: All column names in this table
            
        Returns:
            List of unique column names (columns not appearing in similar pairs)
            
        NOTE: If similar_columns.json doesn't exist (e.g., join_path_discovery hasn't run yet),
        this method returns all columns as unique (conservative approach).
        """
        similar_columns_file = Path(f"output/similar_columns/{database_id}_similar_columns.json")
        
        if not similar_columns_file.exists():
            logger.warning(
                f"Similar columns file not found: {similar_columns_file}. "
                f"This is expected if join path discovery hasn't run yet. "
                f"Treating all columns as unique for table analysis."
            )
            # Return all columns as unique (conservative approach)
            return all_columns
        
        with open(similar_columns_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Collect all columns from this table that appear in similar pairs
        similar_columns_set = set()
        for pair in data.get('similar_column_pairs', []):
            if pair.get('table_a') == table_name:
                similar_columns_set.add(pair.get('column_a'))
            if pair.get('table_b') == table_name:
                similar_columns_set.add(pair.get('column_b'))
        
        # Unique columns are those NOT in similar pairs
        unique_columns = [col for col in all_columns if col not in similar_columns_set]
        
        return unique_columns
    
    def _extract_table_semantic_needs(self, table_needs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Extract tables that need deep semantic analysis
        
        Args:
            table_needs: Dict[table_name, List[missing_fields]]
        
        Returns:
            Dict[table_name, List[semantic_fields_needed]]
        """
        semantic_fields = {'table_role', 'row_definition', 'description'}
        
        tables_needing_semantics = {}
        for table_name, missing_fields in table_needs.items():
            semantic_missing = [f for f in missing_fields if f in semantic_fields]
            if semantic_missing:
                tables_needing_semantics[table_name] = semantic_missing
        
        return tables_needing_semantics
    
    def _analyze_single_table_semantics(
        self, 
        table_name: str, 
        context: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Analyze a single table's deep semantics using LLM
        
        This method integrates:
        - DDL information (PK, FK, column types)
        - Data statistics (row count, cardinality, null ratios)
        - Topology features (in-degree, out-degree)
        - Sample data
        
        Returns:
            Dict with keys: table_role, row_definition, description
        """
        logger.info(f"Analyzing deep semantics for table: {table_name}")
        
        table_detail, unique_columns = self._build_table_detail_for_semantics(table_name, context)
        
        topology_summary = self._build_topology_summary(context)
        
        prompt = PromptFactory.format_table_semantics_prompt(
            table_name=table_name,
            db_domain=context.get('domain', 'Unknown'),
            db_description=context.get('description', 'Unknown'),
            topology_summary=topology_summary,
            table_detail=table_detail,
            unique_columns=unique_columns
        )

        print('-'*100)
        print(prompt)
        print('-'*100)
        
        # Call LLM
        response = self._call_llm(prompt)
        print(response)
        result = self._parse_json_from_response(response)
        
        if not isinstance(result, dict):
            logger.warning(f"Failed to parse LLM response for table {table_name}")
            return None
        
        # Validate table_role with hard topology rules
        table_role = result.get('table_role')
        valid_roles = {'Fact', 'Dimension', 'Bridge', 'Lookup'}
        if table_role not in valid_roles:
            error_msg = f"Invalid table_role '{table_role}' for table '{table_name}'. Must be one of {valid_roles}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Enforce topology-driven classification overrides
        table_ctx = context.get('tables', {}).get(table_name, {})
        in_deg_raw = table_ctx.get('fk_in_degree', 0)
        out_deg_raw = table_ctx.get('fk_out_degree', 0)

        in_deg = 0 if in_deg_raw is None else int(in_deg_raw) if isinstance(in_deg_raw, (int, float)) else 0
        out_deg = 0 if out_deg_raw is None else int(out_deg_raw) if isinstance(out_deg_raw, (int, float)) else 0

        required_role = None
        if in_deg > 0 and out_deg == 0:
            required_role = 'Dimension'
        elif out_deg > 0 and in_deg == 0:
            required_role = 'Fact'

        if required_role and table_role != required_role:
            error_msg = (
                f"Topology rule violated for table '{table_name}': "
                f"in-degree={in_deg}, out-degree={out_deg} requires role '{required_role}', "
                f"but LLM returned '{table_role}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Generated semantics for table '{table_name}': role={table_role}")
        return result
    
    def _build_topology_summary(self, context: Dict) -> str:
        """Build a summary of table topology patterns"""
        tables_info = context.get('tables', {})
        
        # Identify patterns
        high_in_degree = []
        high_out_degree = []
        small_tables = []
        isolated_tables = []
        
        for table_name, table_data in tables_info.items():
            in_deg = table_data.get('fk_in_degree', 0)
            out_deg = table_data.get('fk_out_degree', 0)
            row_count = table_data.get('row_count', 0)
            
            # Convert None values to 0 for comparison
            in_deg = 0 if in_deg is None else int(in_deg) if isinstance(in_deg, (int, float)) else 0
            out_deg = 0 if out_deg is None else int(out_deg) if isinstance(out_deg, (int, float)) else 0
            row_count = 0 if row_count is None else int(row_count) if isinstance(row_count, (int, float)) else 0
            
            if in_deg >= 2:
                high_in_degree.append(f"{table_name} (referenced by {in_deg} tables)")
            if out_deg >= 2:
                high_out_degree.append(f"{table_name} (references {out_deg} tables)")
            if row_count > 0 and row_count < 100:
                small_tables.append(f"{table_name} ({row_count} rows)")
            if in_deg == 0 and out_deg == 0:
                isolated_tables.append(table_name)
        
        summary_parts = []
        if high_in_degree:
            summary_parts.append(f"• High In-Degree (potential Dimensions): {', '.join(high_in_degree)}")
        if high_out_degree:
            summary_parts.append(f"• High Out-Degree (potential Facts): {', '.join(high_out_degree)}")
        if small_tables:
            summary_parts.append(f"• Small Tables (potential Lookups): {', '.join(small_tables)}")
        if isolated_tables:
            summary_parts.append(f"• Isolated Tables (no FK relationships): {', '.join(isolated_tables)}")
        
        return "\n".join(summary_parts) if summary_parts else "No significant topology patterns detected."
    
    def _build_table_detail_for_semantics(self, table_name: str, context: Dict) -> str:
        """
        Build detailed information for a single table (for semantic analysis prompt)
        
        Includes:
        - Basic stats (row count, column count)
        - Primary keys
        - Topology features (FK in/out degree)
        - Unique columns identification
        - Top-5 column profiling (Distinct Count, Top-3 Values, Null Ratio)
        - Reordered sample data (Unique Columns and Top-5 Columns first)
        """
        table_ctx = context.get('tables', {}).get(table_name, {})
        columns_ctx = context.get('columns', {}).get(table_name, {})
        database_id = context.get('database_id', 'unknown')
        
        # === Basic Information ===
        row_count = table_ctx.get('row_count', 'Unknown')
        column_count = len(columns_ctx) if columns_ctx else table_ctx.get('column_count', 'Unknown')
        pk = table_ctx.get('primary_keys', [])
        fks = table_ctx.get('foreign_keys', [])
        
        # === Topology Features ===
        in_degree = table_ctx.get('fk_in_degree', 0)
        out_degree = table_ctx.get('fk_out_degree', 0)
        # Convert None values to 0 for safe comparison and formatting
        in_degree = 0 if in_degree is None else int(in_degree) if isinstance(in_degree, (int, float)) else 0
        out_degree = 0 if out_degree is None else int(out_degree) if isinstance(out_degree, (int, float)) else 0
        
        # === Load Unique Columns ===
        all_columns = list(columns_ctx.keys()) if columns_ctx else []
        unique_columns = []
        if all_columns:
            unique_columns = self._get_unique_columns_for_table(database_id, table_name, all_columns)
        
        # === Top-5 Column Profiling ===
        # Get first 5 columns (usually PK or important identification columns)
        top_5_cols = all_columns[:5] if len(all_columns) >= 5 else all_columns
        top_5_profiling = []
        
        for col_name in top_5_cols:
            col_meta = columns_ctx.get(col_name, {})
            distinct_count = col_meta.get('distinct_count')
            null_count = col_meta.get('null_count', 0)
            top_k_values = col_meta.get('top_k_values', {})
            
            # Calculate null ratio
            null_ratio = None
            # Ensure null_count is a number
            null_count = 0 if null_count is None else int(null_count) if isinstance(null_count, (int, float)) else 0
            if isinstance(row_count, int) and row_count > 0:
                null_ratio = null_count / row_count
            
            # Get top 3 values
            top_3_values = []
            if isinstance(top_k_values, dict):
                top_3_values = list(top_k_values.keys())[:3]
            
            profiling_parts = [f"  - **{col_name}**:"]
            if distinct_count is not None:
                profiling_parts.append(f"Distinct Count: {distinct_count}")
            if null_ratio is not None:
                profiling_parts.append(f"Null Ratio: {null_ratio:.2%}")
            if top_3_values:
                values_str = ", ".join([str(v)[:50] for v in top_3_values])
                profiling_parts.append(f"Top-3 Values: {values_str}")
            
            top_5_profiling.append(" ".join(profiling_parts))
        
        top_5_profiling_text = "\n".join(top_5_profiling) if top_5_profiling else "  (No column statistics available)"
        
        # === Reordered Sample Data ===
        sample_data = table_ctx.get('sample_data', {})
        max_sample_cols = 30  # Limit columns in sample to avoid overwhelming LLM
        reordered_cols = []  # Initialize outside if block for later use
        unique_cols_in_sample = 0  # Count unique columns in reordered sample
        
        if sample_data:
            all_cols = list(sample_data.keys())
            
            # Reorder: Top-5 Columns first, then Unique Columns, then rest
            # 1. Add top-5 columns (if not already added)
            for col in top_5_cols:
                if col in all_cols and col not in reordered_cols:
                    reordered_cols.append(col)
            
            # 2. Add unique columns and count how many are included
            for col in unique_columns:
                if col in all_cols and col not in reordered_cols:
                    reordered_cols.append(col)
                    if len(reordered_cols) <= max_sample_cols:
                        unique_cols_in_sample += 1
            
            # 3. Add remaining columns
            for col in all_cols:
                if col not in reordered_cols:
                    reordered_cols.append(col)
                    if len(reordered_cols) >= max_sample_cols:
                        break
            
            # Create reordered sample data dict
            reordered_sample_data = {col: sample_data[col] for col in reordered_cols[:max_sample_cols] if col in sample_data}
            
            sample_md = self._create_sample_data_markdown(reordered_sample_data, max_cols=len(reordered_sample_data), max_rows=5)
            
            if len(all_cols) > max_sample_cols:
                sample_md += f"\n  (Showing {len(reordered_sample_data)} of {len(all_cols)} columns, reordered with Top-5 Columns and Unique Columns first)"
        else:
            sample_md = "No sample data available"
        
        # === FK Summary ===
        fk_summary = ""
        if fks:
            fk_lines = []
            for fk in fks:
                if isinstance(fk, dict):
                    ref_table = fk.get('referred_table', 'Unknown')
                    fk_col = fk.get('column', 'Unknown')
                    fk_lines.append(f"  - {fk_col} → {ref_table}")
            if fk_lines:
                fk_summary = "\n".join(fk_lines)
        
        # === Unique Columns Annotation ===
        unique_cols_note = ""
        if unique_columns:
            total_unique = len(unique_columns)
            if sample_data and unique_cols_in_sample > 0:
                unique_cols_note = f"**Note:** This table has {total_unique} unique column(s) (columns that appear only in this table or rarely in similar pairs). {unique_cols_in_sample} of them are included in the reordered sample data above, prioritized at the front. These unique columns define the table's unique value and purpose."
            else:
                unique_cols_note = f"**Note:** This table has {total_unique} unique column(s) (columns that appear only in this table or rarely in similar pairs). These unique columns define the table's unique value and purpose."
        else:
            unique_cols_note = "**Note:** No unique columns identified for this table."
        
        # === Assemble Detail ===
        detail = f"""
**Table: {table_name}**

**Basic Statistics:**
- Row Count: {row_count}
- Column Count: {column_count}
- Primary Key: {', '.join(pk) if pk else 'None'}

**Topology Features:**
- FK In-Degree: {in_degree} (this table is referenced by {in_degree} other tables)
- FK Out-Degree: {out_degree} (this table references {out_degree} other tables)

**Foreign Key Relationships:**
{fk_summary if fk_summary else '  None'}

{unique_cols_note}

**Top-5 Column Profiling:**
{top_5_profiling_text}

**Sample Data (First 5 Rows, Reordered):**
{sample_md}
"""
        
        return detail, unique_columns
    

    def _analyze_single_column_with_top_k_values(self, table_name: str, column_name: str, fields_needed: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single column using top_k_values from metadata when sample data is insufficient."""
        # Get column metadata
        column_meta = context.get('columns', {}).get(table_name, {}).get(column_name, {})
        top_k_values = column_meta.get('top_k_values', {})
        
        if not top_k_values or not isinstance(top_k_values, dict):
            logger.warning(f"No top_k_values available for {table_name}.{column_name}")
            return {}
        
        # Get table description for context
        table_info = context.get('tables', {}).get(table_name, {})
        table_description = table_info.get('description', '')
        
        # Get whole_column_name from column metadata
        whole_column_name = column_meta.get('whole_column_name', '')
        
        # Prepare top-k values list (show values, optionally with frequencies)
        value_list = []
        for value, frequency in list(top_k_values.items())[:10]:  # Limit to top 10
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + '...'
            value_list.append(f"  - {value_str}")
        
        values_text = "\n".join(value_list) if value_list else "  (No values available)"
        
        # Build dynamic prompt based on needed fields
        instructions, output_schema = self._build_column_analysis_schema(fields_needed)
        
        # Build prompt using PromptFactory
        prompt = PromptFactory.format_column_analysis_with_top_k_prompt(
            table_name=table_name,
            column_name=column_name,
            table_description=table_description,
            top_values_text=values_text,
            instructions=instructions,
            output_schema=output_schema,
            whole_column_name=whole_column_name  # Pass whole_column_name metadata
        )
        
        logger.info(f"Analyzing {table_name}.{column_name} using top_k_values (no sample data available)")
        response_text = self._call_llm(prompt)
        # print('=============== prompt ===============')
        # print(prompt)
        # print('=============== prompt ===============')
        # print('=============== response_text ===============')
        # print(response_text)
        # print('=============== response_text ===============')
        analysis = self._parse_json_from_response(response_text)
        print('=============== analysis ===============')
        print(analysis)
        print('=============== analysis ===============')
        if not isinstance(analysis, dict):
            logger.warning(f"Failed to parse LLM response for {table_name}.{column_name}")
            return {}
        
        # Process and return the analysis (similar to batch processing but for single column)
        cleaned_analysis = {}
        
        # Process description field
        self._process_description_field(analysis, cleaned_analysis)
        
        # Process pattern field (simple mode, no validation)
        pattern_data = self._extract_pattern_data(analysis)
        if pattern_data and isinstance(pattern_data, dict):
            self._process_pattern_field_simple(pattern_data, cleaned_analysis, table_name, column_name)

        return cleaned_analysis

    def _process_batch_column_results(
        self,
        batch_analysis: Dict[str, Any],
        column_batch: List[str],
        columns_to_analyze: Dict[str, List[str]],
        table_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and validate results from a batch column analysis.
        
        Args:
            batch_analysis: LLM response containing analysis for multiple columns
            column_batch: List of column names in this batch
            columns_to_analyze: Original dict mapping columns to needed fields
            table_name: Name of the table
            context: Full context including column metadata
            
        Returns:
            Dictionary mapping column names to cleaned analysis results
        """
        results = {}
        
        for col_name, result_fields in batch_analysis.items():
            if col_name not in column_batch or col_name not in columns_to_analyze:
                continue
            if not isinstance(result_fields, dict):
                continue
            
            # Get distinct values for validation
            column_info = context.get('columns', {}).get(table_name, {}).get(col_name, {})
            distinct_values = column_info.get('distinct_values')
            
            cleaned_analysis = {}
            
            # Process description field
            self._process_description_field(result_fields, cleaned_analysis)

            # Process pattern field with validation
            pattern_data = self._extract_pattern_data(result_fields)
            if pattern_data and isinstance(pattern_data, dict):
                self._process_pattern_field_with_validation(
                    pattern_data, cleaned_analysis, table_name, col_name, distinct_values
                )

            if cleaned_analysis:
                results[col_name] = cleaned_analysis
        
        return results

    def _batch_analyze_columns_for_table(self, table_name: str, columns_to_analyze: Dict[str, List[str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Performs batch analysis for multiple columns within a single table.
        Processes columns in batches of 5, providing complete table data for each batch.
        For columns with insufficient sample data, uses top_k_values from metadata for individual analysis.
        Dynamically builds prompt based on which fields are needed (description and/or pattern).
        """
        # Get table information
        table_info = context.get('tables', {}).get(table_name, {})
        table_description = table_info.get('description', '')
        sample_data = table_info.get('sample_data')
        
        # Separate columns into those with sufficient data and those without
        columns_with_data = []
        columns_without_data = []
        
        for col_name in columns_to_analyze.keys():
            if self._check_column_has_sufficient_data(sample_data, col_name):
                columns_with_data.append(col_name)
            else:
                columns_without_data.append(col_name)
                logger.info(f"Column {table_name}.{col_name} has insufficient sample data, will use top_k_values instead")
        
        all_results = {}
        
        # Process columns with sufficient sample data in batches
        if columns_with_data:
            # Get primary key columns for this table to surface them in prompts
            primary_keys = table_info.get('primary_keys', [])
            
            batch_size = 5
            column_batches = [columns_with_data[i:i + batch_size] for i in range(0, len(columns_with_data), batch_size)]
            
            for batch_idx, column_batch in enumerate(column_batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(column_batches)} for table '{table_name}': {column_batch}")
                
                # Build display columns for the prompt:
                # - Always put primary key columns at the front (even if not part of this batch)
                # - Follow with the batch columns (keeping their relative order, no duplicates)
                primary_keys_in_table = [pk for pk in primary_keys if pk]
                batch_cols_no_dups = [col for col in column_batch if col not in primary_keys_in_table]
                display_columns = primary_keys_in_table + batch_cols_no_dups
                if display_columns != column_batch:
                    logger.debug(
                        f"Display columns for prompt (PKs first, deduped): {display_columns}"
                    )
                
                # Create markdown table for this batch using the unified method
                # Use display_columns so PK columns appear first to give structure cues to LLM
                table_md = self._create_sample_data_markdown(
                    sample_data, 
                    columns=display_columns, 
                    max_rows=50,
                    cell_max_length=100  # Slightly longer for pattern analysis
                )
                
                # Determine which fields are needed for this batch
                batch_fields = set()
                for col_name in column_batch:  # Use original column_batch for field checking
                    if col_name in columns_to_analyze:
                        batch_fields.update(columns_to_analyze[col_name])
                
                # Build dynamic prompt based on needed fields
                instructions, output_schema = self._build_column_analysis_schema(list(batch_fields))
                
                # Get column metadata (including whole_column_name) for prompt
                column_metadata = {}
                for col_name in column_batch:
                    col_meta = context.get('columns', {}).get(table_name, {}).get(col_name, {})
                    if col_meta:
                        column_metadata[col_name] = col_meta
                
                # Build prompt using PromptFactory
                # Pass column_batch (not reordered_batch) to explicitly list columns to analyze
                # This ensures LLM only analyzes the intended columns, not primary keys if they're not in the batch
                prompt = PromptFactory.format_column_analysis_with_data_prompt(
                    table_markdown=table_md,
                    instructions=instructions,
                    output_schema=output_schema,
                    table_name=table_name,
                    table_description=table_description,
                    columns_to_analyze=column_batch,  # Explicitly list columns to analyze
                    column_metadata=column_metadata  # Pass column metadata including whole_column_name
                )
                
                response_text = self._call_llm(prompt)
                batch_analysis = self._parse_json_from_response(response_text)

                print('=============== analysis ===============')
                print(batch_analysis)
                print('=============== analysis ===============')

                if not isinstance(batch_analysis, dict):
                    logger.warning(f"Failed to parse LLM response for batch {batch_idx + 1}, skipping")
                    continue
                
                # Process and validate results for this batch
                batch_results = self._process_batch_column_results(
                    batch_analysis, column_batch, columns_to_analyze, table_name, context
                )

                print('=============== batch_results ===============')
                print(batch_results)
                print('=============== batch_results ===============')
                all_results.update(batch_results)
        
        # Process columns without sufficient sample data using top_k_values
        if columns_without_data:
            logger.info(f"Processing {len(columns_without_data)} columns with insufficient sample data using top_k_values: {columns_without_data}")
            for col_name in columns_without_data:
                fields_needed = columns_to_analyze.get(col_name, [])
                if not fields_needed:
                    continue
                
                # Analyze using top_k_values
                col_analysis = self._analyze_single_column_with_top_k_values(
                    table_name, col_name, fields_needed, context
                )
                print('=============== final clean col_analysis ===============')
                print(col_analysis)
                print('=============== final clean col_analysis ===============')
                if col_analysis:
                    all_results[col_name] = col_analysis
                    logger.info(f"Generated metadata for {table_name}.{col_name} using top_k_values")
                else:
                    logger.warning(f"Failed to generate metadata for {table_name}.{col_name} using top_k_values")
        
        logger.info(f"Batch analysis generated metadata for columns in '{table_name}': {list(all_results.keys())}")
        return all_results
    
    
    # ===================================================================================
    # LLM API Call Wrapper
    # ===================================================================================
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate response"""
        try:
            # Use the unified client interface
            # For JSON prompts, try to use call_json if available, otherwise use regular call
            if self.provider == 'openai' and hasattr(self.llm_client, 'call_json'):
                return self.llm_client.call_json(prompt)
            else:
                return self.llm_client.call(prompt)
            
        except Exception as e:
            raise LLMProviderError(self.provider, f"LLM API call failed: {e}")

    # ===================================================================================
    # Natural Language Description Generation (Short/Long Descriptions)
    # ===================================================================================
    
    def _generate_mechanical_description(self, column_meta: Dict[str, Any], row_count: int) -> str:
        """
        Generate mechanical (statistical) description from column metadata.
        
        This dynamically generates a description from existing profiling statistics,
        including: null counts, cardinality, min/max values, top-K values, 
        data shape (length, prefix/suffix), and pattern information.
        
        Args:
            column_meta: Dictionary containing column metadata fields
            row_count: Total number of rows in the table
            
        Returns:
            Mechanical description string
        """
        parts = []
        column_name = column_meta.get('column_name', 'Column')
        
        # 1. Null count and total records
        null_count = column_meta.get('null_count', 0)
        non_null_count = row_count - null_count
        parts.append(f"Column {column_name} has {null_count} NULL values out of {row_count} records.")
        
        # 2. Cardinality (distinct values)
        distinct_count = column_meta.get('distinct_count')
        if distinct_count is not None:
            parts.append(f"There are {distinct_count} distinct values.")
        
        # 3. Min/Max values
        min_val = column_meta.get('min_value')
        max_val = column_meta.get('max_value')
        if min_val is not None and max_val is not None:
            parts.append(f"The minimum value is '{min_val}' and the maximum value is '{max_val}'.")
        
        # 4. Top-K values (most common values)
        top_k = column_meta.get('top_k_values')
        if top_k and isinstance(top_k, dict) and len(top_k) > 0:
            # Check if this column has mostly unique values (distinct_count close to non_null_count)
            # If so, these are sample values, not "most common"
            is_mostly_unique = False
            if distinct_count is not None and non_null_count > 0:
                uniqueness_ratio = distinct_count / non_null_count if non_null_count > 0 else 0
                # If > 95% unique, treat as sample values
                if uniqueness_ratio > 0.95:
                    is_mostly_unique = True
            
            # Get top 5 values
            top_values = list(top_k.keys())[:5]
            
            # Truncate long values for display (but keep at least one full example)
            display_values = []
            for i, v in enumerate(top_values):
                v_str = str(v)
                # Keep first value full if possible, truncate others
                if i == 0 and len(v_str) <= 200:
                    display_values.append(v_str)
                elif len(v_str) > 80:
                    display_values.append(v_str[:80] + '...')
                else:
                    display_values.append(v_str)
            
            values_str = "', '".join(display_values)
            
            if is_mostly_unique:
                parts.append(f"Sample non-NULL column values include '{values_str}' (each value appears approximately once).")
            else:
                # Show frequency information if available
                freq_info = []
                for v in top_values[:3]:  # Show frequency for top 3
                    freq = top_k.get(v, 0)
                    if freq > 1:
                        v_display = str(v)[:50] + ('...' if len(str(v)) > 50 else '')
                        freq_info.append(f"'{v_display}' (appears {freq} times)")
                
                if freq_info:
                    parts.append(f"Most common non-NULL column values: {', '.join(freq_info)}.")
                else:
                    parts.append(f"Sample non-NULL column values include '{values_str}'.")
        
        # 5. Data shape: length statistics
        min_len = column_meta.get('min_length')
        max_len = column_meta.get('max_length')
        avg_len = column_meta.get('avg_length')
        
        if min_len is not None and max_len is not None:
            if min_len == max_len:
                parts.append(f"The values are always {min_len} characters long.")
            else:
                avg_str = f", average {avg_len:.1f}" if avg_len else ""
                parts.append(f"Value lengths range from {min_len} to {max_len} characters{avg_str}.")
        
        # 6. Fixed prefix/suffix (Fixity)
        fixed_prefix = column_meta.get('fixed_prefix')
        fixed_suffix = column_meta.get('fixed_suffix')
        if fixed_prefix:
            parts.append(f"All values start with the prefix '{fixed_prefix}'.")
        if fixed_suffix:
            parts.append(f"All values end with the suffix '{fixed_suffix}'.")
        
        # 7. Data type information
        data_type = column_meta.get('data_type')
        semantic_type = column_meta.get('semantic_type')
        if data_type:
            parts.append(f"Database type: {data_type}.")
        if semantic_type:
            parts.append(f"Inferred semantic type: {semantic_type}.")
        
        # 8. Pattern description (if available)
        pattern_desc = column_meta.get('pattern_description')
        if pattern_desc:
            parts.append(f"Pattern: {pattern_desc}")
        
        return " ".join(parts)
    
    def _batch_generate_natural_descriptions(
        self, 
        table_name: str, 
        columns_to_analyze: Dict[str, List[str]], 
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate short and long natural language descriptions for multiple columns.
        
        This is called AFTER all other column metadata has been generated, so that
        pattern_description and other LLM-generated fields are available.
        
        Args:
            table_name: Name of the table
            columns_to_analyze: Dict mapping column names to list of needed fields 
                              (e.g., {'col1': ['short_description', 'long_description']})
            context: Full context including all available column metadata
            
        Returns:
            Dict mapping column names to their generated descriptions
        """
        # Get table row count for mechanical description
        row_count = 0
        if 'tables' in context and table_name in context['tables']:
            table_info = context['tables'][table_name]
            row_count = table_info.get('row_count', 0)
        
        # Get all column names in the table for context
        other_columns = []
        if 'tables' in context and table_name in context['tables']:
            other_columns = context['tables'][table_name].get('columns', [])
        
        # Process columns in batches of 3 (fewer than regular analysis since prompts are longer)
        column_names = list(columns_to_analyze.keys())
        batch_size = 3
        column_batches = [column_names[i:i + batch_size] for i in range(0, len(column_names), batch_size)]
        
        all_results = {}
        for batch_idx, column_batch in enumerate(column_batches):
            logger.info(f"Generating natural descriptions batch {batch_idx + 1}/{len(column_batches)} for table '{table_name}': {column_batch}")
            
            # Build batch prompt
            column_descriptions = []
            for col_name in column_batch:
                # Get column metadata from context
                col_meta = {}
                if 'columns' in context and table_name in context['columns']:
                    col_meta = context['columns'][table_name].get(col_name, {})
                
                # Generate mechanical description
                mechanical_desc = self._generate_mechanical_description(col_meta, row_count)
                
                # Get existing description if available
                existing_desc = col_meta.get('description', '')
                
                column_descriptions.append({
                    'column_name': col_name,
                    'mechanical_description': mechanical_desc,
                    'existing_description': existing_desc,
                    'other_columns': [c for c in other_columns if c != col_name],
                    'column_metadata': col_meta  # Pass full metadata for example extraction
                })
            
            # Build prompt
            prompt = self._build_natural_description_prompt(table_name, column_descriptions)
            
            # Call LLM
            response_text = self._call_llm(prompt)
            batch_analysis = self._parse_json_from_response(response_text)
            
            if not isinstance(batch_analysis, dict):
                logger.warning(f"Failed to parse LLM response for natural descriptions batch {batch_idx + 1}")
                continue
            
            # Process results
            for col_name, descriptions in batch_analysis.items():
                if col_name not in column_batch:
                    continue
                if not isinstance(descriptions, dict):
                    continue
                
                result = {}
                
                # Extract short_description
                short_desc = descriptions.get('short_description')
                if short_desc and isinstance(short_desc, str) and len(short_desc) > 10:
                    result['short_description'] = short_desc
                
                # Extract long_description
                long_desc = descriptions.get('long_description')
                if long_desc and isinstance(long_desc, str) and len(long_desc) > 10:
                    result['long_description'] = long_desc
                
                if result:
                    all_results[col_name] = result
                    logger.info(f"Generated natural descriptions for {table_name}.{col_name}")
        
        return all_results
    
    def _build_natural_description_prompt(
        self, 
        table_name: str, 
        column_descriptions: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for generating short and long natural language descriptions.
        
        Args:
            table_name: Name of the table
            column_descriptions: List of dicts containing column info and mechanical descriptions
            
        Returns:
            Formatted prompt string
        """
        # Build column analysis sections
        column_sections = []
        for col_info in column_descriptions:
            col_name = col_info['column_name']
            mechanical = col_info['mechanical_description']
            existing = col_info['existing_description']
            others = col_info['other_columns'][:10]  # Limit to first 10 other columns
            col_meta = col_info.get('column_metadata', {})
            
            # Get sample values for examples
            top_k = col_meta.get('top_k_values')
            sample_values = []
            if top_k and isinstance(top_k, dict):
                # Get at least one full example (prefer shorter ones for readability)
                sorted_by_length = sorted(top_k.keys(), key=lambda x: len(str(x)))
                for v in sorted_by_length[:3]:  # Get up to 3 examples
                    v_str = str(v)
                    if len(v_str) <= 500:  # Include full value if reasonable length
                        sample_values.append(v_str)
                        if len(sample_values) >= 2:  # Get at least 2 examples
                            break
            
            section = f"""
**Column: {col_name}**

Statistical Profile:
{mechanical}
"""
            if existing:
                section += f"\nExisting Semantic Description: {existing}\n"
            
            if sample_values:
                section += f"\nExample Values:\n"
                for i, val in enumerate(sample_values[:2], 1):  # Show up to 2 full examples
                    section += f"  Example {i}: {val}\n"
            
            if others:
                section += f"\nOther columns in table: {', '.join(others)}\n"
            
            column_sections.append(section)
        
        # Build prompt using PromptFactory
        prompt = PromptFactory.format_natural_description_prompt(
            table_name=table_name,
            column_sections=column_sections
        )
        
        return prompt
    
    # ===================================================================================
    # Relationship Semantics Analysis (Business Meaning Generation)
    # ===================================================================================

    def _build_relationship_sample_rows(
        self,
        table_name: str,
        key_columns: List[str],
        context: Dict[str, Any],
        extra_columns: int = 5,
        max_rows: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Build row-level samples for join keys plus top columns to give relationship prompts more context.
        Returns rows and the column order used so we can format markdown tables consistently.
        """
        table_info = context.get('tables', {}).get(table_name, {})
        sample_data = table_info.get('sample_data', {})
        if not sample_data or not key_columns:
            return [], []

        column_order: List[str] = []
        for col in key_columns:
            if col in sample_data and col not in column_order:
                column_order.append(col)

        # Add first N columns from the table for broader context
        for col in table_info.get('columns', [])[:extra_columns]:
            if col in sample_data and col not in column_order:
                column_order.append(col)

        if not column_order:
            return [], []

        max_available_rows = max(len(sample_data.get(c, [])) for c in column_order)
        num_rows = min(max_available_rows, max_rows)

        rows: List[Dict[str, Any]] = []
        for idx in range(num_rows):
            row = {}
            for col in column_order:
                col_values = sample_data.get(col, [])
                if idx < len(col_values):
                    row[col] = col_values[idx]
            if row:
                rows.append(row)

        return rows, column_order

    def _format_sample_rows_as_markdown(self, sample_rows: List[Dict[str, Any]], column_order: List[str]) -> str:
        """Format sample rows into a markdown table."""
        if not sample_rows or not column_order:
            return "No sample rows available"

        # Build column-wise data from row dicts
        sample_data: Dict[str, List[Any]] = {col: [] for col in column_order}
        for row in sample_rows:
            for col in column_order:
                sample_data[col].append(row.get(col, ""))

        return self._create_sample_data_markdown(
            sample_data=sample_data,
            max_cols=len(column_order),
            max_rows=len(sample_rows),
            columns=column_order,
            cell_max_length=100
        )
    
    def _analyze_relationship_semantics(
        self,
        context: Dict[str, Any],
        relationships_needing_meaning: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze relationship semantics and generate business meaning.
        
        This method processes relationships individually (not in batches) and generates
        business meaning descriptions based on:
        - Table descriptions and roles
        - Cardinality information
        - Sample data
        
        Args:
            context: Full context including tables, columns, and relationships
            relationships_needing_meaning: List of relationship dicts that need business_meaning
            
        Returns:
            List of relationship dicts with business_meaning added
        """
        results = []
        
        # Get all relationships from context for lookup
        context_relationships = context.get('relationships', [])
        
        for rel in relationships_needing_meaning:
            source_table = rel.get('source_table')
            target_table = rel.get('target_table')
            source_columns = rel.get('source_columns', [])
            target_columns = rel.get('target_columns', [])
            
            if not source_table or not target_table:
                continue
            
            # Find the corresponding relationship in context to get complete information
            # This is important because needs['relationships'] may only contain partial data
            complete_rel = None
            cardinality = None
            relationship_type = rel.get('relationship_type')
            database_id = rel.get('database_id')
            source = rel.get('source', 'llm_analysis')
            
            for ctx_rel in context_relationships:
                if (ctx_rel.get('source_table') == source_table and 
                    ctx_rel.get('target_table') == target_table and
                    ctx_rel.get('source_columns') == source_columns and
                    ctx_rel.get('target_columns') == target_columns):
                    # Found matching relationship, use it as the base
                    complete_rel = ctx_rel.copy()
                    cardinality = ctx_rel.get('cardinality')
                    if not relationship_type:
                        relationship_type = ctx_rel.get('relationship_type')
                    if not database_id:
                        database_id = ctx_rel.get('database_id')
                    if not source or source == 'llm_analysis':
                        source = ctx_rel.get('source', 'llm_analysis')
                    break
            
            # Fallback to rel's values if not found in context
            if cardinality is None:
                cardinality = rel.get('cardinality')
            if not relationship_type:
                relationship_type = rel.get('relationship_type', 'value_overlap_join')  # Default for join_path_discovery relationships
            if not database_id:
                database_id = rel.get('database_id')
            
            # Get table metadata from context
            source_table_info = context.get('tables', {}).get(source_table, {})
            target_table_info = context.get('tables', {}).get(target_table, {})
            
            # Get table descriptions and roles
            source_description = source_table_info.get('description', '')
            target_description = target_table_info.get('description', '')
            source_role = source_table_info.get('table_role')  # Keep None if not available, don't use empty string
            target_role = target_table_info.get('table_role')  # Keep None if not available, don't use empty string
            
            # Get sample data for join columns
            source_sample = self._get_column_sample_data(context, source_table, source_columns)
            target_sample = self._get_column_sample_data(context, target_table, target_columns)

            # Build richer sample rows using join keys + top columns for context
            source_sample_rows, source_column_order = self._build_relationship_sample_rows(source_table, source_columns, context)
            target_sample_rows, target_column_order = self._build_relationship_sample_rows(target_table, target_columns, context)
            source_sample_table_md = self._format_sample_rows_as_markdown(source_sample_rows, source_column_order)
            target_sample_table_md = self._format_sample_rows_as_markdown(target_sample_rows, target_column_order)
            source_join_keys = ", ".join(source_columns) if source_columns else "N/A"
            target_join_keys = ", ".join(target_columns) if target_columns else "N/A"
            
            # Build prompt using PromptFactory
            prompt = PromptFactory.format_relationship_semantics_prompt(
                source_table=source_table,
                target_table=target_table,
                source_columns=source_columns,
                target_columns=target_columns,
                cardinality=cardinality,
                source_description=source_description,
                target_description=target_description,
                source_role=source_role,
                target_role=target_role,
                source_sample_table_md=source_sample_table_md,
                target_sample_table_md=target_sample_table_md
            )

            
            response_text = self._call_llm(prompt)

            analysis = self._parse_json_from_response(response_text)
            
            if isinstance(analysis, dict):
                business_meaning = analysis.get('business_meaning')
                if business_meaning and len(business_meaning) > 10:
                    # Build complete relationship result with all required fields
                    if complete_rel:
                        # Use complete relationship from context as base
                        rel_result = complete_rel.copy()
                        # Ensure database_id is set (it might not be in context relationships)
                        if 'database_id' not in rel_result:
                            rel_result['database_id'] = database_id or context.get('database_id', '')
                    else:
                        # Build from scratch with required fields
                        rel_result = {
                            'database_id': database_id or context.get('database_id', ''),
                            'relationship_type': relationship_type or 'value_overlap_join',
                            'source_table': source_table,
                            'target_table': target_table,
                            'source_columns': source_columns,
                            'target_columns': target_columns,
                            'source': source
                        }
                        # Add optional fields if available
                        if cardinality:
                            rel_result['cardinality'] = cardinality
                    
                    # Update with LLM-generated business_meaning
                    rel_result['business_meaning'] = business_meaning
                    
                    # Remove any fields that shouldn't be saved (like 'fields_needed')
                    rel_result.pop('fields_needed', None)
                    
                    results.append(rel_result)
                    logger.info(f"Generated business meaning for {source_table} -> {target_table}: {business_meaning[:80]}...")
                else:
                    logger.warning(f"Failed to generate valid business meaning for {source_table} -> {target_table}")
            else:
                logger.warning(f"Failed to parse LLM response for relationship {source_table} -> {target_table}")
        
        return results
    
    def _get_column_sample_data(
        self,
        context: Dict[str, Any],
        table_name: str,
        columns: List[str]
    ) -> List[Any]:
        """
        Get sample data for specified columns from context.
        
        Args:
            context: Full context including tables and columns metadata
            table_name: Name of the table
            columns: List of column names (usually join key columns)
            
        Returns:
            List of sample values (up to 10) from the first column
        """
        # Early return if no columns provided
        if not columns:
            return []
        
        first_col = columns[0]  # Use first column for sample data
        
        # Try to get from table sample_data first (most reliable, actual row data)
        table_info = context.get('tables', {}).get(table_name, {})
        table_sample_data = table_info.get('sample_data', {})
        
        if table_sample_data and first_col in table_sample_data:
            col_data = table_sample_data[first_col]
            if isinstance(col_data, list) and col_data:
                # Return up to 10 sample values from actual table data
                return list(col_data[:10])
        
        # Fallback: If not found in table sample_data, try to get from column metadata (top_k_values)
        columns_info = context.get('columns', {}).get(table_name, {})
        if first_col in columns_info:
            col_info = columns_info[first_col]
            top_k_values = col_info.get('top_k_values', {})
            if isinstance(top_k_values, dict) and top_k_values:
                # Get up to 10 sample values from top_k_values (most frequent values)
                return list(top_k_values.keys())[:10]
        
        return []
    
    def _build_relationship_semantics_prompt(
        self,
        source_table: str,
        target_table: str,
        source_columns: List[str],
        target_columns: List[str],
        cardinality: Optional[str],
        source_description: str,
        target_description: str,
        source_role: Optional[str],
        target_role: Optional[str],
        source_sample: List[Any],
        target_sample: List[Any]
    ) -> str:
        """
        Build prompt for relationship semantics analysis.
        
        DEPRECATED: Use PromptFactory.format_relationship_semantics_prompt instead.
        This method is kept for backward compatibility.
        """
        return PromptFactory.format_relationship_semantics_prompt(
            source_table=source_table,
            target_table=target_table,
            source_columns=source_columns,
            target_columns=target_columns,
            cardinality=cardinality,
            source_description=source_description,
            target_description=target_description,
            source_role=source_role,
            target_role=target_role,
            source_sample=source_sample,
            target_sample=target_sample
        )