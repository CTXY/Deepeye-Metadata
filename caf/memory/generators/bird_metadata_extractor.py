"""
BIRD Dataset Metadata Extractor

This module extracts and imports metadata from BIRD dataset into SemanticMemoryStore.
It processes the database_description CSV files for each database and converts them
to our metadata format.

IMPORTANT: This extractor only imports data that exists in the original BIRD files.
No additional information is constructed or generated. All fields not present in the
original CSV files are set to None/empty to allow for subsequent generation and validation.
"""

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any

from caf.memory.stores.semantic import SemanticMemoryStore
from caf.memory.types import DatabaseMetadata, TableMetadata, ColumnMetadata, TermDefinition
from caf.memory.generators.llm_analyzer import LLMAnalyzer
from caf.memory.generators.schema_extractor import DatabaseSchemaExtractor

logger = logging.getLogger(__name__)


class BirdMetadataExtractor:
    """
    Extracts BIRD dataset metadata and imports into SemanticMemoryStore.
    
    This extractor only uses data that exists in the original BIRD CSV files.
    No additional information is constructed or generated. Fields not present
    in the original files are set to None/empty to allow for subsequent
    generation and validation.
    """
    
    def __init__(
        self,
        bird_data_dir: Path,
        semantic_store: SemanticMemoryStore,
        llm_config: Optional[Dict[str, Any]] = None,
        only_value_desc: Optional[List[str]] = None
    ):
        """
        Initialize BirdMetadataExtractor.
        
        Args:
            bird_data_dir: Path to BIRD dataset root directory
            semantic_store: SemanticMemoryStore instance to store metadata
            llm_config: LLM configuration dict for parsing value_description
            only_value_desc: Optional filter for value_description extraction
                           (fully-qualified names like "table.column" or "table.*")
        """
        self.bird_data_dir = bird_data_dir
        self.databases_dir = bird_data_dir / "databases" / "dev_databases"
        self.semantic_store = semantic_store
        self.schema_extractor = DatabaseSchemaExtractor()
        
        # Optional filter for value_description extraction
        self.only_value_desc = set()
        if only_value_desc:
            for item in only_value_desc:
                s = str(item).strip()
                if s:
                    self.only_value_desc.add(s.lower())
        
        # Initialize LLM analyzer for parsing value_description
        llm_config = llm_config or {}
        self.llm_analyzer = LLMAnalyzer({
            'provider': llm_config.get('provider', 'openai'),
            'model_name': llm_config.get('model_name', 'gpt-4o-mini'),
            'api_key': llm_config.get('api_key'),
            'base_url': llm_config.get('base_url'),
            'temperature': llm_config.get('temperature', 0.1),
            'max_tokens': llm_config.get('max_tokens', 800),
            'timeout': llm_config.get('timeout', 60)
        })
        
        if not self.databases_dir.exists():
            raise FileNotFoundError(f"BIRD databases directory not found: {self.databases_dir}")
        
        logger.info(f"BirdMetadataExtractor initialized with data dir: {bird_data_dir}")
    
    def get_available_databases(self) -> List[str]:
        """Get list of available database IDs in BIRD dataset."""
        databases = []
        for db_dir in self.databases_dir.iterdir():
            if db_dir.is_dir() and (db_dir / "database_description").exists():
                databases.append(db_dir.name)
        return sorted(databases)
    
    def extract_and_import(self, database_path: str) -> bool:
        """
        Extract and import metadata for a database from its file path.
        
        Args:
            database_path: Path to database file
            
        Returns:
            True if import succeeded, False otherwise
        """
        logger.info(f"Starting metadata extraction for database path: {database_path}")
        
        # Extract actual database_id and schema from database file
        actual_database_id = self.schema_extractor.extract_database_id(database_path)
        actual_schema = self.schema_extractor.extract_actual_schema(database_path)
        
        logger.info(f"Extracted database_id: {actual_database_id}")
        logger.info(f"Found {len(actual_schema['tables'])} tables in database")
        
        # Find corresponding BIRD data directory
        bird_database_id = self._find_bird_database_id(database_path)
        if not bird_database_id:
            raise ValueError(f"No BIRD data found for database: {database_path}")
        
        logger.info(f"Found BIRD data directory: {bird_database_id}")
        
        # Import with schema matching
        self._import_with_schema_matching(
            actual_database_id, actual_schema, bird_database_id, database_path
        )
        
        logger.info(f"Successfully imported metadata for database: {actual_database_id}")
        return True
    
    def _find_bird_database_id(self, database_path: str) -> Optional[str]:
        """Find BIRD database ID that corresponds to the given database file path."""
        db_filename = Path(database_path).stem
        
        # Try direct match first
        if (self.databases_dir / db_filename).exists():
            return db_filename
        
        # Try to find by searching through all BIRD databases
        for bird_db_dir in self.databases_dir.iterdir():
            if bird_db_dir.is_dir():
                # Check if there's a database file with matching name
                db_files = list(bird_db_dir.glob("*.sqlite"))
                if not db_files:
                    db_files = list(bird_db_dir.glob("*.db"))
                
                for db_file in db_files:
                    if db_file.stem == db_filename:
                        return bird_db_dir.name
        
        return None
    
    def _import_with_schema_matching(
        self,
        actual_database_id: str,
        actual_schema: Dict,
        bird_database_id: str,
        database_path: str
    ):
        """Import metadata using schema matching between BIRD data and actual database."""
        db_dir = self.databases_dir / bird_database_id
        desc_dir = db_dir / "database_description"
        
        if not desc_dir.exists():
            raise FileNotFoundError(f"BIRD description directory not found: {desc_dir}")
        
        # Bind to actual database_id
        self.semantic_store.bind_database(actual_database_id)

        # Open a read-only connection to the actual database for sampling values
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        
            # Add database metadata
            self._add_database_metadata(actual_database_id)
            
            # Build table name mapping
            table_mapping = {}
            csv_files = list(desc_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                bird_table_name = csv_file.stem
                try:
                    actual_table_name = self.schema_extractor.find_best_match(
                        bird_table_name, actual_schema['tables']
                    )
                    table_mapping[bird_table_name] = actual_table_name
                    logger.info(f"Table mapping: {bird_table_name} -> {actual_table_name}")
                except ValueError as e:
                    raise ValueError(f"Table mapping failed: {e}")
            
            # Import each table with column mapping
            imported_tables = 0
            imported_columns = 0
            
            for bird_table, actual_table in table_mapping.items():
                csv_file = desc_dir / f"{bird_table}.csv"
                
                # Extract BIRD column names
                bird_columns = self._extract_bird_column_names(csv_file)
                actual_columns = actual_schema['columns'].get(actual_table, [])
                
                # Build column name mapping
                column_mapping = {}
                skipped_columns = []
                for bird_col in bird_columns:
                    try:
                        actual_col = self.schema_extractor.find_best_match(
                            bird_col, actual_columns
                        )
                        column_mapping[bird_col] = actual_col
                        logger.info(f"Column mapping: {bird_table}.{bird_col} -> {actual_table}.{actual_col}")
                    except ValueError as e:
                        # Skip columns that can't be matched instead of failing the entire import
                        skipped_columns.append(bird_col)
                        logger.warning(f"Skipping unmatchable column {bird_table}.{bird_col}: {e}")
                
                if skipped_columns:
                    logger.warning(f"Skipped {len(skipped_columns)} unmatchable columns in table {bird_table}: {skipped_columns}")
                
                if not column_mapping:
                    raise ValueError(f"No columns could be mapped for table {bird_table}. Skipped: {skipped_columns}")
                
                # Import table and column metadata
                self._import_table_metadata_with_mapping(
                    actual_database_id, actual_table, csv_file, column_mapping, conn
                )
                imported_tables += 1
                imported_columns += len(column_mapping)

            # Additionally import term definitions from dev.json (if available)
            try:
                self._import_terms_from_dev(actual_database_id, bird_database_id)
            except Exception as e:
                logger.warning(f"Failed to import term definitions from dev.json for {actual_database_id}: {e}")
            
            # Save all metadata
            self.semantic_store.save_all_metadata()
            
            logger.info(f"Successfully imported {actual_database_id}: {imported_tables} tables, {imported_columns} columns")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def _extract_bird_column_names(self, csv_file: Path) -> List[str]:
        """Extract column names from BIRD CSV file."""
        rows = self._read_csv_with_fallback(csv_file)
        return [row['original_column_name'].strip() for row in rows]
    
    def _import_table_metadata_with_mapping(
        self,
        actual_database_id: str,
        actual_table: str,
        csv_file: Path,
        column_mapping: Dict[str, str],
        conn: Optional[sqlite3.Connection]
    ):
        """Import table and column metadata using column name mapping."""

        # Add table metadata
        table_metadata = TableMetadata(
            database_id=actual_database_id,
            table_name=actual_table,
            description=None
        )
        self.semantic_store.add_table_metadata(table_metadata, source='manual')
        
        # Add column metadata
        rows = self._read_csv_with_fallback(csv_file)
        
        for row in rows:
            bird_column_name = row['original_column_name'].strip()
            
            # Skip columns that weren't successfully mapped
            if bird_column_name not in column_mapping:
                logger.debug(f"Skipping unmapped column: {bird_column_name}")
                continue
                
            actual_column_name = column_mapping[bird_column_name]

            # Try to sample a few distinct values for this column from the actual DB
            sample_values: List[str] = []
            if conn is not None:
                sample_values = self._get_column_sample_values(
                    conn, actual_table, actual_column_name, limit=25
                )
            
            column_metadata = ColumnMetadata(
                database_id=actual_database_id,
                table_name=actual_table,
                column_name=actual_column_name,
                whole_column_name=row['column_name'].strip(),
                description=row['column_description'].strip() if row['column_description'].strip() else None,
                data_format=row['data_format'].strip() if row['data_format'].strip() else None
            )
            self.semantic_store.add_column_metadata(column_metadata, source='manual')
            
            # Handle value_description enrichment if present
            value_description = row['value_description'].strip()
            column_description_text = row['column_description'].strip() if row['column_description'].strip() else ''
            
            # If filter set is provided, only process columns that match the filter
            should_enrich = True
            if self.only_value_desc:
                fqcn = f"{actual_table}.{actual_column_name}".lower()
                table_wildcard = f"{actual_table}.*".lower()
                if fqcn not in self.only_value_desc and table_wildcard not in self.only_value_desc:
                    should_enrich = False

            if value_description and should_enrich:
                try:
                    enriched = self._extract_from_value_description(
                        table_name=actual_table,
                        column_name=actual_column_name,
                        value_description=value_description,
                        column_description=column_description_text,
                        sample_values=sample_values,
                    )
                    
                    if enriched:
                        enriched_fields: Dict[str, Any] = {}
                        if enriched.get('description'):
                            enriched_fields['description'] = enriched['description']
                        if enriched.get('data_format'):
                            enriched_fields['data_format'] = enriched['data_format']
                        if enriched.get('encoding_mapping'):
                            enriched_fields['encoding_mapping'] = enriched['encoding_mapping']
                        if enriched.get('semantic_tags'):
                            enriched_fields['semantic_tags'] = enriched['semantic_tags']
                        
                        if enriched_fields:
                            enriched_metadata = ColumnMetadata(
                                database_id=actual_database_id,
                                table_name=actual_table,
                                column_name=actual_column_name,
                                **enriched_fields
                            )
                            self.semantic_store.add_column_metadata(
                                enriched_metadata, source='llm_bird_value_desc'
                            )

                        # Handle any term definitions extracted from value_description
                        term_defs = enriched.get('term_definitions') or []
                        if term_defs:
                            logger.info(f"Found {len(term_defs)} term definitions in value_description for {actual_table}.{actual_column_name}")
                            for td in term_defs:
                                if self._add_term_definition(
                                    actual_database_id, actual_table, actual_column_name, td
                                ):
                                    logger.debug(f"Added term definition '{td.get('term_name', 'unknown')}' from value_description")
                                else:
                                    logger.warning(f"Failed to add term definition '{td.get('term_name', 'unknown')}' from value_description")
                except Exception as e:
                    logger.warning(f"LLM enrichment failed for {actual_table}.{actual_column_name}: {e}")
    
    def _add_database_metadata(self, database_id: str) -> None:
        """Add database-level metadata."""
        db_metadata = DatabaseMetadata(
            database_id=database_id,
            description=None,  # No description in BIRD files
            domain=None  # Not specified in BIRD files
        )
        self.semantic_store.add_database_metadata(db_metadata, source='manual')
        logger.debug(f"Added database metadata for: {database_id}")
    
    def _read_csv_with_fallback(self, csv_file: Path) -> List[Dict[str, str]]:
        """
        Read CSV file with encoding fallbacks and column name normalization.
        
        Tries UTF-8 with BOM first, then falls back to CP-1252 which
        frequently appears in BIRD descriptions (smart quotes, bullets).
        Also handles common column name typos in BIRD CSV files.
        """
        def normalize_columns(rows):
            """Normalize common column name typos in BIRD CSV files."""
            if not rows:
                return rows
            
            # Create a mapping of common typos to correct names
            column_fixes = {
                'column_desription': 'column_description',  # Common typo in BIRD files
            }
            
            normalized_rows = []
            for row in rows:
                normalized_row = {}
                for key, value in row.items():
                    # Fix column name if it's a known typo
                    fixed_key = column_fixes.get(key, key)
                    normalized_row[fixed_key] = value
                normalized_rows.append(normalized_row)
            
            return normalized_rows
        
        try:
            with open(csv_file, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return normalize_columns(rows)
        except UnicodeDecodeError:
            with open(csv_file, 'r', encoding='cp1252', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return normalize_columns(rows)

    def _get_column_sample_values(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        limit: int = 25
    ) -> List[str]:
        """
        Sample a small set of distinct non-null values for a specific column.
        
        This is used to:
        - Help the LLM understand the actual value space of the column.
        - Post-validate that extracted encoding_mapping keys do appear in data.
        """
        try:
            query = f"""
                SELECT DISTINCT `{column_name}` AS value
                FROM `{table_name}`
                WHERE `{column_name}` IS NOT NULL
                LIMIT {int(limit)}
            """
            cursor = conn.execute(query)
            values: List[str] = []
            for row in cursor:
                v = row["value"]
                if v is not None:
                    values.append(str(v))
            return values
        except Exception as e:
            logger.debug(f"Failed to sample values for {table_name}.{column_name}: {e}")
            return []

    def _filter_encoding_mapping(
        self,
        mapping: Dict[Any, Any],
        value_description: str,
        sample_values: Optional[List[str]]
    ) -> Dict[str, str]:
        """
        Post-filter encoding_mapping to ensure keys are realistic and useful.
        
        Rules:
        - Keys must be present in the actual column sample values if available.
        - If no samples are available, fall back to requiring presence in value_description.
        - Drop keys that look like long sentences or explanations.
        - If after filtering there are fewer than 2 entries, drop the whole mapping.
        """
        if not mapping:
            return {}

        vd_text = value_description.lower()
        samples_normalized: List[str] = []
        if sample_values:
            samples_normalized = [str(v).lower() for v in sample_values if v is not None]

        def key_in_sources(k: str) -> bool:
            k_norm = k.lower().strip()
            if not k_norm:
                return False

            # If we have real data samples, require membership there
            if samples_normalized:
                return any(k_norm == s or k_norm in s for s in samples_normalized)

            # Fallback: key text must at least appear in the value_description
            return k_norm in vd_text

        def looks_like_sentence(k: str) -> bool:
            # Heuristic: too long or contains multiple sentence delimiters
            if len(k) > 80:
                return True
            lower = k.lower()
            bad_tokens = ["for example", "e.g.", "such as", "例如", "比如"]
            if any(t in lower for t in bad_tokens):
                return True
            # Many spaces usually means a full sentence
            if lower.count(" ") > 10:
                return True
            return False

        filtered: Dict[str, str] = {}
        for raw_k, raw_v in mapping.items():
            k = str(raw_k)
            v = str(raw_v)
            if looks_like_sentence(k):
                continue
            if not key_in_sources(k):
                continue
            filtered[k] = v

        # Require at least 2 codes to treat as a meaningful encoding set
        if len(filtered) < 2:
            return {}
        return filtered

    def _extract_from_value_description(
        self,
        table_name: str,
        column_name: str,
        value_description: str,
        column_description: str = "",
        sample_values: Optional[List[str]] = None
    ) -> dict:
        """
        Use LLM to convert BIRD value_description to structured fields.
        
        Returns a dict with optional keys:
        - description
        - data_format
        - encoding_mapping
        - semantic_tags
        - term_definitions: list of business term definitions
        """
        # Prepare a compact representation of sample values for the prompt
        sample_preview = ""
        if sample_values:
            # Deduplicate and truncate for safety
            unique_samples = []
            for v in sample_values:
                s = str(v)
                if s not in unique_samples:
                    unique_samples.append(s)
                if len(unique_samples) >= 15:
                    break
            preview_items = [s.replace("\n", " ")[:80] for s in unique_samples]
            sample_preview = ", ".join(preview_items)

        prompt = f"""
You are a meticulous data analyst. Your task is to extract structured metadata from a column's `Value Description` into a strict JSON object based on the rules below.

**Context**
- Table: `{table_name}`
- Column: `{column_name}`
- Column Sample Values (from actual data, may be truncated and not exhaustive):
  [{sample_preview if sample_preview else "No reliable sample values available"}]
- Value Description to Analyze:
---
{value_description.strip()}
---

**Core Instructions & Decision Logic**

1.  **Understand the intent of the text.**
    - Decide whether the text is describing the **set of possible/coded values for this column**, or just giving examples / general explanation.

2.  **`encoding_mapping` (be liberal; we will post-validate later).**
    - Use `encoding_mapping` whenever the text appears to describe **discrete codes or possible values** of the column and their meanings (e.g. `0 = No`, `1 = Yes`, `F = Exclusively Virtual`).
    - Strong indicators that you should output `encoding_mapping`:
        - Phrases like: "Values are:", "Values are as follows:", "The field is coded as:", "The field is coded as follows:", "Can be one of:", "Codes are:".
        - Repeated patterns of the form `X = Y`, `X: Y`, or `X - Y` where `X` is a short token (single character, short code, small integer) and `Y` is a natural language explanation.
    - **Use the Sample Values above as an additional signal:**
        - If some of the sample values (e.g. `F`, `V`, `C`, `N`, `P`) also appear in the value description as the left-hand side of these code patterns, you should strongly prefer to treat them as an `encoding_mapping`.
    - You do **NOT** need to be 100% certain that the list is absolutely exhaustive.
        - It is acceptable to output a **best-effort** `encoding_mapping` based on what the text clearly presents as codes.
        - A downstream system will validate and filter the mapping against real data samples, so err slightly on the side of **including** plausible codes rather than dropping them.
    - Only set `"encoding_mapping": null` when the text clearly does **not** define a value/code list (for example, pure free-text explanation or a single concrete example instance).

3.  **`semantic_tags`.**
    - Use this field for additional semantic information that is **not** a direct code/value mapping:
        - Illustrative examples, scenarios, or typical cases (often starting with "For example:", "e.g.", "such as:").
        - Business rules, constraints, warnings, notes, or other contextual information.
    - For such content, create objects like:
        - `{{ "type": "COMMONSENSE", "content": "...", "source": "bird_value_description" }}` for general commonsense / example descriptions.
        - `{{ "type": "BUSINESS_RULE", "content": "...", "source": "bird_value_description" }}` for explicit rules or constraints.

4.  **`description` & `data_format`.**
    - Use `description` for a concise explanation of what the column represents, if present.
    - Use `data_format` for clear data type or pattern information (e.g. "single digit integer", "YYYY-MM-DD date", "percentage between 0 and 100").

5.  **`term_definitions`.**
    - This field is ONLY for **business/domain terms or derived metrics**, such as:
        - Rates or ratios: "eligible free rate", "excellence rate".
        - Business concepts: "valid charter number", "exclusively virtual" (when defined as a concept, not just as a single code value).
    - For each such term, add an object:
        - `term_name`: a concise name of the term (e.g., "eligible free rate for K-12").
        - `definition`: a short natural language explanation of what this term means.
        - `formula`: if there is a clear formula (e.g., "eligible free rate = Free Meal Count / Enrollment"), put the full formula string here; otherwise null.
        - `example_usage`: optional; for value descriptions it can usually be null.
    - IMPORTANT:
        - Do **not** put simple category/code mappings (like "0: N; 1: Y", "F = Exclusively Virtual") into `term_definitions`.
        - `term_definitions` is for **concepts/metrics**, not for raw code values themselves.

6.  **JSON output requirements.**
    - Always return a single JSON object with the keys:
        - `description`
        - `data_format`
        - `encoding_mapping`
        - `semantic_tags`
        - `term_definitions`
    - If a field is not applicable or there is no reliable information for it, set it explicitly to `null`.

**Example 1: Input has an explicit code mapping.**
Value Description: "The value is a single digit integer. Possible values are 1 (Low), 2 (Medium), 3 (High), and 4 (Urgent). Important: any ticket marked as 'Urgent' must be assigned to a senior support agent."
Output:
```json
{{
    "description": null,
    "data_format": "single digit integer",
    "encoding_mapping": {{
        "1": "Low",
        "2": "Medium",
        "3": "High",
        "4": "Urgent"
    }},
    "semantic_tags": [
        {{
            "type": "BUSINESS_RULE",
            "content": "Important: any ticket marked as 'Urgent' must be assigned to a senior support agent.",
            "source": "bird_value_description"
        }}
    ],
    "term_definitions": null
}}
```

**Example 2: Input has a list of possible values (not an example instance).**
Value Description: "Values are as follows: · Not in CS (California School) funding model · Locally funded · Directly funded"
Output:
```json
{{
    "description": null,
    "data_format": null,
    "encoding_mapping": {{
        "Not in CS (California School) funding model": "Not in CS (California School) funding model",
        "Locally funded": "Locally funded",
        "Directly funded": "Directly funded"
    }},
    "semantic_tags": null,
    "term_definitions": null
}}
```

Your Task: Now, analyze the provided Value Description (and the sample values) and generate the JSON object.
"""
        print(f'Value Description: {value_description}')

        response_text = self.llm_analyzer._call_llm(prompt)
        print('-------------Response for extracting from value description------------')
        print(response_text)
        
        try:
            text = response_text.strip()
            if text.startswith('```json'):
                start = text.find('{')
                end = text.rfind('}') + 1
                text = text[start:end]
            data = json.loads(text)
            
            # Normalize fields
            result: dict = {}
            desc = data.get('description')
            if isinstance(desc, str) and desc.strip():
                result['description'] = desc.strip()
            
            fmt = data.get('data_format')
            if isinstance(fmt, str) and fmt.strip():
                result['data_format'] = fmt.strip()
            
            mapping = data.get('encoding_mapping')
            if isinstance(mapping, dict) and mapping:
                # Filter encoding_mapping to ensure keys are realistic
                filtered_mapping = self._filter_encoding_mapping(
                    mapping, value_description, sample_values
                )
                if filtered_mapping:
                    result['encoding_mapping'] = filtered_mapping
            
            tags = data.get('semantic_tags')
            # Handle semantic_tags: can be a list, a dict, or null
            if tags:
                if isinstance(tags, list):
                    if tags:  # Only add if list is not empty
                        result['semantic_tags'] = tags
                elif isinstance(tags, dict):
                    # Single tag object, convert to list
                    tag_type = tags.get('type') or 'UNKNOWN'
                    content = tags.get('content')
                    if content:
                        result['semantic_tags'] = [{
                            'type': str(tag_type),
                            'content': str(content),
                            'source': 'bird_value_description'
                        }]

            # Pass through term_definitions from LLM
            term_defs = data.get('term_definitions')
            if isinstance(term_defs, list):
                result['term_definitions'] = term_defs
            
            # Fallback: If all extracted fields are empty but value_description exists,
            # store the value_description as a semantic_tag to preserve the information
            has_meaningful_fields = (
                result.get('description') or
                result.get('data_format') or
                result.get('encoding_mapping') or
                result.get('semantic_tags') or
                result.get('term_definitions')
            )
            
            if not has_meaningful_fields and value_description and value_description.strip():
                result['semantic_tags'] = [{
                    'type': 'COMMONSENSE',
                    'content': value_description.strip(),
                    'source': 'bird_value_description'
                }]
                logger.debug(
                    f"All LLM fields empty for {table_name}.{column_name}, "
                    "storing value_description as semantic_tag"
                )
            
            return result
        except Exception as e:
            logger.debug(
                f"Failed to parse LLM response for {table_name}.{column_name}: {e}; "
                f"response={response_text[:200]}"
            )
            return {}

    def _extract_terms_from_evidence(self, evidence_text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to extract business term definitions from dev.json evidence text.
        
        Returns a list of dicts with keys:
        - term_name (str)
        - definition (str)
        - formula (optional str or null)
        """
        if not evidence_text or not str(evidence_text).strip():
            return []

        prompt = f"""
You are an expert data analyst. Your task is to read a short evidence text from a NL2SQL dataset
and extract ONLY business/domain terms that are being **defined or explained**.

The evidence text may contain formulas or explanations like:
- "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`"
- "Excellence rate = NumGE1500 / NumTstTakr"
- "Exclusively virtual refers to Virtual = 'F'"
- "Valid charter number means the number is not null"

Your job:
1. Identify each business/domain term that is being **defined**.
2. For each such term, create an object with:
   - "term_name": concise name of the term.
   - "definition": a short natural language explanation of what the term means.
   - "formula": if there is an explicit formula (e.g., "Excellence rate = NumGE1500 / NumTstTakr"),
                put the full formula string here; otherwise null.
3. IMPORTANT: Do NOT treat simple category values or codes as terms. For example, do NOT output terms for:
   - "0: N; 1: Y"
   - Lists of possible enum values (like different school types).
4. If there is no clear term definition, return an empty list [].

Return the results as the following JSON format. Example:
```json
[
  {{
    "term_name": "excellence rate",
    "definition": "The proportion of SAT test takers whose score is at least 1500.",
    "formula": "Excellence rate = NumGE1500 / NumTstTakr"
  }}
]
```

Evidence:
---
{evidence_text.strip()}
---
"""

        response_text = self.llm_analyzer._call_llm(prompt)
        print('-------------Response for extracting terms from evidence------------')
        print(response_text)

        if not response_text or not response_text.strip():
            logger.warning("Empty response from LLM for evidence term extraction")
            return []

        try:
            text = response_text.strip()
            
            # Try to extract JSON array from markdown code blocks
            if '```json' in text or '```' in text:
                # Find the JSON array between ``` markers
                json_start = text.find('[')
                json_end = text.rfind(']')
                
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    text = text[json_start:json_end + 1]
                else:
                    # Try to find JSON object and convert to array
                    obj_start = text.find('{')
                    obj_end = text.rfind('}')
                    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                        obj_text = text[obj_start:obj_end + 1]
                        try:
                            obj_data = json.loads(obj_text)
                            # If it's a single object, wrap it in an array
                            if isinstance(obj_data, dict):
                                text = json.dumps([obj_data])
                            else:
                                text = obj_text
                        except:
                            pass

            # Try to parse as JSON
            data = json.loads(text)
            print(f"Parsed term definitions: {data}")
            
            if isinstance(data, list):
                # Validate each term definition has required fields
                valid_terms = []
                for term in data:
                    if isinstance(term, dict) and term.get('term_name') and term.get('definition'):
                        valid_terms.append(term)
                    else:
                        logger.warning(f"Invalid term definition skipped: {term}")
                
                if valid_terms:
                    logger.info(f"Extracted {len(valid_terms)} valid term definitions from evidence")
                return valid_terms
            elif isinstance(data, dict):
                # Single term definition, wrap in array
                if data.get('term_name') and data.get('definition'):
                    logger.info(f"Extracted 1 term definition from evidence (single object)")
                    return [data]
                else:
                    logger.warning(f"Invalid term definition format (missing term_name or definition): {data}")
                    return []
            else:
                logger.warning(f"Unexpected data type from LLM response: {type(data)}")
                return []
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse LLM response as JSON for evidence terms: {e}\n"
                f"Response text (first 500 chars): {response_text[:500]}"
            )
            return []
        except Exception as e:
            logger.warning(
                f"Unexpected error parsing LLM response for evidence terms: {e}\n"
                f"Response text (first 500 chars): {response_text[:500]}",
                exc_info=True
            )
            return []

    def _import_terms_from_dev(self, actual_database_id: str, bird_database_id: str) -> None:
        """
        Import term definitions from BIRD dev.json evidence fields.
        
        - Filters entries by db_id == bird_database_id
        - Extracts business terms from the 'evidence' field using LLM
        - Stores them as TermDefinition in semantic memory
        """
        dev_file = self.bird_data_dir / "dev" / "dev.json"
        if not dev_file.exists():
            logger.info(f"dev.json not found at {dev_file}, skipping dev-based term import")
            return

        try:
            with open(dev_file, "r", encoding="utf-8") as f:
                dev_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load dev.json from {dev_file}: {e}")
            return

        if not isinstance(dev_data, list):
            logger.warning(f"Invalid dev.json format at {dev_file}: expected list, got {type(dev_data)}")
            return

        imported_terms = 0
        for item in dev_data:
            try:
                if not isinstance(item, dict):
                    continue
                if item.get("db_id") != bird_database_id:
                    continue

                evidence = item.get("evidence") or ""
                if not evidence.strip():
                    continue

                question = item.get("question") or ""
                question_id = item.get("question_id")

                term_defs = self._extract_terms_from_evidence(evidence)
                if not term_defs:
                    continue

                for td in term_defs:
                    if self._add_term_definition_from_dev(
                        actual_database_id, bird_database_id, td, question, question_id, evidence
                    ):
                        imported_terms += 1
                    else:
                        logger.debug(f"Failed to add term definition: {td.get('term_name', 'unknown')}")
            except Exception as e:
                logger.warning(f"Failed processing dev.json item (qid={item.get('question_id', 'unknown')}): {e}", exc_info=True)

        if imported_terms:
            logger.info(
                f"Imported {imported_terms} term definitions from dev.json "
                f"for database {actual_database_id}"
            )
    
    def _add_term_definition(
        self,
        database_id: str,
        table_name: str,
        column_name: str,
        term_dict: Dict[str, Any]
    ) -> bool:
        """Add a term definition from value_description extraction."""
        try:
            term_name = (term_dict.get('term_name') or '').strip()
            definition = (term_dict.get('definition') or '').strip()
            formula = (term_dict.get('formula') or None) or None
            example_usage = (term_dict.get('example_usage') or None) or None

            if not term_name:
                logger.debug(f"Skipping term definition: missing term_name in {term_dict}")
                return False

            # definition is required in TermDefinition, ensure it's non-empty
            if not definition:
                # Fallback: if formula exists, use it as a minimal definition
                if formula:
                    definition = f"Defined as: {formula}"
                else:
                    logger.debug(f"Skipping term definition '{term_name}': missing definition and formula")
                    return False

            # Check if database is bound before adding term definition
            if not self.semantic_store.current_database_id:
                logger.warning(
                    f"Cannot add term definition '{term_name}': database not bound. "
                    f"Current database_id should be '{database_id}' but semantic_store.current_database_id is {self.semantic_store.current_database_id}"
                )
                # Try to bind the database
                self.semantic_store.bind_database(database_id)
                logger.info(f"Bound database to {database_id} before adding term definition")

            context = f"BIRD value_description: table={table_name}, column={column_name}"

            term_def = TermDefinition(
                database_id=database_id,
                term_name=term_name,
                definition=definition,
                formula=formula,
                example_usage=example_usage,
                context=context,
            )
            # Treat as llm_analysis for source priority
            self.semantic_store.add_term_definition(term_def, source='llm_analysis')
            logger.info(f"Successfully added term definition '{term_name}' from value_description for {table_name}.{column_name}")
            return True
        except Exception as e:
            logger.warning(
                f"Failed to add term definition from value_description "
                f"for {table_name}.{column_name}, term={term_dict.get('term_name', 'unknown')}: {e}",
                exc_info=True
            )
            return False
    
    def _add_term_definition_from_dev(
        self,
        actual_database_id: str,
        bird_database_id: str,
        term_dict: Dict[str, Any],
        question: str,
        question_id: Any,
        evidence: str
    ) -> bool:
        """Add a term definition from dev.json evidence."""
        try:
            term_name = (term_dict.get("term_name") or "").strip()
            definition = (term_dict.get("definition") or "").strip()
            formula = (term_dict.get("formula") or None) or None

            if not term_name:
                return False

            if not definition:
                if formula:
                    definition = f"Defined as: {formula}"
                else:
                    return False

            # Check if database is bound before adding term definition
            if not self.semantic_store.current_database_id:
                logger.warning(
                    f"Cannot add term definition '{term_name}': database not bound. "
                    f"Current database_id should be '{actual_database_id}' but semantic_store.current_database_id is {self.semantic_store.current_database_id}"
                )
                # Try to bind the database
                self.semantic_store.bind_database(actual_database_id)
                logger.info(f"Bound database to {actual_database_id} before adding term definition")

            # context & example_usage: dev.json question
            context_parts = [f"BIRD dev evidence for db_id={bird_database_id}"]
            if question_id is not None:
                context_parts.append(f"question_id={question_id}")
            if evidence:
                context_parts.append(f"evidence={evidence.strip()}")
            context = " | ".join(context_parts)

            example_usage = question if question else None

            term_def = TermDefinition(
                database_id=actual_database_id,
                term_name=term_name,
                definition=definition,
                formula=formula,
                example_usage=example_usage,
                context=context,
            )
            self.semantic_store.add_term_definition(term_def, source='llm_analysis')
            logger.info(f"Successfully added term definition '{term_name}' from dev.json evidence (qid={question_id})")
            return True
        except Exception as e:
            logger.warning(
                f"Failed to add term definition from dev.json evidence "
                f"(db={bird_database_id}, qid={question_id}, term={term_dict.get('term_name', 'unknown')}): {e}",
                exc_info=True
            )
            return False
