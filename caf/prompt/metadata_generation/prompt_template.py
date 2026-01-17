"""
Metadata Generation Prompt Templates

Prompt templates used for generating database, table, and column metadata.
These templates use Python string formatting with placeholders.
"""

# Database Analysis Prompt Template
DATABASE_ANALYSIS_PROMPT = """Analyze the database structure below. Provide a concise description and a business domain.

Database: {DATABASE_ID}
Tables:
{TABLE_INFO}

Respond in a valid JSON format with keys "description" and "domain".
- "description": A 1-2 sentence summary of the database's business purpose.
- "domain": The most specific business domain (e.g., "E-commerce", "Healthcare").

JSON Analysis:"""


# Table Analysis Prompt Template
TABLE_ANALYSIS_PROMPT = """Analyze the following tables from a database. {DB_CONTEXT_LINE}
For each table, provide a clear 1-2 sentence business description of its purpose.

Tables to analyze:
---
{TABLE_PROMPTS}
---

Respond with a single JSON object where keys are the table names and values are their descriptions.
Example: {{"table_one": "Describes customers...", "table_two": "Tracks product inventory..."}}

JSON Output:"""


# Column Analysis Prompt Template (with sample data)
COLUMN_ANALYSIS_PROMPT = """**Role:** Data Analyst

**Task:** Analyze table columns based on the requirements below.

**Table Context:**
**Table Name:** {TABLE_NAME}
**Table Description:** {TABLE_DESCRIPTION}

**Data:**

{TABLE_MD}

**Columns to Analyze:**
{COLUMNS_TO_ANALYZE}
{COLUMN_METADATA}
**CRITICAL:** You must ONLY analyze the columns listed in "Columns to Analyze" above. Do NOT analyze any other columns, even if they appear in the data table (including primary key columns that are not in the list).

**Requirements:**

{INSTRUCTIONS}

**Important:** For column descriptions, provide a concise, clear, one-sentence summary. Keep it brief and focused on the essential meaning.

**Output Format:**

Return a JSON object with this exact schema for each column:

{OUTPUT_SCHEMA}

**Example:**
```json
{{
  "column_one": {{
    "description": "...",
    "pattern": {{
      "template": "...",
      "regex": "...",
      "fmt_desc": "..."
    }}
  }},
  "column_two": {{
    "description": "...",
    "pattern": {{
      "template": "...",
      "regex": "...",
      "fmt_desc": "..."
    }}
  }}
}}
```

**JSON Output:**

"""

# Column Analysis Prompt Template (with top_k_values)
COLUMN_ANALYSIS_WITH_TOP_K_PROMPT = """**Role:** Data Analyst

**Task:** Analyze a database column based on its top values below.

**Table:** {TABLE_NAME}
**Column:** {COLUMN_NAME}
**Whole Column Name:** {WHOLE_COLUMN_NAME}
**Table Description:** {TABLE_DESCRIPTION}

**Top Values in Column:**
{TOP_VALUES_TEXT}

**Requirements:**

{INSTRUCTIONS}

**Important:** For column descriptions, provide a concise, clear, one-sentence summary. Keep it brief and focused on the essential meaning.

**Output Format:**

Return a JSON object in the format as the following:

{OUTPUT_SCHEMA}

**Example:**
```json
{{
  "description": "...",
  "pattern": {{
    "template": "...",
    "regex": "...",
    "fmt_desc": "..."
  }}
}}
```

**JSON Output:**

"""


# Query Generation Prompt Template
QUERY_GENERATION_PROMPT = """You are generating search phrases (keywords/phrases) that users might type when searching for a specific database column.

**Target Column Information:**
- Column Name: {COLUMN_NAME}
- Column Description: {DESCRIPTION}

**CRITICAL REQUIREMENT:** 
Generate SEARCH PHRASES (keywords/phrases), NOT complete questions. These should be short phrases that users would type in a search box. Focus on the COLUMN itself, not the table or database context.

**Task:** Generate exactly 3 short search phrases (1-4 words each) that users might type when looking for THIS COLUMN's data:

1. **Exact Match #1**: A direct phrase about THIS COLUMN (e.g., for a "dname" column with description "district segment": "district" or "district segment")
2. **Exact Match #2**: Another direct phrase about THIS COLUMN (e.g., for a "dname" column: "district name" or "school district")
3. **Concept-Based**: An abstract or conceptual phrase using layman terms that relates to THIS COLUMN's semantic meaning (e.g., for a "city" column: "location" or "where")

**Important constraints:**
- Generate PHRASES/KEYWORDS, NOT complete questions (e.g., "district" NOT "What is the district?")
- Focus on the COLUMN itself, not the table name or database context
- Do NOT mention the exact column name if it is a technical code (like "cds", "id", "dname", etc.)
- Use natural, search-friendly phrases
- Keep phrases short (1-4 words typically)
- Each phrase should help users find THIS COLUMN's data specifically
- For Exact Match phrases, use words from the description or related terms (e.g., for "dname" with description "district segment", use "district", "district segment", "district name")

**Output format (JSON array only, no other text):**
["phrase1", "phrase2", "phrase3"]

Where phrase1 and phrase2 are Exact Match, and phrase3 is Concept-Based."""


# Table Semantics Analysis Prompt Template
TABLE_SEMANTICS_PROMPT = """# Role
You are an expert Data Architect specializing in Reverse Engineering and NL2SQL metadata optimization.

# Task
Analyze the table "{TABLE_NAME}" from a {DB_DOMAIN} database. You must perform a comprehensive analysis considering ALL aspects of the table (structure, data patterns, topology, unique columns) to determine its role and provide detailed semantic descriptions.

# Database Context
- Domain: {DB_DOMAIN}
- Description: {DB_DESCRIPTION}

# Database Topology Summary
{TOPOLOGY_SUMMARY}

# Table Details
{TABLE_DETAIL}{UNIQUE_COLUMNS_NOTE}

# Hard Topology Classification Rules (follow these before any other guidance)
- If the table has **only In-degree (>0) and Out-degree == 0**, classify as **Dimension** (it is referenced by others).
- If the table has **only Out-degree (>0) and In-degree == 0**, classify as **Fact** (it references others).
- If both In-degree and Out-degree are zero, consider Lookup/isolated patterns together with size.
- If both In-degree and Out-degree are present (>0), use all provided information to decide (topology + structure + unique columns).

# Guidelines for Table Classification

## 1. Fact Table (The "Central Event")
- **Topology Pattern:** **High Out-degree, Low or Zero In-degree.** It heavily references other tables (FKs) to provide context but is rarely referenced by others. 
- **CRITICAL:** If a table has multiple Foreign Keys (Out-degree) but no other tables point to it (Zero In-degree), it is almost certainly a Fact Table.
- **Content:** Each row represents a business event/transaction (orders, logs, measurements).
- **Unique Columns:** Define the granularity of the event (the "What", "When", and "Where").
- **Example:** order_items, payments, user_clicks.

## 2. Dimension Table (The "Contextual Entity")
- **Topology Pattern:** **High In-degree, Low Out-degree.** It is frequently referenced by Fact tables but rarely points to others (except in Snowflake schemas).
- **Content:** Each row describes a unique business entity (users, products, stores, categories).
- **Focus:** Stores descriptive attributes (names, types, addresses).
- **Example:** customers, products, date_dimension.

## 3. Bridge/Junction Table (The "N:M Linker")
- **Topology Pattern:** **Moderate Out-degree (usually 2+), Low In-degree.** - **Content:** Resolves many-to-many relationships. Usually consists of composite Foreign Keys.
- **Example:** student_courses, user_roles.

## 4. Lookup/Reference Table (The "Static Code")
- **Topology Pattern:** **In-degree only or Isolated.** - **Content:** Small, static tables translating codes to labels (status_id -> status_name).
- **Example:** order_status, country_codes.


# Analysis Process

**IMPORTANT: You must perform a holistic analysis using these steps:**

1.  **Topology Check (Priority):**
    - Calculate the **In-degree** (how many tables reference this one) and **Out-degree** (how many tables this one references) based on the `topology_summary`.
    - Apply the rule: High Out + Zero In = Fact; High In + Low Out = Dimension.
    - If In-degree > 0 and Out-degree == 0, you MUST return **Dimension**.
    - If Out-degree > 0 and In-degree == 0, you MUST return **Fact**.
    - When both degrees exist (>0), decide using combined topology, structure, and unique columns.

2.  **Structural Analysis:**
    - Examine Unique Columns and Key constraints. For Fact tables, how do these define the event granularity?
    - For Dimension tables, what entity attributes are provided?

3.  **Semantic Synthesis:**
    - Combine Topology and Structure to determine the role.
    - Draft the `row_definition` and `description` ensuring they align with the identified role.

# Output Format
Respond with a JSON object:

{{
  "table_role": "<Fact|Dimension|Bridge|Lookup>",
  "row_definition": "Each row represents...",
  "description": "<2-3 sentences explaining the table's purpose and how its relationships (topology) support its role.>"
}}

# Requirements

1. **table_role** MUST be exactly one of: "Fact", "Dimension", "Bridge", "Lookup" (case-sensitive)
   - Base your classification on comprehensive analysis of ALL provided information
   - You MUST comply with the hard topology rules above before any other consideration

2. **row_definition** MUST follow these formulas based on the table role:

   **IF ROLE IS [Fact]:**
   - **Formula**: "Each row represents a [Measurement/Event] of [Anchor Entity] at [Temporal/Contextual Grain]."
   - **CRITICAL Requirement**: 
     * You MUST explicitly identify and mention the Unique Columns by name
     * You MUST explain what these Unique Columns measure/record
     * You MUST specify the granularity (temporal, contextual, or both)
   - **Example**: "Each row represents the annual SAT performance scores for a specific school, capturing the average scores in Reading (AvgScrReading), Math (AvgScrMath), and Writing (AvgScrWriting) for a school identified by its CDSCode in a given academic year."

   **IF ROLE IS [Dimension]:**
   - **Formula**: "Each row represents a unique [Anchor Entity] and provides detailed [Category of Information]."
   - **Requirement**: Focus on what descriptive value this table adds to the entity.
   - **Example**: "Each row represents a unique California school and provides its location, district information, and enrollment details."

3. **description** must be concise yet comprehensive (2-3 sentences, avoid redundancy):
   - **For Fact Tables**: 
     * Briefly explain what measurements/events are captured and how Unique Columns define them
     * Mention business context, granularity, and relationship to dimension tables (if applicable)
     * Describe how this table relates to dimension tables (if applicable)
     * Mention the granularity and scope of the measurements
   - **For Dimension Tables**: 
     * Briefly explain what entity attributes are stored and their business significance
     * Mention how this table supports fact tables or enriches the data model
   - **For Bridge/Lookup Tables**: Briefly explain the relationship or reference purpose
   - The description should be informative but concise - avoid repeating information already in row_definition

# Example Outputs

**Fact Table Example:**
{{
  "table_role": "Fact",
  "row_definition": "Each row represents the annual SAT performance scores for a specific school, capturing the average scores in Reading (AvgScrReading), Math (AvgScrMath), and Writing (AvgScrWriting) for a school identified by its CDSCode in a given academic year.",
  "description": "This Fact table captures annual standardized test performance measurements at the school level, with unique columns AvgScrMath, AvgScrReading, and AvgScrWriting defining the core metrics (average scores in each SAT section). The table links to dimension tables through CDSCode and supports educational assessment and policy decisions by tracking academic performance trends over time."
}}

**Dimension Table Example:**
{{
  "table_role": "Dimension",
  "row_definition": "Each row represents a unique California school and provides its location, district information, and enrollment details.",
  "description": "This Dimension table stores descriptive attributes about California schools, including geographic information, administrative details, and enrollment statistics. It serves as a master reference that provides context for fact tables measuring school performance, enabling analysis and filtering by location, district, or school type."
}}

JSON Output:"""


# Natural Language Description Generation Prompt
NATURAL_DESCRIPTION_PROMPT = """**Role:** Database Documentation Expert

**Task:** Generate two types of natural language descriptions for database columns based on their statistical profiles.

**Table:** {TABLE_NAME}

**Columns to Analyze:**
{COLUMN_SECTIONS}

**Requirements:**

For each column, generate TWO descriptions:

1. **short_description** (for Schema Linking):
   - 1-2 concise sentences
   - Focus on semantic meaning and purpose
   - Identify abbreviations or acronyms (e.g., "CDS" means "County-District-School")
   - Describe basic data format
   - Keep brief for quick schema understanding

2. **long_description** (for SQL Generation):
   - Build upon the short description
   - Include specific value ranges and examples from statistics
   - **IMPORTANT: Provide at least one complete, un-truncated example value** to illustrate the actual data format
   - For complex data types (JSON, structured text, etc.), include a full example showing the structure
   - Mention data constraints and patterns
   - Provide concrete details to guide SQL query generation with correct literals
   - 3-5 sentences with actionable details

**Output Format:**

Return a JSON object where each column name maps to its descriptions:

```json
{{
  "column_name_1": {{
    "short_description": "Brief semantic description...",
    "long_description": "Detailed description with statistics..."
  }},
  "column_name_2": {{
    "short_description": "Brief semantic description...",
    "long_description": "Detailed description with statistics..."
  }}
}}
```

**JSON Output:**
"""


# Relationship Semantics Analysis Prompt Template
RELATIONSHIP_SEMANTICS_PROMPT = """# Role
You are an expert Data Architect specializing in database relationship analysis.

# Task
Analyze the relationship between two tables and determine its business meaning based on the provided information.

# Relationship Information

**Source Table:** {SOURCE_TABLE}
- Table Description: {SOURCE_DESCRIPTION}
- Table Role: {SOURCE_ROLE}
- Join Key Columns: {SOURCE_COLUMNS}
- Sample Table:
{SOURCE_SAMPLE_TABLE_MD}

**Target Table:** {TARGET_TABLE}
- Table Description: {TARGET_DESCRIPTION}
- Table Role: {TARGET_ROLE}
- Join Key Columns: {TARGET_COLUMNS}
- Sample Table:
{TARGET_SAMPLE_TABLE_MD}

**Observed Cardinality:** {CARDINALITY}
*(Note: Use cardinality strictly as a hint. Focus on the logical intent of the join.)*

# Analysis Guidelines (Mental Sandbox)

Use the following **common patterns** to help formulate your description, but **do not restrict yourself** to them if the relationship is unique:

* **Extension/Detailing**: Target adds more columns/attributes to the Source. (Verbs: *extends, details, enriches*)
* **Hierarchy/Grouping**: Target is a parent category or grouping entity. (Verbs: *groups, categorizes, contains*)
* **Transaction/Logging**: Target records events occurring to the Source. (Verbs: *logs, tracks history of, records metrics for*)
* **Reference/Lookup**: Target defines codes used in the Source. (Verbs: *defines, translates code for*)
* **Association/Bridge**: Target connects the Source to another entity (e.g., in Many-to-Many). (Verbs: *links, associates, connects*)
* **Filtering/Scoping**: Target restricts the Source data to a specific subset. (Verbs: *filters, restricts scope to*)

# Output Requirements

Respond with a JSON object: `{{ "business_meaning": "..." }}`

The `business_meaning` must:
1.  **Be Direct**: Do NOT include category labels (e.g., "Identifies:", "Type 1:").
2.  **Sentence Structure**: Start with **"{TARGET_TABLE} [Verb] {SOURCE_TABLE}..."** or **"{TARGET_TABLE} serves as..."**.
3.  **Context-Aware**: Mention specific attributes or business concepts found in the descriptions (e.g., "student enrollment," "sales performance").

# Example Outputs

**Example 1:**
{{
  "business_meaning": "User_Profile extends the User table by containing detailed preferences and bio information for each account."
}}

**Example 2:**
{{
  "business_meaning": "Click_Logs tracks the interaction history for each Session, recording timestamps and button clicks."
}}

**Example 3:**
{{
  "business_meaning": "Course_Enrollment links Students to Courses, recording the grade and semester for each registration."
}}


JSON Output:"""