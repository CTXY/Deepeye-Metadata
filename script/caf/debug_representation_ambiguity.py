"""Debug script to check why no 1-to-1 pairs are detected"""

import sqlite3
import json
from pathlib import Path
from caf.config import CAFConfig
from caf.memory.stores.semantic import SemanticMemoryStore

# Load config
config = CAFConfig.from_file("config/caf_config.yaml")
semantic_store = SemanticMemoryStore(config.memory)
semantic_store.bind_database("california_schools")

# Get columns
column_df = semantic_store.dataframes.get("column")
print(f"Total columns: {len(column_df)}")

# Load database mapping
mapping_path = Path('data/database_mapping.json')
with open(mapping_path, 'r') as f:
    mapping = json.load(f)

# Find california_schools database
db_path = None
for path, db_id in mapping.items():
    if db_id == 'california_schools':
        db_path = path
        break

if not db_path:
    print("Database not found!")
    exit(1)

print(f"\nDatabase path: {db_path}")

# Connect and check
conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
conn.row_factory = sqlite3.Row

# Get tables
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"\nTables: {tables}")

# Check columns in each table
for table in tables[:3]:  # Check first 3 tables
    print(f"\n=== Table: {table} ===")
    cursor = conn.execute(f'PRAGMA table_info(`{table}`)')
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    print(f"Columns ({len(col_names)}): {col_names[:10]}...")
    
    # Find ID and Name columns
    id_cols = [c for c in col_names if 'id' in c.lower()]
    name_cols = [c for c in col_names if 'name' in c.lower()]
    code_cols = [c for c in col_names if 'code' in c.lower()]
    
    print(f"ID columns: {id_cols}")
    print(f"Name columns: {name_cols}")
    print(f"Code columns: {code_cols}")
    
    # Test 1-to-1 check on ID/Name pairs
    if id_cols and name_cols:
        for id_col in id_cols[:2]:  # Test first 2 ID columns
            for name_col in name_cols[:2]:  # Test first 2 Name columns
                try:
                    query = f"""
                        SELECT
                            COUNT(DISTINCT `{id_col}`) as n_id,
                            COUNT(DISTINCT `{name_col}`) as n_name,
                            COUNT(DISTINCT (`{id_col}`, `{name_col}`)) as n_pair
                        FROM `{table}`
                        WHERE `{id_col}` IS NOT NULL AND `{name_col}` IS NOT NULL
                    """
                    cursor = conn.execute(query)
                    row = cursor.fetchone()
                    
                    n_id = row["n_id"]
                    n_name = row["n_name"]
                    n_pair = row["n_pair"]
                    
                    print(f"\n  Testing: {id_col} vs {name_col}")
                    print(f"    N_ID: {n_id}, N_NAME: {n_name}, N_PAIR: {n_pair}")
                    
                    if n_id > 0 and n_name > 0 and n_pair > 0:
                        ratio_id = n_pair / n_id
                        ratio_name = n_pair / n_name
                        print(f"    Ratio_ID: {ratio_id:.4f}, Ratio_NAME: {ratio_name:.4f}")
                        
                        threshold = 0.02
                        is_1to1 = abs(1.0 - ratio_id) <= threshold and abs(1.0 - ratio_name) <= threshold
                        print(f"    Is 1-to-1 (threshold={threshold}): {is_1to1}")
                        
                        if not is_1to1:
                            print(f"    Deviation: ID={abs(1.0 - ratio_id):.4f}, NAME={abs(1.0 - ratio_name):.4f}")
                except Exception as e:
                    print(f"    Error: {e}")

# Also check what columns the miner is actually checking
print("\n\n=== Columns from semantic store ===")
for _, row in column_df.head(20).iterrows():
    print(f"{row['table_name']}.{row['column_name']}")

conn.close()


















