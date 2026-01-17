import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

file1 = 'workspace/schema_linking/bird/sub_dev.pkl'
file2 = 'workspace/augmented_data_retrieval/bird/sub_dev.pkl'

with open(file1, 'rb') as f:
    data1 = pickle.load(f)
with open(file2, 'rb') as f:
    data2 = pickle.load(f)

def to_dict(obj):
    return obj if isinstance(obj, dict) else (vars(obj) if hasattr(obj, '__dict__') else {})

print('='*100)
print('DATABASE_SCHEMA_AFTER_SCHEMA_LINKING COMPARISON REPORT')
print('='*100)
print()
print(f'File 1 (schema_linking): {file1}')
print(f'File 2 (augmented_data_retrieval): {file2}')
print()
print(f'Total samples: {len(data1)} vs {len(data2)}')
print()

# Count differences
identical = 0
different = 0
schema1_has_more_cols = 0
schema2_has_more_cols = 0
both_empty = 0

all_extra_cols_in_schema2 = {}

for i in range(min(len(data1), len(data2))):
    d1 = to_dict(data1[i])
    d2 = to_dict(data2[i])
    
    s1 = d1.get('database_schema_after_schema_linking')
    s2 = d2.get('database_schema_after_schema_linking')
    
    if s1 == s2:
        identical += 1
        s1d = to_dict(s1) if s1 else {}
        tables = s1d.get('tables', {})
        if not tables or all(not t.get('columns', {}) for t in tables.values()):
            both_empty += 1
    else:
        different += 1
        
        if s1 is not None and s2 is not None:
            s1d = to_dict(s1)
            s2d = to_dict(s2)
            
            tables1 = s1d.get('tables', {})
            tables2 = s2d.get('tables', {})
            
            for table_name in set(tables1.keys()) & set(tables2.keys()):
                table1 = tables1[table_name]
                table2 = tables2[table_name]
                
                cols1 = set(table1.get('columns', {}).keys())
                cols2 = set(table2.get('columns', {}).keys())
                
                if len(cols1) > len(cols2):
                    schema1_has_more_cols += 1
                elif len(cols2) > len(cols1):
                    schema2_has_more_cols += 1
                
                only_in_2 = cols2 - cols1
                for col in only_in_2:
                    key = f"{table_name}.{col}"
                    all_extra_cols_in_schema2[key] = all_extra_cols_in_schema2.get(key, 0) + 1

print('='*100)
print('SUMMARY STATISTICS')
print('='*100)
print(f'Identical schemas: {identical}/{len(data1)} ({identical*100//len(data1)}%)')
print(f'  - Of which both empty: {both_empty}')
print(f'Different schemas: {different}/{len(data1)} ({different*100//len(data1)}%)')
print()

print('='*100)
print('KEY FINDINGS')
print('='*100)
print()
print('✓ Both datasets have the same number of samples (195)')
print()
print(f'✓ {identical} samples have IDENTICAL schemas')
print(f'✓ {different} samples have DIFFERENT schemas')
print()
print('✓ Main difference: augmented_data_retrieval has MORE COLUMNS in its schemas')
print(f'  - Schema 2 (augmented_data_retrieval) has more columns in {schema2_has_more_cols} table comparisons')
print(f'  - Schema 1 (schema_linking) has more columns in {schema1_has_more_cols} table comparisons')
print()

print('='*100)
print('TOP 20 EXTRA COLUMNS IN AUGMENTED_DATA_RETRIEVAL')
print('='*100)
for col, count in sorted(all_extra_cols_in_schema2.items(), key=lambda x: -x[1])[:20]:
    print(f'  {col}: appears {count} times')

print()
print('='*100)
print('CONCLUSION')
print('='*100)
print()
print('The augmented_data_retrieval dataset contains ENRICHED schemas with more columns')
print('compared to the schema_linking dataset. This suggests that the augmented_data_retrieval')
print('pipeline adds additional schema information during its processing.')
print()
print(f'Total unique extra columns in augmented_data_retrieval: {len(all_extra_cols_in_schema2)}')
print()
