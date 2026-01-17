import pickle
import sys
from pathlib import Path
import json

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
print('Analyzing Tables Field Differences')
print('='*100)
print()

# Find samples with differences and analyze them
diff_samples = []

for i in range(min(len(data1), len(data2))):
    d1 = to_dict(data1[i])
    d2 = to_dict(data2[i])
    
    s1 = d1.get('database_schema_after_schema_linking')
    s2 = d2.get('database_schema_after_schema_linking')
    
    if s1 != s2 and s1 is not None and s2 is not None:
        s1d = to_dict(s1)
        s2d = to_dict(s2)
        
        tables1 = s1d.get('tables', {})
        tables2 = s2d.get('tables', {})
        
        if tables1 != tables2:
            diff_samples.append({
                'index': i,
                'question_id': d1.get('question_id'),
                'question': d1.get('question'),
                'tables1': tables1,
                'tables2': tables2
            })

print(f'Found {len(diff_samples)} samples with different tables')
print()

# Analyze first few differences in detail
for idx, sample in enumerate(diff_samples[:5]):
    print('-'*100)
    print(f"Sample {sample['index']} (Question ID: {sample['question_id']})")
    print(f"Question: {sample['question']}")
    print()
    
    tables1 = sample['tables1']
    tables2 = sample['tables2']
    
    # Compare table names
    table_names1 = set(tables1.keys())
    table_names2 = set(tables2.keys())
    
    print(f"Schema 1 tables: {sorted(table_names1)}")
    print(f"Schema 2 tables: {sorted(table_names2)}")
    
    if table_names1 != table_names2:
        print(f"  Tables only in Schema 1: {sorted(table_names1 - table_names2)}")
        print(f"  Tables only in Schema 2: {sorted(table_names2 - table_names1)}")
    else:
        print("  Both schemas have same tables")
    print()
    
    # Compare columns in common tables
    for table_name in sorted(table_names1 & table_names2):
        table1 = tables1[table_name]
        table2 = tables2[table_name]
        
        cols1 = set(table1.get('columns', {}).keys())
        cols2 = set(table2.get('columns', {}).keys())
        
        if cols1 != cols2:
            print(f"  Table '{table_name}':")
            print(f"    Schema 1 columns ({len(cols1)}): {sorted(cols1)}")
            print(f"    Schema 2 columns ({len(cols2)}): {sorted(cols2)}")
            
            only_in_1 = cols1 - cols2
            only_in_2 = cols2 - cols1
            
            if only_in_1:
                print(f"    Columns ONLY in Schema 1: {sorted(only_in_1)}")
            if only_in_2:
                print(f"    Columns ONLY in Schema 2: {sorted(only_in_2)}")
            print()

print()
print('='*100)
print('STATISTICAL ANALYSIS')
print('='*100)

# Collect all column differences
all_column_diffs = {}

for sample in diff_samples:
    tables1 = sample['tables1']
    tables2 = sample['tables2']
    
    for table_name in set(tables1.keys()) & set(tables2.keys()):
        table1 = tables1[table_name]
        table2 = tables2[table_name]
        
        cols1 = set(table1.get('columns', {}).keys())
        cols2 = set(table2.get('columns', {}).keys())
        
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1
        
        for col in only_in_1:
            key = f"{table_name}.{col} (only in schema_linking)"
            all_column_diffs[key] = all_column_diffs.get(key, 0) + 1
        
        for col in only_in_2:
            key = f"{table_name}.{col} (only in augmented_data_retrieval)"
            all_column_diffs[key] = all_column_diffs.get(key, 0) + 1

print(f'\nColumn differences across all samples:')
for col_diff, count in sorted(all_column_diffs.items(), key=lambda x: -x[1])[:20]:
    print(f"  {col_diff}: {count} times")

print()
print(f'Total unique column differences: {len(all_column_diffs)}')
