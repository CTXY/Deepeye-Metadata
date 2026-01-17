import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

file1 = 'workspace/schema_linking/bird/sub_dev.pkl'
file2 = 'workspace/augmented_data_retrieval/bird/sub_dev.pkl'

print('='*100)
print('Loading pickle files...')
print(f'File 1: {file1}')
print(f'File 2: {file2}')

with open(file1, 'rb') as f:
    data1 = pickle.load(f)
with open(file2, 'rb') as f:
    data2 = pickle.load(f)

print(f'Loaded: Dataset 1 has {len(data1)} samples, Dataset 2 has {len(data2)} samples')
print('='*100)
print()

# Count differences
diff_count = 0
identical_count = 0

for i in range(min(len(data1), len(data2))):
    d1 = data1[i] if isinstance(data1[i], dict) else vars(data1[i])
    d2 = data2[i] if isinstance(data2[i], dict) else vars(data2[i])
    
    s1 = d1.get('database_schema_after_schema_linking')
    s2 = d2.get('database_schema_after_schema_linking')
    
    if s1 == s2:
        identical_count += 1
    else:
        diff_count += 1

print('SUMMARY:')
print(f'  Identical schemas: {identical_count}/{min(len(data1), len(data2))}')
print(f'  Different schemas: {diff_count}/{min(len(data1), len(data2))}')
print()
print('='*100)
print('DETAILED COMPARISON (First 10 samples)')
print('='*100)

def to_dict(obj):
    return obj if isinstance(obj, dict) else (vars(obj) if hasattr(obj, '__dict__') else {})

for i in range(min(10, len(data1), len(data2))):
    d1 = to_dict(data1[i])
    d2 = to_dict(data2[i])
    
    s1 = d1.get('database_schema_after_schema_linking')
    s2 = d2.get('database_schema_after_schema_linking')
    
    print(f'\nSample {i}:')
    print(f'  Question ID: {d1.get("question_id", "N/A")}')
    print(f'  Database: {d1.get("database_id", "N/A")}')
    print(f'  Question: {d1.get("question", "N/A")}')
    print()
    
    if s1 is None and s2 is None:
        print('  Both schemas are None - IDENTICAL')
    elif s1 is None or s2 is None:
        print(f'  Schema 1 is None: {s1 is None}')
        print(f'  Schema 2 is None: {s2 is None}')
        print('  STATUS: DIFFERENT')
    else:
        s1d = to_dict(s1)
        s2d = to_dict(s2)
        
        tables1 = s1d.get('table_names', [])
        tables2 = s2d.get('table_names', [])
        cols1 = s1d.get('column_names', [])
        cols2 = s2d.get('column_names', [])
        fks1 = s1d.get('foreign_keys', [])
        fks2 = s2d.get('foreign_keys', [])
        pks1 = s1d.get('primary_keys', [])
        pks2 = s2d.get('primary_keys', [])
        
        if s1 == s2:
            print('  STATUS: IDENTICAL')
            print(f'    Tables: {tables1}')
            print(f'    Columns: {len(cols1)} columns')
            print(f'    Foreign keys: {fks1}')
            print(f'    Primary keys: {pks1}')
        else:
            print('  STATUS: DIFFERENT')
            print()
            print('  Schema 1 (schema_linking):')
            print(f'    Tables: {tables1}')
            print(f'    Columns ({len(cols1)}): {cols1}')
            print(f'    Foreign keys: {fks1}')
            print(f'    Primary keys: {pks1}')
            print()
            print('  Schema 2 (augmented_data_retrieval):')
            print(f'    Tables: {tables2}')
            print(f'    Columns ({len(cols2)}): {cols2}')
            print(f'    Foreign keys: {fks2}')
            print(f'    Primary keys: {pks2}')
            print()
            print('  Specific differences:')
            if tables1 != tables2:
                print(f'    - Tables differ')
            if cols1 != cols2:
                print(f'    - Columns differ')
            if fks1 != fks2:
                print(f'    - Foreign keys differ')
            if pks1 != pks2:
                print(f'    - Primary keys differ')
    
    print('-'*100)
