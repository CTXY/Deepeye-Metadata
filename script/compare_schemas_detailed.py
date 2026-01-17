import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

file1 = 'workspace/schema_linking/bird/sub_dev.pkl'
file2 = 'workspace/augmented_data_retrieval/bird/sub_dev.pkl'

print('='*100)
print('Loading pickle files...')

with open(file1, 'rb') as f:
    data1 = pickle.load(f)
with open(file2, 'rb') as f:
    data2 = pickle.load(f)

print(f'Loaded: {len(data1)} vs {len(data2)} samples')
print('='*100)
print()

def to_dict(obj):
    return obj if isinstance(obj, dict) else (vars(obj) if hasattr(obj, '__dict__') else {})

# Find first different sample
print('Finding first sample with different schemas...')
print()

for i in range(min(len(data1), len(data2))):
    d1 = to_dict(data1[i])
    d2 = to_dict(data2[i])
    
    s1 = d1.get('database_schema_after_schema_linking')
    s2 = d2.get('database_schema_after_schema_linking')
    
    if s1 != s2:
        print(f'Found first difference at Sample {i}:')
        print(f'Question ID: {d1.get("question_id")}')
        print(f'Question: {d1.get("question")}')
        print()
        
        print(f'Schema 1 type: {type(s1)}')
        print(f'Schema 2 type: {type(s2)}')
        print()
        
        if s1 is not None:
            s1d = to_dict(s1)
            print('Schema 1 content:')
            print(f'  Type: {type(s1)}')
            print(f'  Dict keys: {list(s1d.keys())}')
            for key, val in s1d.items():
                print(f'    {key}: {val} (type: {type(val)})')
            print()
        else:
            print('Schema 1 is None')
            print()
            
        if s2 is not None:
            s2d = to_dict(s2)
            print('Schema 2 content:')
            print(f'  Type: {type(s2)}')
            print(f'  Dict keys: {list(s2d.keys())}')
            for key, val in s2d.items():
                print(f'    {key}: {val} (type: {type(val)})')
            print()
        else:
            print('Schema 2 is None')
            print()
        
        print('Comparison:')
        print(f'  s1 == s2: {s1 == s2}')
        print(f'  s1 is s2: {s1 is s2}')
        print(f'  id(s1): {id(s1)}')
        print(f'  id(s2): {id(s2)}')
        
        if s1 is not None and s2 is not None:
            s1d = to_dict(s1)
            s2d = to_dict(s2)
            print()
            print('Field-by-field comparison:')
            all_keys = set(s1d.keys()) | set(s2d.keys())
            for key in sorted(all_keys):
                v1 = s1d.get(key)
                v2 = s2d.get(key)
                if v1 != v2:
                    print(f'  {key}: DIFFERENT')
                    print(f'    Schema 1: {v1} (type: {type(v1)})')
                    print(f'    Schema 2: {v2} (type: {type(v2)})')
                else:
                    print(f'  {key}: SAME ({v1})')
        
        print()
        print('='*100)
        break

# Count differences by examining object attributes
print()
print('Analyzing all differences...')
print()

diff_types = {}
for i in range(min(len(data1), len(data2))):
    d1 = to_dict(data1[i])
    d2 = to_dict(data2[i])
    
    s1 = d1.get('database_schema_after_schema_linking')
    s2 = d2.get('database_schema_after_schema_linking')
    
    if s1 != s2:
        if s1 is None and s2 is None:
            diff_type = 'both_none_but_different'
        elif s1 is None:
            diff_type = 's1_none'
        elif s2 is None:
            diff_type = 's2_none'
        else:
            s1d = to_dict(s1)
            s2d = to_dict(s2)
            
            diff_fields = []
            for key in set(s1d.keys()) | set(s2d.keys()):
                if s1d.get(key) != s2d.get(key):
                    diff_fields.append(key)
            
            if not diff_fields:
                diff_type = 'no_field_diff_but_not_equal'
            else:
                diff_type = f'fields_differ: {",".join(sorted(diff_fields))}'
        
        diff_types[diff_type] = diff_types.get(diff_type, 0) + 1

print('Difference types:')
for diff_type, count in sorted(diff_types.items(), key=lambda x: -x[1]):
    print(f'  {diff_type}: {count}')
