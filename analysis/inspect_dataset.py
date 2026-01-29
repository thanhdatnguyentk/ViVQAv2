import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
file_path = os.path.join(ROOT_DIR, 'data', 'ds102', 'test.json')

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'annotations' in data:
        print(f"Num annotations: {len(data['annotations'])}")
        print("First annotation:", json.dumps(data['annotations'][0], indent=2, ensure_ascii=False))
    elif isinstance(data, list):
         print("First item:", json.dumps(data[0], indent=2, ensure_ascii=False))
             
except Exception as e:
    print(f"Error reading file: {e}")
