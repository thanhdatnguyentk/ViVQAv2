import json

path = 'data/vivqa_v2/vivqa_v2_dev.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    if 'annotations' in data and len(data['annotations']) > 0:
        print(data['annotations'][0])
    else:
        print("No annotations found")
