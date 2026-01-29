import json

path = 'data/ds102/vivqa_v2_train.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
    anns = data['annotations']
    imgs = data['images']
    
    ann_img_ids = set(a['image_id'] for a in anns)
    img_ids = [i['id'] for i in imgs]
    unique_img_ids = set(img_ids)
    
    print(f"Annotations: {len(anns)}")
    print(f"Unique Image IDs in Annotations: {len(ann_img_ids)}")
    print(f"Images in list: {len(imgs)}")
    print(f"Unique Image IDs in Images list: {len(unique_img_ids)}")
    
    # Check for duplicates in images list
    if len(imgs) != len(unique_img_ids):
        print("WARNING: Duplicate images found in images list!")
        from collections import Counter
        c = Counter(img_ids)
        print(f"Top 5 duplicates: {c.most_common(5)}")
        
    # Check sync
    missing_imgs = ann_img_ids - unique_img_ids
    extra_imgs = unique_img_ids - ann_img_ids
    
    print(f"Missing images (in anns but not in images): {len(missing_imgs)}")
    print(f"Extra images (in images but not in anns): {len(extra_imgs)}")
