import os
import json
import shutil
from tqdm import tqdm

DATA_DIR = "data/vivqa_v2"
OUTPUT_DIR = "data/ds102"

def split_dataset(data_dir, output_dir, max_train=4200, max_dev=500, max_test=500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    files_config = {
        'vivqa_v2_train.json': max_train,
        'vivqa_v2_dev.json': max_dev,
        'vivqa_v2_test.json': max_test
    }

    for file_name, max_count in files_config.items():
        file_path = os.path.join(data_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    new_data = {}
                    if 'annotations' in data:
                        new_data['annotations'] = data['annotations'][:max_count]
                        
                        # Collect image IDs from the selected annotations
                        image_ids = set(ann['image_id'] for ann in new_data['annotations'])
                        
                        if 'images' in data:
                            # Filter images to keep only those referenced by the annotations, removing duplicates
                            unique_images = {}
                            for img in data['images']:
                                if img['id'] in image_ids and img['id'] not in unique_images:
                                    unique_images[img['id']] = img
                            new_data['images'] = list(unique_images.values())
                        else:
                            new_data['images'] = []
                    else:
                        new_data['annotations'] = []
                        new_data['images'] = []
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(new_data, out_f, ensure_ascii=False, indent=4)
                    
                    print(f"Saved {file_name} to {output_dir} with {len(new_data.get('annotations', []))} annotations and {len(new_data.get('images', []))} images.")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")

def move_images(data_dir, output_dir):
    source_images_dir = os.path.join(data_dir, 'images')
    dest_images_dir = os.path.join(output_dir, 'images')
    
    if not os.path.exists(dest_images_dir):
        os.makedirs(dest_images_dir)
        print(f"Created directory: {dest_images_dir}")
        
    files = ['vivqa_v2_train.json', 'vivqa_v2_dev.json', 'vivqa_v2_test.json']
    
    total_copied = 0
    for file_name in files:
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    images = data.get('images', [])
                    
                    count = 0
                    # Wrap the image iteration with tqdm for progress tracking
                    for img in tqdm(images, desc=f"Copying images for {file_name}", leave=False):
                        img_filename = img['file_name']
                        src_path = os.path.join(source_images_dir, img_filename)
                        dest_path = os.path.join(dest_images_dir, img_filename)
                        
                        if os.path.exists(src_path):
                            if not os.path.exists(dest_path):
                                shutil.copy2(src_path, dest_path)
                                count += 1
                        else:
                            print(f"Warning: Image not found at {src_path}")
                    
                    print(f"Copied {count} images for {file_name}")
                    total_copied += count
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")
            
    print(f"Total images copied: {total_copied}")


def len_dataset(output_dir):
    files = ['vivqa_v2_train.json', 'vivqa_v2_dev.json', 'vivqa_v2_test.json']
    
    print("\n--- Dataset Lengths ---")
    for file_name in files:
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    annotations_len = len(data.get('annotations', []))
                    images_len = len(data.get('images', []))
                    print(f"{file_name}: {annotations_len} annotations, {images_len} images")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    split_dataset(DATA_DIR, OUTPUT_DIR)
    move_images(DATA_DIR, OUTPUT_DIR)
    len_dataset(OUTPUT_DIR)