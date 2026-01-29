
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import json
import os
import glob
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(ROOT_DIR, "data", "ds102", "images")
OUTPUT_FILE = os.path.join(ROOT_DIR, "detected_objects.json")
THRESHOLD = 0.5

# COCO Class Names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Vietnamese Mapping (Approximate)
VN_MAPPING = {
    'person': 'người', 'bicycle': 'xe đạp', 'car': 'ô tô', 'motorcycle': 'xe máy',
    'airplane': 'máy bay', 'bus': 'xe buýt', 'train': 'tàu hỏa', 'truck': 'xe tải',
    'boat': 'thuyền', 'traffic light': 'đèn giao thông', 'fire hydrant': 'vòi chữa cháy',
    'stop sign': 'biển báo dừng', 'parking meter': 'đồng hồ đỗ xe', 'bench': 'ghế dài',
    'bird': 'chim', 'cat': 'mèo', 'dog': 'chó', 'horse': 'ngựa', 'sheep': 'cừu',
    'cow': 'bò', 'elephant': 'voi', 'bear': 'gấu', 'zebra': 'ngựa vằn', 'giraffe': 'hươu cao cổ',
    'backpack': 'ba lô', 'umbrella': 'ô', 'handbag': 'túi xách', 'tie': 'cà vạt',
    'suitcase': 'vali', 'frisbee': 'đĩa ném', 'skis': 'ván trượt tuyết', 'snowboard': 'ván trượt tuyết',
    'sports ball': 'bóng', 'kite': 'diều', 'baseball bat': 'gậy bóng chày', 'baseball glove': 'găng tay bóng chày',
    'skateboard': 'ván trượt', 'surfboard': 'ván lướt sóng', 'tennis racket': 'vợt tennis',
    'bottle': 'chai', 'wine glass': 'ly rượu', 'cup': 'cốc', 'fork': 'nĩa',
    'knife': 'dao', 'spoon': 'thìa', 'bowl': 'bát', 'banana': 'chuối',
    'apple': 'táo', 'sandwich': 'bánh mì kẹp', 'orange': 'cam', 'broccoli': 'súp lơ',
    'carrot': 'cà rốt', 'hot dog': 'xúc xích', 'pizza': 'bánh pizza', 'donut': 'bánh donut',
    'cake': 'bánh ngọt', 'chair': 'ghế', 'couch': 'ghế sofa', 'potted plant': 'chậu cây',
    'bed': 'giường', 'dining table': 'bàn ăn', 'toilet': 'nhà vệ sinh', 'tv': 'tv',
    'laptop': 'máy tính xách tay', 'mouse': 'chuột', 'remote': 'điều khiển', 'keyboard': 'bàn phím',
    'cell phone': 'điện thoại', 'microwave': 'lò vi sóng', 'oven': 'lò nướng', 'toaster': 'máy nướng bánh mì',
    'sink': 'bồn rửa', 'refrigerator': 'tủ lạnh', 'book': 'sách', 'clock': 'đồng hồ',
    'vase': 'lọ hoa', 'scissors': 'kéo', 'teddy bear': 'gấu bông', 'hair drier': 'máy sấy tóc',
    'toothbrush': 'bàn chải đánh răng'
}

def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect(model, image_path, device, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    
    detected_classes = []
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()
    
    for label, score in zip(labels, scores):
        if score > threshold:
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            detected_classes.append(class_name)
            
    return list(set(detected_classes)) # Unique objects

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model()
    model.to(device)
    
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    print(f"Found {len(image_paths)} images.")
    
    results = {}
    
    for img_path in tqdm(image_paths):
        try:
            filename = os.path.basename(img_path)
            # Image ID is usually the filename without extension (and maybe stripped leading zeros)
            # In dataset inspection: "image_id": 439374 for 000439374.jpg?
            # Let's check format. File list shows: 001171.jpg
            # Dataset ID: 142470.
            # Let's save key as filename to be safe, or try to parse int.
            # Safe bet: save as filename, and partial match later.
            
            eng_objects = detect(model, img_path, device, THRESHOLD)
            vn_objects = [VN_MAPPING.get(obj, obj) for obj in eng_objects]
            
            results[filename] = {
                "english": eng_objects,
                "vietnamese": vn_objects
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Saved detections to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
