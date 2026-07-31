"""
Static Bilingual Dictionary (Vietnamese to YOLO COCO labels)
"""

from typing import Dict, List

# Map from Vietnamese entity names to COCO English labels
VI_TO_EN_COCO: Dict[str, str] = {
    # Person
    "người": "person",
    "người đàn ông": "person",
    "người phụ nữ": "person",
    "cậu bé": "person",
    "cô gái": "person",
    "chàng trai": "person",
    "bé trai": "person",
    "bé gái": "person",
    "anh ấy": "person",
    "cô ấy": "person",
    "bạn": "person",
    "họ": "person",
    
    # Vehicles
    "xe đạp": "bicycle",
    "xe hơi": "car",
    "ô tô": "car",
    "xe máy": "motorcycle",
    "xe mô tô": "motorcycle",
    "máy bay": "airplane",
    "xe buýt": "bus",
    "tàu hỏa": "train",
    "xe lửa": "train",
    "xe tải": "truck",
    "thuyền": "boat",
    "tàu": "boat",
    "đèn giao thông": "traffic light",
    "biển báo": "stop sign",
    
    # Furniture & Objects
    "ghế": "chair",
    "ghế sofa": "couch",
    "giường": "bed",
    "bàn": "dining table",
    "tivi": "tv",
    "laptop": "laptop",
    "chuột": "mouse",
    "bàn phím": "keyboard",
    "điện thoại": "cell phone",
    "sách": "book",
    "đồng hồ": "clock",
    "bình hoa": "vase",
    "kéo": "scissors",
    "gấu bông": "teddy bear",
    "ô": "umbrella",
    "cà vạt": "tie",
    "mũ bảo hộ": "helmet", # Non-COCO standard, but in some extended YOLO / DINO
    "mũ bảo hiểm": "helmet",
    
    # Animals
    "chó": "dog",
    "cún": "dog",
    "mèo": "cat",
    "ngựa": "horse",
    "bò": "cow",
    "voi": "elephant",
    "gấu": "bear",
    "ngựa vằn": "zebra",
    "hươu cao cổ": "giraffe",
    "chim": "bird",
    "cừu": "sheep",
    
    # Sports / Outdoors
    "bóng": "sports ball",
    "quả bóng": "sports ball",
    "diều": "kite",
    "ván lướt sóng": "surfboard",
    "vợt tennis": "tennis racket",
    
    # Food
    "chuối": "banana",
    "táo": "apple",
    "bánh sandwich": "sandwich",
    "cam": "orange",
    "bông cải xanh": "broccoli",
    "cà rốt": "carrot",
    "xúc xích": "hot dog",
    "pizza": "pizza",
    "bánh donut": "donut",
    "bánh ngọt": "cake",
    "chai": "bottle",
    "cốc": "cup",
    "ly": "cup",
    "cái bát": "bowl",
    "tô": "bowl"
}

def get_static_translation(entity: str) -> str:
    """
    Look up the YOLO label for a given Vietnamese entity.
    Returns the label if found, else None.
    """
    return VI_TO_EN_COCO.get(entity.lower().strip(), None)
