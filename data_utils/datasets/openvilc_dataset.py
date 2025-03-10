import torch
from torch.utils import data
from data_utils.utils import preprocess_sentence
from data_utils.datasets.base_dataset import BaseDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET
import json
import os
import numpy as np
from typing import Dict, List, Any

@META_DATASET.register()
class OpenViLCImageCaptioningDataset(data.Dataset):
    """
    Dataset này được thiết kế cho bài toán image captioning trên OpenViLC.
    Mỗi annotation chứa một trường "caption" (chuỗi mô tả ảnh).
    """
    def __init__(self, json_path: str, vocab, config) -> None:
        super(OpenViLCImageCaptioningDataset, self).__init__()
        with open(json_path, 'r', encoding="utf8") as file:
            json_data = json.load(file)
            
        # Lưu trữ từ điển
        self.vocab = vocab
        
        # Tải annotations từ file JSON
        self.annotations = self.load_annotations(json_data)
        
        # Đường dẫn đến folder chứa image features (được trích xuất từ VinVL, Faster-RCNN, ...)
        self.image_features_path = config.FEATURE_PATH.FEATURES

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        # Giả sử file JSON có hai trường "annotations" và "images"
        for ann in json_data["annotations"]:
            # Tìm thông tin ảnh tương ứng với annotation
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    # Lấy caption và tiền xử lý (sử dụng tokenizer của vocab nếu có)
                    caption = ann["caption"]
                    # Ở đây, chúng ta giữ nguyên caption dạng chuỗi, việc encoding sẽ được thực hiện ở __getitem__
                    annotation = {
                        "id": ann["id"],
                        "image_id": ann["image_id"],
                        "file_name": image["file_name"],
                        "caption": caption
                    }
                    break
            annotations.append(annotation)
        return annotations

    def load_features(self, image_id: int) -> Dict[str, Any]:
        """
        Hàm này sử dụng phương thức load_features đã được định nghĩa trong lớp BaseDataset.
        """
        return super().load_features(image_id)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        file_name = item["file_name"]
        # Ví dụ: file_name "536725.jpg" -> 536725
        features = self.load_features(int(file_name.split(".")[0]))
        
        # Tiền xử lý caption: token hóa và encode bằng từ điển caption
        # Chú ý: ở đây ta sử dụng encode_caption thay vì encode_question/encode_answer
        caption_tokens = preprocess_sentence(item["caption"], self.vocab.tokenizer)
        caption_encoded = self.vocab.encode_caption(caption_tokens)
        
        return Instance(
            id=item["id"],
            image_id=image_id,
            filename=file_name,
            caption=item["caption"],
            caption_tokens=caption_encoded,
            **features
        )

    def __len__(self) -> int:
        return len(self.annotations)


@META_DATASET.register()
class OpenViLCImageCaptioningFeatureDataset(BaseDataset):
    """
    Dataset này kế thừa từ BaseDataset và được thiết kế cho bài toán image captioning,
    sử dụng image features cùng với caption.
    """
    def __init__(self, json_path: str, vocab, config) -> None:
        super(OpenViLCImageCaptioningFeatureDataset, self).__init__(json_path, vocab, config)
    
    @property
    def captions(self) -> List[str]:
        # Trả về danh sách caption cho toàn bộ annotations
        return [ann["caption"] for ann in self.annotations]
    
    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    # Ở bài toán captioning, ta chỉ cần caption và thông tin ảnh
                    caption = ann["caption"]
                    question_id = ann["id"]  # đặt tên question_id theo cách cũ để duy trì tính nhất quán nếu cần
                    annotation = {
                        "question_id": question_id,
                        "image_id": ann["image_id"],
                        "file_name": image["file_name"],
                        "caption": caption
                    }
                    annotations.append(annotation)
        return annotations
    
    def load_features(self, image_id: int) -> Dict[str, Any]:
        return super().load_features(image_id)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        file_name = item["file_name"]
        features = self.load_features(int(file_name.split(".")[0]))
        features = {str(key): value for key, value in features.items()}
        # Xử lý caption: tiền xử lý và encode bằng vocab
        caption_tokens = preprocess_sentence(item["caption"], self.vocab.tokenizer)
        caption_encoded = self.vocab.encode_caption(caption_tokens)
        # Tạo tensor cho caption dịch phải một vị trí
        caption_encoded = torch.tensor(caption_encoded, dtype=torch.long)
        shifted_right_caption_tokens = torch.zeros_like(caption_encoded).fill_(self.vocab.padding_idx)
        shifted_right_caption_tokens[:-1] = caption_encoded[1:]
        return Instance(
            question_id=item["question_id"],
            image_id=image_id,
            filename=file_name,
            caption=item["caption"],
            caption_tokens=caption_encoded,
            shifted_right_caption_tokens=shifted_right_caption_tokens,
            **features
        )
    
    def __len__(self) -> int:
        return len(self.annotations)
