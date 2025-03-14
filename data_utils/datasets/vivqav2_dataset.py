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
class Vivqav2Dataset(data.Dataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(Vivqav2Dataset, self).__init__()
        with open(json_path, 'r', encoding="utf8") as file:
            json_data = json.load(file)
            

        # vocab
        self.vocab = vocab

        # quesion-answer pairs
        self.annotations = self.load_annotations(json_data)

        # image features
        self.image_features_path = config.FEATURE_PATH.FEATURES

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    id = ann["id"]
                    question = preprocess_sentence(ann["question"], self.vocab.tokenizer)
                    answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    answers = [" ".join(answer) for answer in answers]
                    annotation = {
                        "id": id,
                        "image_id": ann["image_id"],
                        "file_name": image["file_name"],
                        "question": question,
                        "answer": answers
                    }
                    break

            annotations.append(annotation)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        file_name = item["file_name"]
        # 536725.jpg -> 536725
        features = self.load_features(int(file_name.split(".")[0]))
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answer = item["answers"]

        return Instance(
            id=item["id"],
            image_id=image_id,
            file_name=file_name,
            question=question,
            question_tokens=question_tokens,
            answer=answer,
            **features
        )

@META_DATASET.register()
class Vivqav2FeatureDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(Vivqav2FeatureDataset, self).__init__(json_path, vocab, config)
    
    @property
    def question(self):
        return [ann["question"] for ann in self.annotations]
    
    @property
    def answers(self):
        return [ann["answer"] for ann in self.annotations]
    
    def load_annotations(self, json_data):
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answer = ann["answers"]
                    question_id = ann["id"]
                    annotation = {
                        "question_id": question_id,
                        "image_id": ann["image_id"],
                        "file_name": image["file_name"],
                        "question": question,
                        "answer": answer
                    }
                    annotations.append(annotation)      
                
        return annotations
    
    def load_features(self, image_id):
        return super().load_features(image_id)


    def __getitem__(self, idx: int):
        item = self.annotations[idx]

        image_id = item["image_id"]
        file_name = item["file_name"]
        features = self.load_features(int(file_name.split(".")[0]))
        features = {str(key): value for key, value in features.items()}
        question = preprocess_sentence(item["question"], self.vocab.tokenizer)
        question = self.vocab.encode_question(question)
        answer = preprocess_sentence(item["answer"], self.vocab.tokenizer)
        answer = self.vocab.encode_answer(answer)
        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        return Instance(
            question_id=item["question_id"],
            image_id=image_id,
            filename=file_name,
            question=item["question"],
            question_tokens=question,
            answer=item["answer"],
            answer_tokens=answer,
            shifted_right_answer_tokens=shifted_right_answer,
            **features
        )
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    