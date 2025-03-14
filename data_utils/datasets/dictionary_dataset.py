from data_utils.utils import preprocess_sentence
from data_utils.datasets.base_dataset import BaseDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

from typing import Dict, List

@META_DATASET.register()
class DictionaryDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(DictionaryDataset, self).__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = preprocess_sentence(ann["question"], self.vocab.tokenizer)
                    # answers = [preprocess_sentence(answer, self.vocab.tokenizer) for answer in ann["answers"]]
                    # answers = [" ".join(answer) for answer in answers]
                    answers = preprocess_sentence(ann["answers"], self.vocab.tokenizer)
                    annotation = {
                        "question_id": ann["id"],
                        # "type": ann["QA-type"],
                        "question": question,
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["file_name"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        filename = filename.split(".")[0]
        filename = int(filename)
        features = self.load_features(filename)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]
        answers_tokens = self.vocab.encode_answer(answers)

        return Instance(
            question_id=item["question_id"],
            # type=item["type"],
            image_id=image_id,
            filename=filename,
            question=question,
            question_tokens=question_tokens,
            answers=answers,
            answers_tokens = answers_tokens,
            **features
        )