import torch

from data_utils.vocabs.vocab import Vocab
from data_utils.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB

from collections import Counter
import json
from typing import List, Union

@META_VOCAB.register()
class VQAv2ClassificationVocab(Vocab):
    # This class is especially designed for ViVQA dataset by treating the VQA as a classification task. 
    # For more information, please visit https://arxiv.org/abs/1708.02711

    def __init__(self, config):
        self.max_answer_length = 0
        super(VQAv2ClassificationVocab, self).__init__(config)

    def make_vocab(self, json_dirs):
        self.freqs = Counter()
        itoa = set()
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir,encoding="utf8"))
            for ann in json_data["annotations"]:
                question = preprocess_sentence(ann["question"], self.tokenizer)
                self.freqs.update(question)
                answer = " ".join(preprocess_sentence(ann["answers"], self.tokenizer))
                itoa.add(answer)
                if len(question) + 2 > self.max_question_length:
                        self.max_question_length = len(question) + 2
                if len(answer) > self.max_answer_length:
                    self.max_answer_length = len(answer)

        self.itoa = {ith: answer for ith, answer in enumerate(itoa)}
        self.atoi = {answer: ith for ith, answer in self.itoa.items()}
        self.total_answers = len(self.atoi)
        print(f"Total answers: {self.total_answers}")

    def encode_answer(self, answer: List[str]) -> torch.Tensor:
        answer = " ".join(answer)
        return torch.tensor([self.atoi[answer]], dtype=torch.long)
        
    def decode_answer(self, answer_vecs: torch.Tensor, join_word=False) -> Union[List[str], List[List[str]]]:
        answers = []
        list_answers = answer_vecs.tolist()
        for answer_idx in list_answers:
            answers.append(self.itoa[answer_idx] if join_word else self.itoa[answer_idx].split())

        return answers