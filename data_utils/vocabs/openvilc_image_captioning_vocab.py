import torch
from data_utils.vocabs.vocab import Vocab
from data_utils.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB

from collections import Counter
import json
from typing import List, Union

@META_VOCAB.register()
class OpenViLCCaptioningVocab(Vocab):
    """
    Đây là lớp từ điển (vocabulary) được xây dựng cho bài toán image captioning.
    Từ điển được tạo ra từ các câu caption trong dataset.
    Các token đặc biệt (BOS, EOS, PAD, UNK) được thêm vào từ đầu.
    """
    def __init__(self, config):
        # Khởi tạo các token đặc biệt với giá trị từ config nếu có, hoặc dùng giá trị mặc định
        self.bos_token = config.BOS_TOKEN if config.BOS_TOKEN is not None else "<bos>"
        self.eos_token = config.EOS_TOKEN if config.EOS_TOKEN is not None else "<eos>"
        self.pad_token = config.PAD_TOKEN if config.PAD_TOKEN is not None else "<pad>"
        self.unk_token = config.UNK_TOKEN if config.UNK_TOKEN is not None else "<unk>"
        self.min_freq = config.MIN_FREQ if hasattr(config, "MIN_FREQ") else 1
        self.tokenizer = config.TOKENIZER  # Có thể là None nếu không dùng tokenizer đặc biệt
        self.max_answer_length = 0
        super(OpenViLCCaptioningVocab, self).__init__(config)

    def make_vocab(self, json_dirs: List[str]):
        """
        Hàm tạo từ điển dựa trên các file JSON chứa thông tin caption.
        Mỗi file JSON cần có trường "annotations" với mỗi annotation chứa một trường "caption".
        """
        self.freqs = Counter()
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir, encoding="utf8"))
            for ann in json_data["annotations"]:
                # Lấy caption và tiền xử lý để tách thành danh sách token
                caption_tokens = preprocess_sentence(ann["caption"], self.tokenizer)
                self.freqs.update(caption_tokens)
                # Cập nhật độ dài caption lớn nhất (bao gồm BOS và EOS)
                caption_len = len(caption_tokens) + 2
                if caption_len > self.max_answer_length:
                    self.max_answer_length = caption_len

        # Xây dựng từ điển: thêm các token đặc biệt đầu tiên
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.itoa = {idx: token for idx, token in enumerate(special_tokens)}
        start_idx = len(special_tokens)
        # Thêm các token có tần suất >= min_freq và chưa nằm trong special_tokens
        for token, freq in self.freqs.items():
            if freq >= self.min_freq and token not in special_tokens:
                self.itoa[start_idx] = token
                start_idx += 1
        self.atoi = {token: idx for idx, token in self.itoa.items()}
        self.total_words = len(self.itoa)
        print(f"Total words in vocabulary: {self.total_words}")
    
    def encode_caption(self, caption: Union[str, List[str]]) -> torch.Tensor:
        """
        Hàm chuyển một câu caption (dạng string hoặc danh sách token) thành tensor các chỉ số.
        Thêm token BOS và EOS vào đầu và cuối câu.
        """
        if isinstance(caption, str):
            tokens = preprocess_sentence(caption, self.tokenizer)
        else:
            tokens = caption
        tokens = [self.bos_token] + tokens + [self.eos_token]
        indices = [self.atoi.get(token, self.atoi[self.unk_token]) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_caption(self, caption_vec: torch.Tensor, join_words: bool = True) -> Union[str, List[str]]:
        """
        Hàm chuyển tensor các chỉ số thành câu caption.
        Nếu join_words=True thì trả về chuỗi, ngược lại trả về danh sách token.
        Bỏ qua các token PAD, BOS và EOS khi giải mã.
        """
        tokens = []
        for idx in caption_vec.tolist():
            token = self.itoa.get(idx, self.unk_token)
            if token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            tokens.append(token)
        return " ".join(tokens) if join_words else tokens
