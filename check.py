from transformers import T5Tokenizer

try:
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small", legacy=True)
except Exception as e:
    print(f"Lỗi xảy ra khi tải tokenizer: {e}")

import sentencepiece
print(sentencepiece.__version__)
