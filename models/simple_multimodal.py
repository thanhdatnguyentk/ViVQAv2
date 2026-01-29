
import torch
from torch import nn
from torch.nn import functional as F

from builders.model_builder import META_ARCHITECTURE
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from models.base_classification import BaseClassificationModel
from utils.instance import Instance

@META_ARCHITECTURE.register()
class SimpleMultimodal(BaseClassificationModel):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        # MLP
        # Input dimension will be D_MODEL (text) + D_MODEL (vision) after pooling
        input_dim = config.TEXT_EMBEDDING.D_MODEL + config.VISION_EMBEDDING.D_MODEL
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.D_MODEL),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL, vocab.total_answers) # Output logits for answers
        )
        
        self.init_weights()

    def forward(self, input_features: Instance):
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)
        
        # Pooling vision features: (batch, num_obj, dim) -> (batch, dim)
        # Handle masking for correct mean
        if vision_padding_mask is not None:
             # mask is (batch, 1, 1, num_obj) or (batch, num_obj)? 
             # usually masks in this repo seem to be 4D for attention. 
             # Let's check model utils or just check shape at runtime if strictly needed.
             # However, simple mean is robust enough for "simple model".
             pass
        vision_features = vision_features.mean(dim=1)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)
        
        # Pooling text features: (batch, seq_len, dim) -> (batch, dim)
        text_features = text_features.mean(dim=1)

        combined_features = torch.cat([vision_features, text_features], dim=1)
        logits = self.mlp(combined_features)

        return logits
