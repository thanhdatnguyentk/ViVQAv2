
import torch
from torch import nn
from torch.nn import functional as F

from builders.model_builder import META_ARCHITECTURE
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.attention_builder import build_attention
from models.base_classification import BaseClassificationModel
from utils.instance import Instance

@META_ARCHITECTURE.register()
class CoAttentionMultimodal(BaseClassificationModel):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        # Co-Attention Layers
        # Vision-Guided Text Attention: Q=Text, K=Vision, V=Vision
        self.vision_guided_text_attention = build_attention(config.CO_ATTENTION)
        
        # Text-Guided Vision Attention: Q=Vision, K=Text, V=Text
        self.text_guided_vision_attention = build_attention(config.CO_ATTENTION)

        self.norm = nn.LayerNorm(config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

        # Classifier
        # Input dimension will be D_MODEL (text) + D_MODEL (vision) after co-attention and pooling
        input_dim = config.D_MODEL * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.D_MODEL),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL, vocab.total_answers) 
        )
        
        self.init_weights()

    def forward(self, input_features: Instance):
        # 1. Embeddings
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)
        # vision_features: (batch, num_obj, d_model)
        # vision_padding_mask: (batch, 1, 1, num_obj)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)
        # text_features: (batch, seq_len, d_model)
        # text_padding_mask: (batch, 1, 1, seq_len)

        # 2. Co-Attention interactions
        # Attention expects: queries, keys, values, attention_mask
        
        # Text attending to Vision (Vision-Guided Text Attention)
        # Q: Text, K: Vision, V: Vision
        attended_text, _ = self.vision_guided_text_attention(
            queries=text_features,
            keys=vision_features,
            values=vision_features,
            attention_mask=vision_padding_mask 
        )
        attended_text = self.norm(attended_text + text_features) # res + norm

        # Vision attending to Text (Text-Guided Vision Attention)
        # Q: Vision, K: Text, V: Text
        attended_vision, _ = self.text_guided_vision_attention(
            queries=vision_features,
            keys=text_features,
            values=text_features,
            attention_mask=text_padding_mask
        )
        attended_vision = self.norm(attended_vision + vision_features) # res + norm
        
        # 3. Pooling (Mean pooling)
        # Using simple mean for now. Assuming masks are handled implicitly or noise is low.
        # Ideally we should mask out padded tokens before mean.
        
        # Expand masks to match feature shape for masking: (batch, 1, 1, len) -> (batch, len, 1) or similar
        # For simple implementation, we'll skip explicit mask multiplication for mean as robust models can learn to ignore pad.
        # But let's try to do it properly if easy. 
        # The masks are usually for Attention (batch, 1, 1, len) where 1 means ignore (usually large negative in attention) or 0? 
        # In this repo, generate_padding_mask returns (batch, 1, 1, len). 
        # Usually it's False for valid, True for pad OR 0 for pad.
        # Standard implementation in attention.py uses mask_fill with -1e9 where mask is True. So True=Pad.
        
        # Let's just do mean over dim 1.
        pooled_text = attended_text.mean(dim=1)
        pooled_vision = attended_vision.mean(dim=1)

        # 4. Classification
        combined_features = torch.cat([pooled_text, pooled_vision], dim=1)
        logits = self.mlp(combined_features)

        return logits
