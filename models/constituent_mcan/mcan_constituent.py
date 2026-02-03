import torch
from torch import nn
from torch.nn import functional as F

from models.base_classification import BaseClassificationModel
from utils.instance import Instance
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

from .constituent_module import QuestionConstituentEncoder

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc2 = nn.Linear(config.D_MODEL, 1)

    def forward(self, features: torch.Tensor):
        output = self.dropout(self.relu(self.fc1(features)))
        output = self.fc2(output)

        return output

@META_ARCHITECTURE.register()
class ConstituentMCAN(BaseClassificationModel):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        # Integration point: Constituent Module
        # We need to define parameters for QuestionConstituentEncoder. 
        # For simplicity, we can use the same d_model and head as the main encoder.
        self.constituent_encoder = QuestionConstituentEncoder(
            head=config.SELF_ENCODER.SELF_ATTENTION.HEAD,
            d_model=config.D_MODEL,
            d_kv=config.SELF_ENCODER.SELF_ATTENTION.D_KEY,
            d_ff=config.SELF_ENCODER.SELF_ATTENTION.D_FF,
            num_layers=getattr(config, "CONSTITUENT_LAYERS", 3),
            dropout=getattr(config, "DROPOUT", config.SELF_ENCODER.SELF_ATTENTION.DROPOUT)
        )

        self.self_encoder = build_encoder(config.SELF_ENCODER)
        self.guided_encoder = build_encoder(config.GUIDED_ENCODER)

        self.vision_attr_reduce = MLP(config.VISION_ATTR_REDUCE)
        self.text_attr_reduce = MLP(config.TEXT_ATTR_REDUCE)

        self.vision_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.text_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.classify = nn.Linear(config.D_MODEL, vocab.total_answers)

    def forward(self, input_features: Instance):
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)

        # ----------------------------------------------------------------------
        # NEW: Constituent Module Refinement
        # Note: text_padding_mask is typically (bs, 1, 1, seq_len) in MCAN, 
        # but our ConstituentEncoder expects (bs, seq_len) or handles the dimensions.
        # We'll normalize the mask to (bs, seq_len) if it comes in a different shape.
        mask = text_padding_mask
        if mask is not None:
            if mask.dim() == 4: # (b, 1, 1, s)
                mask = mask.view(mask.size(0), mask.size(-1))
            elif mask.dim() == 3: # (b, 1, s)
                mask = mask.view(mask.size(0), mask.size(-1))
        
        text_features, break_probs = self.constituent_encoder(text_features, mask)
        # ----------------------------------------------------------------------

        # SA
        text_features = self.self_encoder(
            features=text_features,
            padding_mask=text_padding_mask
        )

        # GSA
        vision_features = self.guided_encoder(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        )

        attended_vision_features = self.vision_attr_reduce(vision_features)
        attended_vision_features = F.softmax(attended_vision_features, dim=1)
        attended_text_features = self.text_attr_reduce(text_features)
        attended_text_features = F.softmax(attended_text_features, dim=1)

        weighted_vision_features = (vision_features * attended_vision_features).sum(dim=1)
        weighted_text_features = (text_features * attended_text_features).sum(dim=1)

        output = self.layer_norm(self.vision_proj(weighted_vision_features) + self.text_proj(weighted_text_features))
        output = self.classify(output)

        return F.log_softmax(output, dim=-1), break_probs
