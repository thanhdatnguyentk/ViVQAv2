import torch
from torch import nn
from builders.encoder_builder import META_ENCODER
from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from .layers import CoTextEncoderLayer

@META_ENCODER.register()
class PhrasalEncoder(nn.Module):
    def __init__(self, config):
        super(PhrasalEncoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([CoTextEncoderLayer(config.SELF_ATTENTION) for _ in range(config.LAYERS)])

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        # Initial Embedding + Position
        out = self.layer_norm(features) + self.pos_embedding(features)
        
        # Pass through Co-Text Fusion Layers
        for layer in self.layers:
            out = layer(queries=out, keys=out, values=out, attention_mask=padding_mask)

        return out
