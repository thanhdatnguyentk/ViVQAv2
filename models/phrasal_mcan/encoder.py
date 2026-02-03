"""
Phrasal-aware Encoders for Vietnamese VQA.

Implements:
- PhrasalEncoder: Self-attention encoder with phrasal score propagation
- PhrasalGuidedAttentionEncoder: Cross-modal encoder with phrasal-aware attention
"""

import torch
from torch import nn

from models.modules.pos_embeddings import SinusoidPositionalEmbedding
from models.phrasal_mcan.layers import PhrasalEncoderLayer, PhrasalGuidedEncoderLayer
from builders.encoder_builder import META_ENCODER


@META_ENCODER.register()
class PhrasalEncoder(nn.Module):
    """
    Self-attention encoder with phrasal score propagation across layers.
    
    Implements Co-Text Module from ViWordFormer:
    - Each layer updates the phrasal score matrix based on historical context.
    - Initial running_phrasal_scores is None.
    - Phrasal scores are passed and updated sequentially.
    """

    def __init__(self, config):
        super(PhrasalEncoder, self).__init__()
        
        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL
        self.layers = nn.ModuleList([
            PhrasalEncoderLayer(config.SELF_ATTENTION) 
            for _ in range(config.LAYERS)
        ])

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        """
        Forward pass with phrasal score propagation (Co-Text).
        """
        # Initial embedding with positional encoding
        out = self.layer_norm(features) + self.pos_embedding(features)
        
        # Initialize running phrasal scores (Co-Text state)
        running_phrasal_scores = None
        
        for layer in self.layers:
            # Each layer receives and updates the running phrasal scores
            out, running_phrasal_scores = layer(
                queries=out, 
                keys=out, 
                values=out, 
                attention_mask=padding_mask,
                prev_phrasal_scores=running_phrasal_scores
            )

        return out


@META_ENCODER.register()
class PhrasalGuidedAttentionEncoder(nn.Module):
    """
    Guided attention encoder with phrasal-aware cross-modal attention.
    
    Refactored to use running_phrasal_scores for language structure propagation.
    """

    def __init__(self, config):
        super(PhrasalGuidedAttentionEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.d_model = config.D_MODEL

        self.guided_attn_layers = nn.ModuleList([
            PhrasalGuidedEncoderLayer(config.GUIDED_ATTENTION) 
            for _ in range(config.LAYERS)
        ])

    def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
                language_features: torch.Tensor, language_padding_mask: torch.Tensor):
        """
        Forward pass with guided attention and phrasal propagation.
        """
        # Initial embedding
        out = self.layer_norm(vision_features) + self.pos_embedding(vision_features)
        
        # Initialize running phrasal scores for guiding modality (language)
        running_phrasal_scores = None
        
        for guided_attn_layer in self.guided_attn_layers:
            out, running_phrasal_scores = guided_attn_layer(
                queries=out,
                keys=language_features,
                values=language_features,
                self_attention_mask=vision_padding_mask,
                guided_attention_mask=language_padding_mask,
                prev_phrasal_scores=running_phrasal_scores
            )

        return out
