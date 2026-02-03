"""
Phrasal-aware Layers for Vietnamese VQA.

Implements:
- PhrasalMultiHeadAttention: Multi-head wrapper with phrasal score propagation
- PhrasalEncoderLayer: Encoder layer with Co-Text score propagation
"""

import torch
from torch import nn

from models.modules.containers import Module
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from builders.attention_builder import build_attention


class PhrasalMultiHeadAttention(Module):
    """
    Multi-head attention with phrasal score propagation.
    
    Wraps PhrasalScaledDotProductAttention and handles phrasal score pass-through.
    """

    def __init__(self, config):
        super(PhrasalMultiHeadAttention, self).__init__()
        
        d_model = config.D_MODEL

        self.use_aoa = getattr(config, 'USE_AOA', False)
        
        if self.use_aoa:
            self.informative_attention = nn.Linear(2 * d_model, d_model)
            self.gated_attention = nn.Linear(2 * d_model, d_model)

        self.attention = build_attention(config)

        self.dropout = nn.Dropout(p=config.DROPOUT)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = getattr(config, 'CAN_BE_STATEFUL', False)
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask, prev_phrasal_scores=None, **kwargs):
        """
        Forward pass with phrasal score propagation.
        """
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        # Call attention with prev_phrasal_scores
        attention_output = self.attention(
            queries, keys, values, attention_mask, 
            prev_phrasal_scores=prev_phrasal_scores, 
            **kwargs
        )
        
        if len(attention_output) == 3:
            out, _, phrasal_scores = attention_output
        else:
            out, _ = attention_output
            phrasal_scores = None

        # Residual connection and layer normalization
        out = self.dropout(out)
        out = self.layer_norm(queries + out)

        if self.use_aoa:
            aoa_input = torch.cat([queries, out], dim=-1)
            i = self.informative_attention(aoa_input)
            g = torch.sigmoid(self.gated_attention(aoa_input))
            out = i * g
            
        return out, phrasal_scores


class PhrasalEncoderLayer(nn.Module):
    """
    Encoder layer with Co-Text phrasal score propagation.
    
    Removed legacy gated feature fusion as per ViWordFormer theory.
    Phrasal information is now preserved via score accumulation across layers.
    """

    def __init__(self, config):
        super(PhrasalEncoderLayer, self).__init__()
        self.mhatt = PhrasalMultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask, prev_phrasal_scores=None, **kwargs):
        """
        Forward pass with Phrasal Score propagation.
        """
        # Self-attention with phrasal scoring and Co-Text propagation
        att_out, phrasal_scores = self.mhatt(
            queries=queries, 
            keys=keys, 
            values=values, 
            attention_mask=attention_mask, 
            prev_phrasal_scores=prev_phrasal_scores,
            **kwargs
        )
        
        # Feed-forward network
        ff_out = self.pwff(att_out)
        
        return ff_out, phrasal_scores


class PhrasalGuidedEncoderLayer(nn.Module):
    """
    Guided encoder layer with phrasal-aware attention.
    Refactored to match PhrasalEncoderLayer logic.
    """

    def __init__(self, config):
        super(PhrasalGuidedEncoderLayer, self).__init__()
        
        # Self-attention on queries
        self.self_mhatt = PhrasalMultiHeadAttention(config)
        # Guided attention from keys/values
        self.guided_mhatt = PhrasalMultiHeadAttention(config)
        # Feed-forward
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, self_attention_mask, guided_attention_mask, 
                prev_phrasal_scores=None, **kwargs):
        """
        Forward pass with guided attention and phrasal score propagation.
        """
        # Self-attention with phrasal scoring
        self_att, phrasal_scores = self.self_mhatt(
            queries=queries,
            keys=queries, 
            values=queries,
            attention_mask=self_attention_mask,
            prev_phrasal_scores=prev_phrasal_scores,
            **kwargs
        )
        
        # Guided attention (cross-modal)
        guided_att, _ = self.guided_mhatt(
            queries=self_att, 
            keys=keys, 
            values=values,
            attention_mask=guided_attention_mask,
            **kwargs
        )

        # Feed-forward
        ff_out = self.pwff(guided_att)

        return ff_out, phrasal_scores
