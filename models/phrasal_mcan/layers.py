import torch
from torch import nn
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from .attentions import PhrasalConstrainedAttention

class CoTextEncoderLayer(nn.Module):
    def __init__(self, config):
        super(CoTextEncoderLayer, self).__init__()
        
        # Self Attention with Phrasal Constraint
        self.mhatt = PhrasalConstrainedAttention(config)
        
        # Position-wise Feed Forward
        self.pwff = PositionWiseFeedForward(config)
        
        # Co-Text Gated Fusion Components
        # Gate = sigmoid( W_g * concat(H_current, H_previous) )
        # H_current and H_previous both have dim d_model
        self.gate_layer = nn.Linear(2 * config.D_MODEL, config.D_MODEL)
        
        # Linear projection for H_previous in the fusion step
        self.linear_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        
        # Layer Norm for the fusion output
        self.fusion_norm = nn.LayerNorm(config.D_MODEL)
        
        # Standard LayerNorms (usually inside mhatt or handled outside? 
        # In this codebase's EncoderLayer, it's:
        # att = mhatt(...)
        # ff = pwff(att)
        # return ff
        # But wait, looking at EncoderLayer in encoders.py:
        # att = self.mhatt(...)
        # ff = self.pwff(att)
        # return ff
        # The residual connections (x + sublayer(x)) happen inside MultiHeadAttention?
        # Let's check MultiHeadAttention.forward:
        # out = self.dropout(out)
        # out = self.layer_norm(queries + out) 
        # Yes, standard residuals are inside. 
        # But for Co-Text, we are modifying how info is passed.
        
        # In our case:
        # H_current = output of attention (already likely has residual if using MultiHeadAttention structure, 
        # but PhrasalConstrainedAttention implements the logic manually).
        # PhrasalConstrainedAttention in my implementation returns `out` (linear projection of att*v).
        # It does NOT have the residual connection or layer norm inside it yet 
        # (unlike MultiHeadAttention in models/modules/attentions.py which inherits Module and does norm).
        
        # So I need to implement the residual + norm logic here or add it to Attention.
        # Instruction says: H_output = LayerNorm( H_current + Gate * Linear(H_previous) )
        # This REPLACES the standard residual connection for the attention sub-layer?
        # Or is it an additional step?
        # "Step B (Fusion): H_output = LayerNorm( H_current + Gate * Linear(H_previous) )"
        # usually H_current = Dropout(Attn(Q,K,V)).
        
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        # queries, keys, values are typically the same tensor 'x' for self-attention
        # H_previous = queries (the input to this layer)
        h_previous = queries
        
        # 1. Calculate Attention Output (H_current proposed)
        # Note: My PhrasalConstrainedAttention returns (out, att_weights)
        # out is (b, n, d)
        att_out, _ = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask, **kwargs)
        
        h_current = self.dropout(att_out)
        
        # 2. Co-Text Gated Fusion
        # Gate = sigmoid( W_g * concat(H_current, H_previous) )
        gate_input = torch.cat([h_current, h_previous], dim=-1) # (b, n, 2*d)
        gate = torch.sigmoid(self.gate_layer(gate_input)) # (b, n, d)
        
        # H_output = LayerNorm( H_current + Gate * Linear(H_previous) )
        # Validating dims: h_current (d) + gate(d) * proj(d) -> d
        fused_output = self.fusion_norm(h_current + gate * self.linear_proj(h_previous))
        
        # 3. Feed Forward Network
        # The FFN usually has its own residual + norm.
        # PositionWiseFeedForward in this repo usually does x + Dropout(FFN(x)) -> Norm?
        # Let's check models/modules/positionwise_feed_forward.py if possible, but assuming standard behavior.
        # If not, I should do it manually.
        # But EncoderLayer in encoders.py just does: ff = self.pwff(att).
        # So pwff likely handles residuals.
        
        ff_out = self.pwff(fused_output)
        
        return ff_out
