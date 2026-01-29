import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from builders.attention_builder import META_ATTENTION

@META_ATTENTION.register()
class PhrasalConstrainedAttention(nn.Module):
    '''
    Scaled dot-product attention with Phrasal Lexeme constraint
    '''

    def __init__(self, config):
        super(PhrasalConstrainedAttention, self).__init__()

        d_model = config.D_MODEL
        h = config.HEAD
        d_k = config.D_KEY
        d_v = config.D_VALUE

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        
        # Phrasal Lexeme Scorer components
        self.phrasal_linear = nn.Linear(d_model, d_model)
        
        # Lambda learnable parameter for constraint strength
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.xavier_uniform_(self.phrasal_linear.weight)
        
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.constant_(self.phrasal_linear.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # Standard Attention Projections
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Basic Attention Scores: (Q * K^T) / sqrt(d_k)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # Phrasal Constrained Attention Logic
        # Calculate Phrasal Score P
        # P_ij = sigmoid(h_i * W * h_j^T + b)
        # using queries and keys as h_i and h_j (before projection to heads)
        # Note: In self-attention queries==keys, effectively computing relation between all tokens.
        
        # (b_s, nq, d_model) -> (b_s, nq, d_model)
        phrasal_proj = self.phrasal_linear(queries) 
        
        # (b_s, nq, d_model) @ (b_s, nk, d_model)^T -> (b_s, nq, nk)
        phrasal_logits = torch.matmul(phrasal_proj, keys.transpose(-2, -1))
        
        P = torch.sigmoid(phrasal_logits)
        
        # Apply constraint: Attention_Final = Softmax( Attention_Scores + Lambda * log(P + epsilon) )
        # Broadcast P to match heads: (b_s, 1, nq, nk)
        epsilon = 1e-9
        phrasal_constraint = self.lambda_param * torch.log(P + epsilon)
        
        att = att + phrasal_constraint.unsqueeze(1)

        if attention_mask is not None:
            # 1. Ensure mask is 4D (b_s, 1, nq, nk)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Handle Beam Search broadcasting
            if att.shape[0] != attention_mask.shape[0]:
                beam_size = att.shape[0] // attention_mask.shape[0]
                attention_mask = attention_mask.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
                attention_mask = attention_mask.view(att.shape[0], *attention_mask.shape[2:])
            
            att += attention_mask

        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, att
