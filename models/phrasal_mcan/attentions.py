"""
Phrasal-aware Attention Mechanisms for Vietnamese VQA.

Implements Phrasal Lexeme module from ViWordFormer paper:
- PhrasalScaledDotProductAttention: Computes phrasal scores via Bilinear layer
  and integrates them into standard scaled dot-product attention.
  Refactored to follow ViWordFormer (Eq 2, 3, 4, 8).
"""

import torch
from torch import nn
import numpy as np
from builders.attention_builder import META_ATTENTION


@META_ATTENTION.register()
class PhrasalScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention enhanced with Phrasal Score computation.
    
    The Phrasal Score P(i,j) captures the likelihood that tokens i and j 
    belong to the same phrase (cụm từ) in Vietnamese syllable-level input.
    """

    def __init__(self, config):
        super(PhrasalScaledDotProductAttention, self).__init__()

        d_model = config.D_MODEL
        h = config.HEAD
        d_k = config.D_KEY
        d_v = config.D_VALUE

        # Standard Q, K, V projections
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        # Phrasal Score computation: Bilinear(hidden_dim, hidden_dim, 1)
        self.phrasal_bilinear = nn.Bilinear(d_model, d_model, 1, bias=True)
        
        # Learnable lambda parameter to control phrasal influence
        lambda_init = getattr(config, 'LAMBDA_INIT', 1.0)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

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
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        # Initialize bilinear layer
        nn.init.xavier_uniform_(self.phrasal_bilinear.weight)
        nn.init.constant_(self.phrasal_bilinear.bias, 0)

    def compute_phrasal_scores(self, x):
        """
        Compute Phrasal Score matrix P for all token pairs (i, j).
        
        Simulates Neighboring logic (Eq 2, 3, 4) using vectorization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            P_current: Phrasal score matrix of shape (batch_size, seq_len, seq_len)
        """
        b_s, n, d = x.shape
        
        # self.phrasal_bilinear.weight has shape (1, d, d)
        W = self.phrasal_bilinear.weight[0]  # (d, d)
        
        # Content correlation (Part of Eq 4)
        x_W = torch.matmul(x, W)
        P_content = torch.matmul(x_W, x.transpose(-1, -2)) + self.phrasal_bilinear.bias  # (b_s, n, n)
        
        # Neighboring distance bias (Eq 2, 3)
        # Eq 2: dist(i, j) = |i - j|
        indices = torch.arange(n, device=x.device)
        dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).float()
        
        # Eq 3: Neighboring score (simulation via distance-based penalty)
        dist_bias = -torch.log(1 + dist)
        
        # Eq 4: Combining features and neighboring logic
        P_current = torch.sigmoid(P_content + dist_bias.unsqueeze(0))
        
        return P_current

    def forward(self, queries, keys, values, attention_mask=None, prev_phrasal_scores=None, **kwargs):
        """
        Forward pass with phrasal-enhanced attention and Co-Text Module.
        
        Args:
            queries: (batch_size, n_queries, d_model)
            keys: (batch_size, n_keys, d_model)
            values: (batch_size, n_keys, d_model)
            attention_mask: Optional mask for attention
            prev_phrasal_scores: Phrasal scores from previous layer (P_old)
        
        Returns:
            out: Attention output (batch_size, n_queries, d_model)
            att: Attention weights (batch_size, heads, n_queries, n_keys)
            phrasal_scores: Updated phrasal scores (P_new)
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # Step 1: Compute standard Q, K, V projections
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)      # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)    # (b_s, h, nk, d_v)

        # Step 2: Compute standard attention scores A = (Q @ K^T) / sqrt(d_k)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # Step 3: Compute current Phrasal Scores P_current
        # Only compute for self-attention case (queries are used for language structure)
        P_current = self.compute_phrasal_scores(queries)  # (b_s, nq, nq)
        
        # Step 4: Co-Text Module (Eq 8): P_new = P_old + (1 - P_old) * P_current
        if prev_phrasal_scores is not None:
            phrasal_scores = prev_phrasal_scores + (1 - prev_phrasal_scores) * P_current
        else:
            phrasal_scores = P_current
            
        # Step 5: Integrate phrasal scores into attention
        if nq == nk:
            # Self-attention case: A_final = A + lambda * log(P + eps)
            # Expand phrasal scores for multi-head: (b_s, nq, nk) -> (b_s, 1, nq, nk)
            phrasal_log = self.lambda_param * torch.log(phrasal_scores.unsqueeze(1) + 1e-9)
            att = att + phrasal_log

        # Step 6: Apply attention mask if provided
        if attention_mask is not None:
            att = att + attention_mask

        # Step 7: Softmax normalization
        att = torch.softmax(att, dim=-1)

        # Step 8: Compute output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)

        return out, att, phrasal_scores
