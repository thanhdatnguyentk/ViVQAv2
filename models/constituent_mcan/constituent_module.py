import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([module for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as per official BERT implementation.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int, d_kv: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_kv
        self.d_kv = d_kv
        self.head = head

        self.fc_q = nn.Linear(d_model, head * d_kv)
        self.fc_k = nn.Linear(d_model, head * d_kv)
        self.fc_v = nn.Linear(d_model, head * d_kv)

    def forward(self, queries, keys, values, group_prob, attention_mask):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)   # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)     # (b_s, h, nk, d_kv)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nk, d_kv)

        att = torch.matmul(q, k) / np.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            # attention_mask is typically (b_s, 1, 1, nk) for standard attention
            # but here we need it to be compatible with (b_s, h, nq, nk)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            att.masked_fill(attention_mask == 0, -1e4)
            
        att = torch.softmax(att, dim=-1)
        att = att * group_prob
        output = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, -1, self.d_model)

        return output

class GroupAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(GroupAttention, self).__init__()
        self.h = head
        self.d_k = d_model // head
        self.linear_key = nn.Linear(self.d_k, self.d_k)
        self.linear_query = nn.Linear(self.d_k, self.d_k)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, pad_mask, prior):
        bs, seq_len = context.size()[:2]

        context = self.norm(context).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)

        a = torch.diag(torch.ones(seq_len - 1), 1).long().to(context.device)
        b = torch.diag(torch.ones(seq_len), 0).long().to(context.device)
        c = torch.diag(torch.ones(seq_len - 1), -1).long().to(context.device)

        # Simplified mask logic from provided code
        # pad_mask is expected as (bs, seq_len)
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)
            mask = torch.logical_and(mask, (a + c).unsqueeze(0).unsqueeze(0))
        else:
            mask = (a + c).bool()

        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k
        
        # apply mask
        scores = scores.masked_fill(mask == 0, -1e4)
        neibor_attn = F.softmax(scores, dim = -1)
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-4)
        neibor_attn = prior + (1. - prior)*neibor_attn

        tri_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal = 0).float().to(context.device)
        tri_matrix = tri_matrix.unsqueeze(0).unsqueeze(0)
        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-4)
        
        return g_attn, neibor_attn

class ConstituentEncoderLayer(nn.Module):
    def __init__(self, head, d_model, d_kv, d_ff, dropout=0.1):
        super().__init__()
        self.group_attn = GroupAttention(head, d_model, dropout=dropout)
        self.self_attn = ScaledDotProductAttention(head, d_model, d_kv)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.size = d_model

    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        x = self.sublayer[1](x, self.feed_forward)
        
        return x, group_prob, break_prob
    
class QuestionConstituentEncoder(nn.Module):
    def __init__(self, head, d_model, d_kv, d_ff, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([ConstituentEncoderLayer(head, d_model, d_kv, d_ff, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x: (bs, seq_len, d_model)
        # mask: (bs, seq_len)
        break_probs = []
        group_prob = 0.
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask, group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1) # (bs, num_layers, head, seq_len, seq_len)

        return x, break_probs
