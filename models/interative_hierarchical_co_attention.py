import torch
from torch import nn
from torch.nn import init, functional as F

from .utils import generate_padding_mask
from models.base_transformer import BaseTransformer
from utils.instance import InstanceList
from builders.model_builder import META_ARCHITECTURE
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from utils.instance import Instance

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

class HierarchicalFeaturesExtractor(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.ngrams = config.N_GRAMS
        self.convs = nn.ModuleList()
        for ngram in self.ngrams:
            self.convs.append(
                nn.Conv1d(in_channels=config.WORD_EMBEDDING_DIM, out_channels=config.D_MODEL, kernel_size=ngram)
            )

        self.reduce_features = nn.Linear(config.D_MODEL, config.D_MODEL)

    def forward(self, features: torch.Tensor):
        ngrams_features = []
        for conv in self.convs:
            ngrams_features.append(conv(features.permute((0, -1, 1))).permute((0, -1, 1)))
        
        features_len = features.shape[-1]
        unigram_features = ngrams_features[0]
        # for each token in the unigram
        for ith in range(features_len):
            # for each n-gram, we ignore the unigram
            for ngram in range(1, max(self.ngrams)):
                # summing all possible n-gram tokens into the unigram
                for prev_ith in range(max(0, ith-ngram+1), min(ith+1, ngrams_features[ngram].shape[1])):
                    unigram_features[:, ith] += ngrams_features[ngram][:, prev_ith]

        return unigram_features

@META_ARCHITECTURE.register()
class IterativeHierarchicalCoAttention(BaseTransformer):
    def __init__(self, config, vocab) -> None:
        super().__init__(config, vocab)

        # embedding module
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        self.question_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        
        # hierarchical feature extractors for texts
        self.hierarchical_extractor = HierarchicalFeaturesExtractor(config.HIERARCHICAL)

        # co-attention module
        self.encoder = build_encoder(config.ENCODER)

        # attributes reduction and classifier
        self.vision_attr_reduce = MLP(config.VISION_ATTR_REDUCE)
        self.text_attr_reduce = MLP(config.TEXT_ATTR_REDUCE)

        self.vision_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.text_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.decoder = build_decoder(config.DECODER, vocab)

        self.padding_idx = vocab.padding_idx

        self.initialize()
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def apply_attention(self, input, attention):
        """ Apply any number of attention maps over the input. """
        input = input.unsqueeze(1).permute(0, 1, -1, -2) # [n, 1, dim, s]
        attention = attention.permute(0, -1, 1) # [n, g, s]
        attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
        weighted = attention * input # [n, g, dim, s]
        weighted = weighted.sum(dim=1).permute(0, -1, 1) # [n, s, dim]
        
        return weighted

    def encoder_forward(self, input_features: Instance):
        v = input_features.region_features
        q = input_features.question_tokens

        v, v_padding_mask = self.vision_embedding(v)
        q = self.question_embedding(q)
        q_padding_mask = generate_padding_mask(q.unsqueeze(dim=1), padding_idx=self.padding_idx)

        # performing hierarchical feature extraction
        q = self.hierarchical_extractor(q)

        v, q = self.encoder(v, v_padding_mask, q, q_padding_mask)
        
        attended_v = self.vision_attr_reduce(v)
        attended_v = F.softmax(attended_v, dim=1)
        attended_q = self.text_attr_reduce(q)
        attended_q = F.softmax(attended_q, dim=1)

        weighted_v = (v * attended_v).sum(dim=1)
        weighted_q = (q * attended_q).sum(dim=1)

        weighted_q = weighted_q.unsqueeze(dim=1)
        combined = torch.cat([weighted_v, weighted_q], dim=1)
        combined_mask = torch.cat([v_padding_mask, q_padding_mask], dim=-1)
        combined = self.fusion(combined)
        combined = combined.masked_fill(combined_mask.squeeze(1).squeeze(1).unsqueeze(-1).bool(), value=0)
        combined = self.norm(combined)

        return combined, combined_mask
    def forward(self, input_features: InstanceList):
        combined, combined_mask = self.encoder_forward(input_features)

        answer_tokens = input_features.answer_tokens
        out = self.decoder(
            answer_tokens=answer_tokens,
            encoder_features=combined,
            encoder_attention_mask=combined_mask
        )

        return F.log_softmax(out, dim=-1)

    