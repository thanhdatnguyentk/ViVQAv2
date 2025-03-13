import torch
from torch import nn
from torch.nn import init, functional as F


from utils.instance import Instance
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

class CoAttention(nn.Module):
    def __init__(self, config):
        super(CoAttention, self).__init__()
        self.v_conv = nn.Linear(config.D_VISION, config.D_MODEL, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(config.D_LANGUAGE, config.D_MODEL)
        self.x_conv = nn.Linear(config.D_MODEL, config.GLIMPSES)

        self.drop = nn.Dropout(config.DROPOUT)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = q.permute(0, 2, 1)  # Đổi trật tự thành [64, 512, 32] để nội suy theo chiều 32
        q = F.interpolate(q, size=int(v.shape[1]), mode='linear', align_corners=False)  # Mở rộng lên 74
        q = q.permute(0, 2, 1)  # Đưa về lại [64, 74, 512]
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


@META_ARCHITECTURE.register()
class ViLBERT(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        
        self.device = torch.device(config.DEVICE)
        self.vocab = vocab
        
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        self.text_encoder = build_encoder(config.TEXT_ENCODER)
        self.vision_encoder = build_encoder(config.VISION_ENCODER)
        self.co_attention = CoAttention(config.CO_ATTENTION)

        self.text_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.vision_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.classifier = Classifier(
            in_features=config.CO_ATTENTION.GLIMPSES * config.CO_ATTENTION.D_VISION + config.CO_ATTENTION.D_LANGUAGE,
            mid_features=1024,
            out_features=vocab.total_answers,
            drop=0.5,
        )
        
        self.initialize()
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def apply_attention(self, input, attention):
        """ Apply any number of attention maps over the input. """
        n = input.shape[0]

        # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
        input = input.view(n, 1, -1, 512).permute(0, 1, 3, 2) # [n, 1, d_model, s]
        attention = attention.permute(0, -1, 1)
        attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
        weighted = attention * input # [n, g, v, s]
        weighted_mean = weighted.sum(dim=-1) # [n, g, v]
        
        return weighted_mean.view(n, -1)

    def forward(self, input_features: Instance):

        
        
        # Xử lý hình ảnh
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)
        vision_features = self.vision_encoder(features=vision_features, padding_mask=vision_padding_mask)
        
        # Xử lý văn bản
        text_features, text_padding_mask = self.text_embedding(input_features.question)
        text_features = self.text_encoder(features=text_features, padding_mask=text_padding_mask)
    
        # Co-Attention
        
        vision_features = vision_features / (vision_features.norm(p=2, dim=1, keepdim=True).expand_as(vision_features) + 1e-8)
        a = self.co_attention(vision_features, text_features)
        vision_features = self.apply_attention(vision_features, a)
        vision_features = vision_features.view(vision_features.size(0), -1)
        text_features = text_features.mean(dim=1)
        combined = torch.cat([vision_features, text_features], dim=1)

        # Tổng hợp đặc trư
        # Dự đoán
        output = self.classifier(combined)
        return F.log_softmax(output, dim=-1)
