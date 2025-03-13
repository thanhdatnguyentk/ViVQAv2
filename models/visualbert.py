import torch
from torch import nn
from torch.nn import init, functional as F

from utils.instance import Instance
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


@META_ARCHITECTURE.register()
class VisualBERT(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        
        self.device = torch.device(config.DEVICE)
        
        # Xây dựng embedding cho văn bản và hình ảnh
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        
        # Sử dụng một encoder duy nhất thay vì text_encoder & vision_encoder riêng biệt
        self.encoder = build_encoder(config.UNIFIED_ENCODER)
        
        # Classifier sử dụng token [CLS]
        self.classifier = Classifier(
            in_features=config.D_MODEL,  
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

    def forward(self, input_features: Instance):
        # Xử lý văn bản
        text_features, text_padding_mask = self.text_embedding(input_features.question)

        # Xử lý hình ảnh
        vision_features, vision_padding_mask = self.vision_embedding(input_features.region_features)

        # Kết hợp embeddings: [CLS] + Văn bản + Ảnh
        combined_features = torch.cat([text_features, vision_features], dim=1)
        combined_padding_mask = torch.cat([text_padding_mask, vision_padding_mask], dim=1)

        # Transformer Encoder
        encoded_features = self.encoder(features=combined_features, padding_mask=combined_padding_mask)

        # Lấy embedding từ token [CLS] (đầu tiên của chuỗi)
        cls_features = encoded_features[:, 0, :]

        # Dự đoán
        output = self.classifier(cls_features)
        return F.log_softmax(output, dim=-1)
