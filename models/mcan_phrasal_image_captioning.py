import torch
from torch import nn
from torch.nn import functional as F

from models.base_transformer import BaseTransformer
from utils.instance import Instance
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.decoder_builder import build_decoder
from builders.model_builder import META_ARCHITECTURE

# Import để đăng ký (register) các module Phrasal vào hệ thống Builder
import models.phrasal_mcan.attentions  # noqa: F401
import models.phrasal_mcan.encoder      # noqa: F401

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

@META_ARCHITECTURE.register()
class PhrasalMCANCaptioning(BaseTransformer):
    """
    Phrasal MCAN Image Captioning Model.
    
    Kết hợp kiến trúc Image Captioning với các module Phrasal Encoder 
    để tận dụng thông tin cấu trúc ngữ nghĩa (phrasal scores) trong quá trình huấn luyện.
    """
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        # Xây dựng các Embedding
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        # Xây dựng Encoders (Sẽ sử dụng PhrasalEncoder và PhrasalGuidedAttentionEncoder dựa trên file config)
        self.self_encoder = build_encoder(config.SELF_ENCODER)
        self.guided_encoder = build_encoder(config.GUIDED_ENCODER)

        # Các lớp giảm chiều dữ liệu (Attention Reduction)
        self.vision_attr_reduce = MLP(config.VISION_ATTR_REDUCE)
        self.text_attr_reduce = MLP(config.TEXT_ATTR_REDUCE)

        # Các lớp Projection và Fusion
        self.vision_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.text_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        
        self.fusion = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)
        self.layer_norm = nn.LayerNorm(config.D_MODEL) # Dùng cho output fusion
        
        # Decoder để sinh caption
        self.decoder = build_decoder(config.DECODER, vocab)

    def forward(self, input_features: Instance):
        """
        Forward pass cho quá trình Training.
        Input bao gồm cả Image Features và Caption Tokens (Teacher Forcing).
        """
        # 1. Vision Embedding
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        # 2. Text Embedding
        caption_tokens = input_features.caption_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(caption_tokens)
        
        # 3. Phrasal Self-Encoder (Xử lý Text Features với Phrasal Attention)
        # Input: Text Features -> Output: Contextualized Text Features
        text_features = self.self_encoder(
            features=text_features,
            padding_mask=text_padding_mask
        )

        # 4. Phrasal Guided-Encoder (Xử lý Vision guided bởi Text)
        # Input: Vision (Queries) + Text (Keys/Values) -> Output: Guided Vision Features
        vision_features = self.guided_encoder(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        )

        # 5. Attention Reduction & Fusion
        # Tính toán trọng số attention để tổng hợp features
        attended_vision_features = self.vision_attr_reduce(vision_features)
        attended_vision_features = F.softmax(attended_vision_features, dim=1)
        
        attended_text_features = self.text_attr_reduce(text_features)
        attended_text_features = F.softmax(attended_text_features, dim=1)

        # Tổng hợp feature có trọng số
        weighted_vision_features = (vision_features * attended_vision_features).sum(dim=1)
        weighted_text_features = (text_features * attended_text_features).sum(dim=1)

        # Lưu ý: Trong kiến trúc MCAN Captioning gốc, Decoder thường nhận vào 
        # weighted_vision_features (hoặc fused features) làm encoder_features.
        
        # 6. Decoding
        output = self.decoder(
            answer_tokens=caption_tokens, # Teacher forcing input
            encoder_features=weighted_vision_features, # Features từ encoder truyền sang decoder
            encoder_attention_mask=vision_padding_mask # Mask tương ứng
        )
        
        return F.log_softmax(output, dim=-1)

    def encoder_forward(self, input_features: Instance):
        """
        Forward pass cho quá trình Inference (Generation).
        Chỉ xử lý phần hình ảnh để chuẩn bị cho Decoder chạy từng bước.
        """
        # Lấy features từ input (grid hoặc region features)
        features = input_features.region_features # Điều chỉnh tùy theo tên field trong Instance

        # Vision Embedding
        vision_features, vision_padding_mask = self.vision_embedding(features)

   
        # Projection/Fusion đơn giản (nếu cần thiết để khớp chiều với Decoder)
        fused_features = vision_features 
        fused_features = self.dropout(self.gelu(self.fusion(fused_features)))
        
        fused_padding_mask = vision_padding_mask

        return fused_features, fused_padding_mask