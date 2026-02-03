import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
from models.constituent_mcan.mcan_constituent import ConstituentMCAN
from utils.instance import Instance

def visualize_constituents(question_tokens, break_probs, layer_idx=0, head_idx=None, save_path="constituent_heatmap.png"):
    """
    question_tokens: List of words (e.g., ['Con', 'mèo', 'màu', 'gì', '?'])
    break_probs: Tensor shape (Batch, Layers, Heads, Seq_Len, Seq_Len)
    layer_idx: Which layer of Constituent Encoder to view.
    head_idx: Which head to view. If None, average over heads.
    """
    # break_probs shape: [Batch, Layers, Heads, Seq_Len, Seq_Len]
    # Take first sample in batch
    attn_map = break_probs[0, layer_idx].detach().cpu()
    
    if head_idx is not None:
        attn_map = attn_map[head_idx]
    else:
        # Average over heads
        attn_map = attn_map.mean(dim=0)
    
    attn_map = attn_map.numpy()
    
    # Trim to token length
    seq_len = len(question_tokens)
    attn_map = attn_map[:seq_len, :seq_len]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(attn_map, xticklabels=question_tokens, yticklabels=question_tokens, cmap="viridis", annot=True, fmt=".2f")
    plt.title(f"Constituent Grouping (Layer {layer_idx}, {'Mean Heads' if head_idx is None else f'Head {head_idx}'})")
    plt.xlabel("Key")
    plt.ylabel("Query")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

def test_visualization():
    # Mock Config
    config = SimpleNamespace(
        DEVICE="cpu",
        DROPOUT=0.1,
        D_MODEL=512,
        CONSTITUENT_LAYERS=3,
        TEXT_EMBEDDING=SimpleNamespace(
            ARCHITECTURE="UsualEmbedding",
            D_MODEL=512,
            WORD_EMBEDDING=None,
            DROPOUT=0.1
        ),
        VISION_EMBEDDING=SimpleNamespace(
            ARCHITECTURE="FeatureEmbedding",
            D_MODEL=512,
            D_FEATURE=2048,
            DROPOUT=0.1
        ),
        SELF_ENCODER=SimpleNamespace(
            ARCHITECTURE="Encoder",
            LAYERS=2,
            D_MODEL=512,
            SELF_ATTENTION=SimpleNamespace(
                ARCHITECTURE="ScaledDotProductAttention",
                HEAD=8,
                D_MODEL=512,
                D_KEY=64,
                D_VALUE=64,
                D_FF=2048,
                DROPOUT=0.1,
                USE_AOA=False,
                CAN_BE_STATEFUL=False
            )
        ),
        GUIDED_ENCODER=SimpleNamespace(
            ARCHITECTURE="GuidedAttentionEncoder",
            LAYERS=2,
            D_MODEL=512,
            GUIDED_ATTENTION=SimpleNamespace(
                ARCHITECTURE="ScaledDotProductAttention",
                HEAD=8,
                D_MODEL=512,
                D_KEY=64,
                D_VALUE=64,
                D_FF=2048,
                DROPOUT=0.1,
                USE_AOA=False,
                CAN_BE_STATEFUL=False
            )
        ),
        VISION_ATTR_REDUCE=SimpleNamespace( D_MODEL=512, DROPOUT=0.1 ),
        TEXT_ATTR_REDUCE=SimpleNamespace( D_MODEL=512, DROPOUT=0.1 )
    )

    # Mock Vocab
    class MockVocab:
        def __init__(self):
            self.total_answers = 1000
            self.stoi = {"<pad>": 0}
            self.padding_token = "<pad>"
            self.padding_idx = 0
        def __len__(self):
            return 100
    
    vocab = MockVocab()

    # Instantiate Model
    print("Instantiating ConstituentMCAN...")
    model = ConstituentMCAN(config, vocab)
    model.eval()

    # Mock Input: "Con mèo này màu gì ?" (5 tokens + padding)
    tokens = ["Con", "mèo", "này", "màu", "gì", "?"]
    seq_len = 10 # Let's say max seq len is 10
    token_ids = torch.randint(1, 100, (1, seq_len))
    
    # Create input features
    input_features = Instance(
        region_features=torch.randn(1, 36, 2048),
        question_tokens=token_ids
    )

    # Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        logits, group_probs = model(input_features)
    
    print(f"Group probs shape: {group_probs.shape}") # (bs, layers, heads, s, s)

    # Visualize Layer 0
    visualize_constituents(tokens, group_probs, layer_idx=0, save_path="constituent_layer0.png")
    # Visualize Layer 1
    visualize_constituents(tokens, group_probs, layer_idx=1, save_path="constituent_layer1.png")
    # Visualize Layer 2
    visualize_constituents(tokens, group_probs, layer_idx=2, save_path="constituent_layer2.png")

if __name__ == "__main__":
    test_visualization()
