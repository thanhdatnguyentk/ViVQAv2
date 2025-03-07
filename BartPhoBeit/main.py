from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BeitFeatureExtractor, BeitModel
import torch
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import tqdm
import json
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Đường dẫn đến thư mục chứa ảnh
image_path = "data/vivqa_v2/images"

# Tải BARTPho (Xử lý văn bản)
bartpho_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
bartpho_model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable")

# Tải BEiT-2 (Xử lý hình ảnh)
beit_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
beit_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

def preprocess_image(image_path):
    """Tiền xử lý ảnh và trích xuất đặc trưng bằng BEiT-2."""
    #rezie ảnh về kích thước 224x224
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    pixel_values = beit_feature_extractor(images=image, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        image_features = beit_model(pixel_values).last_hidden_state

    return image_features  # Tensor đặc trưng ảnh


def preprocess_text(question):
    """Tiền xử lý câu hỏi và trích xuất đặc trưng bằng BARTPho."""
    inputs = bartpho_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=50)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Dùng get_encoder() để lấy encoder đúng cách
    encoder = bartpho_model.get_encoder()
    
    with torch.no_grad():
        text_features = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    #normalize text_features -> 768
    linear = nn.Linear(1024, 768)
    text_features = linear(text_features) 
    return text_features, attention_mask # Tensor đặc trưng văn bản

class MultiHeadSelfAttention(nn.Module):
    """Lớp Attention học đặc trưng từ hình ảnh & văn bản."""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        return x

    
class VisionLanguageFNN(nn.Module):
    """Mô hình hợp nhất thông tin từ ảnh & văn bản."""
    def __init__(self, embed_dim, hidden_dim):
        super(VisionLanguageFNN, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LanguageFNN(nn.Module):
    """Mô hình mạng nơ-ron truyền thẳng cho xử lý văn bản."""
    def __init__(self, embed_dim, hidden_dim):
        super(LanguageFNN, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class VisionFNN(nn.Module):
    """Mô hình mạng nơ-ron truyền thẳng cho xử lý hình ảnh."""
    def __init__(self, embed_dim, hidden_dim):
        super(VisionFNN, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class VQA_Model(nn.Module):
    """Mô hình Vision-Language dựa trên BEiT-2 & BARTPho."""
    def __init__(self, embed_dim=768, hidden_dim=512, num_heads=8):
        super(VQA_Model, self).__init__()

        # Self Attention
        self.text_attention = MultiHeadSelfAttention(embed_dim, num_heads) # (b , l , e) 
        self.image_attention = MultiHeadSelfAttention(embed_dim, num_heads)

        # Feed-Forward Networks
        self.language_fnn = LanguageFNN(embed_dim, hidden_dim)
        self.vision_fnn = VisionFNN(embed_dim, hidden_dim)
        self.vl_fnn = VisionLanguageFNN(embed_dim, hidden_dim)  # Vision-Language Fusion

    def forward(self, image_features, text_features, labels=None):
        # Áp dụng Attention
        text_features = self.text_attention(text_features)
        image_features = self.image_attention(image_features)

        # Hợp nhất đặc trưng
        text_features = torch.cat((text_features, image_features), dim=1)
        image_features = torch.cat((image_features, text_features), dim=1)

        # Áp dụng FNN
        text_features = self.language_fnn(text_features)
        image_features = self.vision_fnn(image_features)

         # Áp dụng Attention
        text_features = self.text_attention(text_features)
        image_features = self.image_attention(image_features)

        # Hợp nhất đặc trưng
        combined_features = torch.cat((text_features, image_features), dim=1)

        output = self.vl_fnn(combined_features)
        output = self.language_fnn(output)
        # if labels is not None:
        #     loss = nn.CrossEntropyLoss()(output, labels)
        #     return output, loss
        return output
    

class ViVQADataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding="utf8") as file:
            json_data = json.load(file)
        self.annotations = self.load_annotations(json_data)

    
    def load_annotations(self, json_data):
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answer = ann["answers"]
                    question_id = ann["id"]
                    annotation = {
                        "question_id": question_id,
                        "image_id": ann["image_id"],
                        "file_name": image["file_name"],
                        "question": question,
                        "answer": answer
                    }
                    annotations.append(annotation)      
                
        return annotations

    def __getitem__(self, idx):
        item = self.annotations[idx]
       
        # Xử lý hình ảnh
        image_tensor = preprocess_image( os.path.join(image_path, item["file_name"]))
        
        
        # Xử lý câu hỏi
        input_ids, attention_mask = preprocess_text(item["question"])
        
        # Xử lý câu trả lời
        target_ids, attention_mask= preprocess_text(item["answer"])

        return {
            "image": image_tensor.squeeze(0),
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": target_ids.squeeze(0)
        }
    def __len__(self) -> int:
        return len(self.annotations)

def collate_fn(batch):
    image = [item["image"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    


    image = pad_sequence(image, batch_first=True)
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    return {
        "image": image,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for it, batch in enumerate(dataloader):
            optimizer.zero_grad()
        
            input_ids = batch["input_ids"].to(device)
            print(input_ids.shape)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            print(labels.shape)
            # outputs, loss = model(image_features=batch["image"].to(device), text_features=input_ids, labels=labels)
            outputs = model(image_features=batch["image"].to(device), text_features=input_ids)
            outputs = outputs.squeeze(0)
            labels = labels.squeeze(0)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(image_features=batch["image"], text_features=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(num_epochs = 5):
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss = evaluate(model, val_dataloader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

# df_train = pd.read_json("data/vivqa_v2/vivqa_v2_train.json")
# df_dev = pd.read_json("data/vivqa_v2/vivqa_v2_dev.json")
# df_test = pd.read_json("data/vivqa_v2/vivqa_v2_test.json")


train_dataset = ViVQADataset("data/vivqa_v2/vivqa_v2_train.json")
val_dataset = ViVQADataset("data/vivqa_v2/vivqa_v2_dev.json")
test_dataset = ViVQADataset("data/vivqa_v2/vivqa_v2_test.json")

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQA_Model().to(device)

# Loss function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

train(num_epochs=5)

# Lưu mô hình
model.save_pretrained("bartphobeit_vqa")

# Đánh giá trên tập test
test_loss = evaluate(model, test_dataloader, device)
print(f"Test Loss: {test_loss:.4f}")

