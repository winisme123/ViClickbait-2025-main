import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModel, ViTModel

class ViClickbaitDataset(Dataset):
    """
    Dataset cho bài toán phân loại Clickbait đa phương thức tiếng Việt.
    Đọc trực tiếp từ train.csv, val.csv, hoặc test.csv đã được split chuẩn.
    """
    def __init__(self, dataframe, img_dir, tokenizer, processor, max_length=128):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = str(row['title'])
        
        # Tokenize văn bản tiêu đề
        text_encoding = self.tokenizer(
            title,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Xử lý hình ảnh thumbnail
        # Cột thumbnail_url có dạng data/images/article_xxxx_image.png hoặc url
        url_or_path = str(row['thumbnail_url'])
        clean_name = os.path.basename(url_or_path.split('?')[0])
        if not clean_name or not clean_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif')):
            clean_name = f"thumb_{idx}.jpg"

        img_path = os.path.join(self.img_dir, clean_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback nếu ảnh thiếu hoặc không mở được
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        image_encoding = self.processor(images=image, return_tensors="pt")

        # Gán nhãn nhị phân: 1.0 = clickbait, 0.0 = non-clickbait
        label = 1.0 if str(row['label']).strip().lower() == 'clickbait' else 0.0

        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': image_encoding['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class MultimodalClickbaitModel(nn.Module):
    """
    Kiến trúc baseline trước đây (1 chiều Text-to-Image Cross-Attention)
    dùng để tương thích với checkpoint cũ: multimodal_clickbait_model.pth
    """
    def __init__(self, phobert_name="vinai/phobert-base", vit_name="google/vit-base-patch16-224", num_heads=8):
        super(MultimodalClickbaitModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        self.image_encoder = ViTModel.from_pretrained(vit_name)
        
        text_hidden_size = self.text_encoder.config.hidden_size
        image_hidden_size = self.image_encoder.config.hidden_size
        
        if text_hidden_size != image_hidden_size:
            self.image_projection = nn.Linear(image_hidden_size, text_hidden_size)
        else:
            self.image_projection = nn.Identity()
            
        self.cross_attention = nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state 
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state
        image_features = self.image_projection(image_features)
        
        attn_output, _ = self.cross_attention(query=text_features, key=image_features, value=image_features)
        text_pooled = torch.mean(text_features, dim=1)
        multimodal_pooled = torch.mean(attn_output, dim=1)
        
        combined = torch.cat((text_pooled, multimodal_pooled), dim=1)
        out = self.classifier(combined)
        return out.squeeze(1)


class MultimodalCoAttentionModel(nn.Module):
    """
    Mô hình đề xuất chính của đề tài:
    - Nhánh Text: PhoBERT-base trích xuất đặc trưng tiêu đề.
    - Nhánh Image: ViT-base trích xuất patch embeddings thumbnail.
    - Bi-directional Cross-Attention (Co-Attention):
        1) Text-to-Image Cross-Attention: Text truy vấn không gian vùng ảnh.
        2) Image-to-Text Cross-Attention: Image truy vấn từ ngữ trong tiêu đề.
    - Phân loại: Fully Connected Network với Dropout để tránh overfitting.
    """
    def __init__(self, phobert_name="vinai/phobert-base", vit_name="google/vit-base-patch16-224", num_heads=8, dropout=0.3):
        super(MultimodalCoAttentionModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        self.image_encoder = ViTModel.from_pretrained(vit_name)

        text_dim = self.text_encoder.config.hidden_size    # 768
        image_dim = self.image_encoder.config.hidden_size  # 768

        # Projection layers nếu kích thước khác nhau
        self.proj_img_to_text = nn.Linear(image_dim, text_dim) if image_dim != text_dim else nn.Identity()
        self.proj_text_to_img = nn.Linear(text_dim, image_dim) if text_dim != image_dim else nn.Identity()

        # Co-Attention hai chiều
        self.cross_attn_text_to_img = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_img_to_text = nn.MultiheadAttention(embed_dim=image_dim, num_heads=num_heads, batch_first=True)

        # Lớp phân loại nhị phân (kết hợp text CLS + img CLS + 2 multimodal aligned features)
        combined_dim = text_dim * 4  # 768 * 4 = 3072
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # 1. Trích xuất văn bản qua PhoBERT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = text_outputs.last_hidden_state  # (B, seq_len, 768)
        # Sử dụng CLS token (token 0 - <s> của PhoBERT) làm đặc trưng đại diện câu chuẩn
        text_cls = text_hidden[:, 0, :]  # (B, 768)

        # 2. Trích xuất hình ảnh qua ViT
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_hidden = image_outputs.last_hidden_state  # (B, num_patches + 1, 768)
        # Token 0 của ViT là [CLS] token đại diện toàn ảnh
        image_cls = image_hidden[:, 0, :]  # (B, 768)

        # 3. Bi-directional Cross-Attention
        # a) Text query Image
        img_proj = self.proj_img_to_text(image_hidden)
        t2i_attn, _ = self.cross_attn_text_to_img(query=text_hidden, key=img_proj, value=img_proj)
        # Lấy vector pooled tại vị trí CLS sau tương tác đa phương thức
        t2i_pooled = t2i_attn[:, 0, :]  # (B, 768)

        # b) Image query Text
        txt_proj = self.proj_text_to_img(text_hidden)
        # Key padding mask: đảo ngược attention_mask (1 -> valid, 0 -> pad; PyTorch MultiheadAttention yêu cầu True cho pad)
        key_padding_mask = (attention_mask == 0)
        i2t_attn, _ = self.cross_attn_img_to_text(
            query=image_hidden, 
            key=txt_proj, 
            value=txt_proj,
            key_padding_mask=key_padding_mask
        )
        i2t_pooled = i2t_attn[:, 0, :]  # (B, 768)

        # 4. Dung hợp đặc trưng (Feature Fusion)
        combined = torch.cat([text_cls, image_cls, t2i_pooled, i2t_pooled], dim=1)  # (B, 3072)
        out = self.classifier(combined)
        return out.squeeze(1)
