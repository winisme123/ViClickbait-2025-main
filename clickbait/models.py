"""Kiến trúc mô hình dùng chung cho scripts/train.py, scripts/evaluate.py, app.py."""
import os

import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

PHOBERT_NAME = 'vinai/phobert-base'
VIT_NAME = 'google/vit-base-patch16-224'
CHECKPOINT_DIR = 'checkpoints'


def masked_mean(hidden_states, attention_mask):
  """Mean-pooling theo chuỗi văn bản, bỏ qua các token padding."""
  mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
  return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def make_classifier(in_features):
  """Bộ phân loại MLP dùng chung cho mọi mode."""
  return nn.Sequential(
      nn.Linear(in_features, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, 1),
  )


class TextOnlyClickbaitModel(nn.Module):
  """Chỉ sử dụng PhoBERT cho tiêu đề"""

  def __init__(self, phobert_name=PHOBERT_NAME):
    super().__init__()
    self.text_encoder = AutoModel.from_pretrained(phobert_name)
    self.classifier = make_classifier(self.text_encoder.config.hidden_size)

  def forward(self, input_ids, attention_mask, pixel_values=None):
    text_outputs = self.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask)
    pooled = masked_mean(text_outputs.last_hidden_state, attention_mask)
    return self.classifier(pooled).squeeze(1)


class ImageOnlyClickbaitModel(nn.Module):
  """Chỉ sử dụng ViT cho thumbnail"""

  def __init__(self, vit_name=VIT_NAME):
    super().__init__()
    self.image_encoder = ViTModel.from_pretrained(vit_name)
    self.classifier = make_classifier(self.image_encoder.config.hidden_size)

  def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
    image_outputs = self.image_encoder(pixel_values=pixel_values)
    pooled = torch.mean(image_outputs.last_hidden_state, dim=1)
    return self.classifier(pooled).squeeze(1)


class ConcatClickbaitModel(nn.Module):
  """Hợp nhất muộn đơn giản: nối mean-pooling PhoBERT + ViT, không cross-attention"""

  def __init__(self, phobert_name=PHOBERT_NAME, vit_name=VIT_NAME):
    super().__init__()
    self.text_encoder = AutoModel.from_pretrained(phobert_name)
    self.image_encoder = ViTModel.from_pretrained(vit_name)
    self.classifier = make_classifier(
        self.text_encoder.config.hidden_size
        + self.image_encoder.config.hidden_size)

  def forward(self, input_ids, attention_mask, pixel_values):
    text_features = self.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    image_features = self.image_encoder(
        pixel_values=pixel_values).last_hidden_state
    # Ảnh không có padding nên giữ mean-pooling thường
    combined = torch.cat(
        (masked_mean(text_features, attention_mask),
         torch.mean(image_features, dim=1)),
        dim=1)
    return self.classifier(combined).squeeze(1)


class MultimodalClickbaitModel(nn.Module):
  """Kết hợp PhoBERT + ViT + Cross-Modal Attention"""

  def __init__(self, phobert_name=PHOBERT_NAME, vit_name=VIT_NAME, num_heads=8):
    super().__init__()
    self.text_encoder = AutoModel.from_pretrained(phobert_name)
    self.image_encoder = ViTModel.from_pretrained(vit_name)

    text_hidden_size = self.text_encoder.config.hidden_size
    image_hidden_size = self.image_encoder.config.hidden_size

    if text_hidden_size != image_hidden_size:
      self.image_projection = nn.Linear(image_hidden_size, text_hidden_size)
    else:
      self.image_projection = nn.Identity()

    self.cross_attention = nn.MultiheadAttention(
        embed_dim=text_hidden_size, num_heads=num_heads, batch_first=True)
    self.classifier = make_classifier(text_hidden_size * 2)

  def forward(self, input_ids, attention_mask, pixel_values):
    text_outputs = self.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask)
    text_features = text_outputs.last_hidden_state

    image_outputs = self.image_encoder(pixel_values=pixel_values)
    image_features = self.image_projection(image_outputs.last_hidden_state)

    attn_output, _ = self.cross_attention(
        query=text_features, key=image_features, value=image_features)

    # attn_output có một vector cho mỗi token văn bản (query) nên dùng cùng mask
    text_pooled = masked_mean(text_features, attention_mask)
    multimodal_pooled = masked_mean(attn_output, attention_mask)

    combined = torch.cat((text_pooled, multimodal_pooled), dim=1)
    return self.classifier(combined).squeeze(1)


MODEL_CLASSES = {
    'text': TextOnlyClickbaitModel,
    'image': ImageOnlyClickbaitModel,
    'multimodal': MultimodalClickbaitModel,
    'concat': ConcatClickbaitModel,
    # Thí nghiệm B (docs/protocol-v3.md): cùng kiến trúc, đầu vào văn bản là tiêu đề + đoạn mở đầu
    'text_lead': TextOnlyClickbaitModel,
    'concat_lead': ConcatClickbaitModel,
}
MODES = ['text', 'image', 'multimodal', 'concat']  # 4 mô hình của bảng kết quả chính
EXTRA_MODES = ['text_lead', 'concat_lead']
ALL_MODES = MODES + EXTRA_MODES


def checkpoint_path(mode, seed):
  return os.path.join(CHECKPOINT_DIR, f'{mode}_seed{seed}.pth')
