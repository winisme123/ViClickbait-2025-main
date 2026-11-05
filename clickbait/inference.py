"""Suy diễn cho demo: nạp checkpoint, dự đoán một tiêu đề + ảnh, heatmap attention.

Tiền xử lý giống hệt lúc huấn luyện (clickbait/dataset.py): tách từ pyvi,
độ dài tối đa MAX_LENGTH token, ảnh qua ViTImageProcessor.
"""
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, ViTModel

from .dataset import MAX_LENGTH
from .models import MODEL_CLASSES, PHOBERT_NAME, VIT_NAME, checkpoint_path
from .text_utils import normalize_text, segment_text

# Demo dùng checkpoint của seed 42
DEMO_SEED = 42


def load_pipeline(mode, seed=DEMO_SEED, device=None):
  """Nạp mô hình `mode` + tokenizer + processor.

  Trả về dict gồm model, tokenizer, processor, eager_vit (ViT dùng để vẽ
  heatmap, None với mode text), device, checkpoint và loaded (False nếu
  không tìm thấy checkpoint, khi đó mô hình mang trọng số ngẫu nhiên).
  """
  device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = MODEL_CLASSES[mode]()
  ckpt = checkpoint_path(mode, seed)
  loaded = os.path.exists(ckpt)
  if loaded:
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
  model.to(device).eval()

  eager_vit = None
  if hasattr(model, 'image_encoder'):
    # ViT bản eager để lấy attentions cho heatmap (sdpa không trả về attentions).
    # Dùng chung trọng số ViT đã fine-tune của mô hình.
    eager_vit = ViTModel.from_pretrained(VIT_NAME, attn_implementation='eager')
    eager_vit.load_state_dict(model.image_encoder.state_dict())
    eager_vit.to(device).eval()

  return {
      'mode': mode,
      'model': model,
      'tokenizer': AutoTokenizer.from_pretrained(PHOBERT_NAME),
      'processor': ViTImageProcessor.from_pretrained(VIT_NAME),
      'eager_vit': eager_vit,
      'device': device,
      'checkpoint': ckpt,
      'loaded': loaded,
  }


@torch.no_grad()
def predict_proba(pipeline, title, image):
  """Xác suất clickbait của một mẫu. `image` là PIL.Image hoặc None (mode text)."""
  mode, device = pipeline['mode'], pipeline['device']
  input_ids = attention_mask = pixel_values = None
  if mode != 'image':
    enc = pipeline['tokenizer'](
        segment_text(normalize_text(title)), max_length=MAX_LENGTH, padding='max_length',
        truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
  if mode != 'text':
    if image is None:
      raise ValueError(f'Mode {mode} cần ảnh thumbnail.')
    pixel_values = pipeline['processor'](
        images=image.convert('RGB'), return_tensors='pt')['pixel_values'].to(device)
  logit = pipeline['model'](input_ids, attention_mask, pixel_values)
  return torch.sigmoid(logit).item()


@torch.no_grad()
def attention_heatmap(pipeline, image, alpha=0.55):
  """Attention rollout trên ViT → ảnh 224×224, vùng đỏ là vùng mô hình tập trung."""
  eager_vit, device = pipeline['eager_vit'], pipeline['device']
  if eager_vit is None or image is None:
    return None
  pixel = pipeline['processor'](
      images=image.convert('RGB'), return_tensors='pt')['pixel_values'].to(device)
  out = eager_vit(pixel_values=pixel, output_attentions=True)
  n = out.attentions[0].shape[-1]
  eye = torch.eye(n, device=device)
  roll = eye
  for a in out.attentions:
    a = a.mean(dim=1) + eye                # trung bình các head + kết nối tắt
    a = a / a.sum(dim=-1, keepdim=True)    # chuẩn hoá theo hàng
    roll = roll @ a
  cls = roll[0, 0, 1:]                     # hàng của token CLS, bỏ chính CLS
  side = int(cls.numel() ** 0.5)           # 14×14 patch
  grid = cls.reshape(side, side)
  grid = grid / grid.max().clamp(min=1e-9)
  grid = F.interpolate(grid[None, None], size=(224, 224), mode='bilinear')[0, 0]
  mask = Image.fromarray((grid * 255).byte().cpu().numpy(), mode='L')
  mask = mask.point(lambda p: int(p * alpha))
  base = image.convert('RGB').resize((224, 224))
  red = Image.new('RGB', base.size, (255, 40, 40))
  return Image.composite(red, base, mask)
