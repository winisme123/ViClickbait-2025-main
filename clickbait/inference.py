"""Suy diễn cho demo: nạp checkpoint, dự đoán một tiêu đề + ảnh, heatmap attention.

Tiền xử lý giống hệt lúc huấn luyện (clickbait/dataset.py): tách từ pyvi,
độ dài tối đa MAX_LENGTH token, ảnh qua ViTImageProcessor.
"""
import math
import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, ViTModel

from .dataset import MAX_LENGTH
from .models import MODEL_CLASSES, PHOBERT_NAME, VIT_NAME, checkpoint_path
from .text_utils import normalize_text, segment_text

# Demo dùng checkpoint của seed 42
DEMO_SEED = 42
# Giải thích thay tạm image_encoder của mô hình dùng chung, nên khóa khi suy diễn
_LOCK = threading.Lock()


def load_pipeline(mode, seed=DEMO_SEED, device=None):
  """Nạp mô hình `mode` + tokenizer + processor.

  Trả về dict gồm model, tokenizer, processor, eager_vit (ViT dùng để vẽ
  heatmap, None với mode text), device, checkpoint và loaded (False nếu
  không tìm thấy checkpoint, khi đó mô hình mang trọng số ngẫu nhiên).
  Thiết bị: tham số `device`, hoặc biến môi trường CLICKBAIT_DEVICE, hoặc GPU nếu có.
  """
  # CLICKBAIT_DEVICE=cpu ép chạy CPU (ví dụ khi GPU đang bận huấn luyện)
  device = device or torch.device(
      os.environ.get('CLICKBAIT_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))
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
  with _LOCK:
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


def _encode_texts(pipeline, texts):
  enc = pipeline['tokenizer'](texts, max_length=MAX_LENGTH, padding='max_length',
                              truncation=True, return_tensors='pt')
  return enc['input_ids'].to(pipeline['device']), enc['attention_mask'].to(pipeline['device'])


class _CachedImageEncoder(torch.nn.Module):
  """Thay tạm image_encoder: trả lại đặc trưng ảnh đã tính sẵn cho cả batch."""

  def __init__(self, hidden):
    super().__init__()
    self.hidden = hidden

  def forward(self, pixel_values):
    return SimpleNamespace(last_hidden_state=self.hidden.expand(pixel_values.shape[0], -1, -1))


@contextmanager
def _image_features_cached(model, pixel_values):
  """Chạy ViT một lần cho ảnh, dùng lại cho mọi biến thể tiêu đề (giải thích nhanh hơn nhiều)."""
  if pixel_values is None or not hasattr(model, 'image_encoder'):
    yield
    return
  original = model.image_encoder
  model.image_encoder = _CachedImageEncoder(original(pixel_values=pixel_values).last_hidden_state)
  try:
    yield
  finally:
    model.image_encoder = original


@torch.no_grad()
def _batch_proba(pipeline, segmented_texts, pixel_values, batch_size=32):
  """Xác suất clickbait cho nhiều tiêu đề (đã tách từ) cùng một ảnh."""
  probs = []
  model = pipeline['model']
  with _LOCK, _image_features_cached(model, pixel_values):
    for i in range(0, len(segmented_texts), batch_size):
      chunk = segmented_texts[i:i + batch_size]
      ids, mask = _encode_texts(pipeline, chunk)
      pix = None if pixel_values is None else pixel_values.expand(len(chunk), -1, -1, -1)
      probs.extend(torch.sigmoid(model(ids, mask, pix)).float().cpu().tolist())
  return probs


def _pixel_values(pipeline, image):
  if pipeline['mode'] == 'text':
    return None
  if image is None:
    raise ValueError(f"Mode {pipeline['mode']} cần ảnh thumbnail.")
  return pipeline['processor'](images=image.convert('RGB'), return_tensors='pt')['pixel_values'].to(pipeline['device'])


def _logit(p, eps=1e-6):
  p = min(max(p, eps), 1 - eps)
  return math.log(p / (1 - p))


@torch.no_grad()
def word_importance(pipeline, title, image=None, n_permutations=10, seed=0):
  """Đóng góp của từng từ vào quyết định, ước lượng giá trị Shapley bằng hoán vị.

  Với mỗi hoán vị ngẫu nhiên của các từ, thêm dần từng từ vào một tiêu đề rỗng
  (giữ nguyên thứ tự gốc của các từ đã có) và ghi lại log-odds tăng thêm bao
  nhiêu; đóng góp của một từ là trung bình qua các hoán vị. Khác với bỏ từng từ
  một, cách này chia đều ảnh hưởng cho các từ cùng mang một tín hiệu (ví dụ
  "SỐC", "sự thật", "kinh hoàng" cùng chỉ giật tít). Tổng đóng góp bằng
  logit(P tiêu đề) − logit(P tiêu đề rỗng).

  Trả về (xác suất gốc, list[(từ, đóng góp log-odds)]) theo thứ tự trong tiêu đề.
  Đóng góp > 0: kéo về phía clickbait; < 0: kéo về phía chính thống. Từ ghép
  của pyvi (nối bằng '_') tính là một từ, trả về với dấu cách.
  """
  words = segment_text(normalize_text(title)).split()
  n = len(words)
  pixel_values = _pixel_values(pipeline, image)
  rng = np.random.default_rng(seed)
  perms = []
  for k in range(n_permutations):
    # Cặp hoán vị xuôi/ngược giúp giảm phương sai ước lượng
    perms.append(perms[-1][::-1] if k % 2 else rng.permutation(n))
  subsets = {frozenset(perm[:j]) for perm in perms for j in range(n + 1)}
  subsets = sorted(subsets, key=lambda sub: (len(sub), sorted(sub)))
  texts = [' '.join(words[i] for i in sorted(sub)) for sub in subsets]
  logits = dict(zip(subsets, (_logit(p) for p in _batch_proba(pipeline, texts, pixel_values))))
  phi = np.zeros(n)
  for perm in perms:
    for j in range(n):
      phi[perm[j]] += logits[frozenset(perm[:j + 1])] - logits[frozenset(perm[:j])]
  phi /= len(perms)
  base = 1 / (1 + math.exp(-logits[frozenset(range(n))]))
  return base, [(w.replace('_', ' '), float(v)) for w, v in zip(words, phi)]


@torch.no_grad()
def image_contribution(pipeline, title, image):
  """Thay ảnh bằng ảnh xám trung tính (pixel 0 sau chuẩn hóa) và đo xác suất thay đổi.

  Trả về (P với ảnh thật, P với ảnh xám), hoặc None nếu mô hình không dùng ảnh.
  """
  if pipeline['mode'] == 'text' or image is None:
    return None
  pixel_values = _pixel_values(pipeline, image)
  text = [segment_text(normalize_text(title))]
  with_img = _batch_proba(pipeline, text, pixel_values)[0]
  without = _batch_proba(pipeline, text, torch.zeros_like(pixel_values))[0]
  return with_img, without
