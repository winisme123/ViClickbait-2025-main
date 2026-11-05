"""Chọn ngưỡng quyết định trên validation rồi áp dụng một lần lên test (docs/protocol-v3.md, mục C).

    python -m scripts.tune_threshold --modes text image concat multimodal --seeds 42 43 44

Với mỗi checkpoint: tính xác suất trên val và test, chọn ngưỡng trong lưới 0,05–0,95
(bước 0,01) có Macro-F1 trên val cao nhất (bằng nhau thì chọn ngưỡng gần 0,5 nhất),
báo cáo kết quả test ở ngưỡng 0,5 và ở ngưỡng đã chọn. Kết quả: results/threshold/.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor

from clickbait.dataset import ViClickbaitDataset, load_split
from clickbait.metrics import binary_metrics
from clickbait.models import ALL_MODES, MODEL_CLASSES, MODES, PHOBERT_NAME, VIT_NAME, checkpoint_path

OUT_DIR = os.path.join('results', 'threshold')
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


@torch.no_grad()
def predict(model, df, mode, tokenizer, processor, device):
  loader = DataLoader(ViClickbaitDataset(df, tokenizer, processor, mode), batch_size=8)
  probs, labels = [], []
  for batch in loader:
    inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask', 'pixel_values') if k in batch}
    with autocast(device.type, enabled=device.type == 'cuda'):
      logits = model(**inputs)
    probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    labels.append(batch['label'].numpy())
  return np.concatenate(probs), np.concatenate(labels).astype(int)


def choose_threshold(probs, labels):
  scores = [f1_score(labels, (probs >= t).astype(int), average='macro', zero_division=0) for t in GRID]
  best = max(range(len(GRID)), key=lambda i: (round(scores[i], 10), -abs(GRID[i] - 0.5)))
  return float(GRID[best]), float(scores[best])


def main():
  if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--modes', nargs='+', default=MODES, choices=ALL_MODES)
  parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)
  processor = ViTImageProcessor.from_pretrained(VIT_NAME)
  os.makedirs(OUT_DIR, exist_ok=True)

  rows = []
  for mode in args.modes:
    for seed in args.seeds:
      ckpt = checkpoint_path(mode, seed)
      out_path = os.path.join(OUT_DIR, f'{mode}_seed{seed}.json')
      if not os.path.exists(ckpt):
        print(f'Bỏ qua {mode} seed {seed}: chưa có {ckpt}')
        continue
      if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
          r = json.load(f)
      else:
        model = MODEL_CLASSES[mode]()
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        model.to(device).eval()
        val_p, val_y = predict(model, load_split(seed, 'val'), mode, tokenizer, processor, device)
        thr, val_f1 = choose_threshold(val_p, val_y)
        # Tập test: dùng một lần với ngưỡng đã chốt từ val
        test_p, test_y = predict(model, load_split(seed, 'test'), mode, tokenizer, processor, device)
        r = {'mode': mode, 'seed': seed, 'threshold': thr, 'val_macro_f1_at_threshold': val_f1,
             'val_macro_f1_at_0.5': float(f1_score(val_y, (val_p >= 0.5).astype(int), average='macro')),
             'test_at_0.5': binary_metrics(test_y, (test_p >= 0.5).astype(int)),
             'test_at_threshold': binary_metrics(test_y, (test_p >= thr).astype(int))}
        with open(out_path, 'w', encoding='utf-8') as f:
          json.dump(r, f, ensure_ascii=False, indent=2)
        del model
        torch.cuda.empty_cache()
      rows.append({'mode': mode, 'seed': seed, 'threshold': r['threshold'],
                   'test_f1_0.5': r['test_at_0.5']['macro_f1'],
                   'test_f1_thr': r['test_at_threshold']['macro_f1']})
      print(f"{mode:<11} seed {seed}: ngưỡng {r['threshold']:.2f} | test Macro-F1 "
            f"{rows[-1]['test_f1_0.5']:.4f} → {rows[-1]['test_f1_thr']:.4f}")

  df = pd.DataFrame(rows)
  if not df.empty:
    df.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    print(df.groupby('mode')[['threshold', 'test_f1_0.5', 'test_f1_thr']].agg(['mean', 'std']).round(4))


if __name__ == '__main__':
  main()
