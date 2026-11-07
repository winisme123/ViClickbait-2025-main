"""Kiểm chứng rò rỉ dữ liệu của checkpoint multimodal_clickbait_model.pth (bản trên Drive).

Không cần GPU. Đầu vào là drive_ckpt_preds.csv: xác suất mà checkpoint đó dự đoán cho
toàn bộ 3.414 mẫu (tạo bằng evaluate.py --model_type legacy --test_csv <file 3.414 mẫu>).

Chạy từ thư mục experiments/coattention_80_10_10:
    python reproduction/verify_leakage.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))

preds = pd.read_csv(os.path.join(HERE, 'drive_ckpt_preds.csv'))
preds['y'] = (preds['label'] == 'clickbait').astype(int)
preds['pred'] = (preds['prob'] >= 0.5).astype(int)
preds = preds.set_index('id')


def score(ids):
  d = preds.loc[list(ids)]
  return len(d), accuracy_score(d['y'], d['pred']) * 100, f1_score(d['y'], d['pred'], average='macro') * 100


print('1) Điểm của checkpoint trên từng tập do split_data.py tạo ra')
print(f"   {'Tập':<12}{'n':>6}{'Accuracy':>10}{'Macro-F1':>10}")
for name in ['train.csv', 'val.csv', 'test.csv']:
  n, acc, f1 = score(pd.read_csv(name)['id'])
  print(f'   {name:<12}{n:>6}{acc:>9.2f}%{f1:>9.2f}%')
n, acc, f1 = score(preds.index)
print(f"   {'toàn bộ':<12}{n:>6}{acc:>9.2f}%{f1:>9.2f}%")
print('   → Một mô hình chỉ học train.csv phải có điểm trên val/test thấp hơn rõ rệt so với train.')

print('\n2) Số mẫu test.csv rơi vào phần train nếu checkpoint được huấn luyện bằng')
print('   random_split 80/20 trên toàn bộ 3.414 mẫu (cách của train.py gốc, commit efd89c8)')
full = pd.read_csv('clickbait_dataset_with_splits.csv')['id'].tolist()
test_ids = set(pd.read_csv('test.csv')['id'])
n_train = int(0.8 * len(full))
overlaps = []
for seed in range(1000):
  perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed)).tolist()
  train_ids = {full[i] for i in perm[:n_train]}
  overlaps.append(len(test_ids & train_ids))
overlaps = np.array(overlaps)
print(f'   1.000 cách chia ngẫu nhiên: trung bình {overlaps.mean():.1f}/{len(test_ids)} mẫu '
      f'({overlaps.mean() / len(test_ids):.0%}), thấp nhất {overlaps.min()}, cao nhất {overlaps.max()}')
print('   → Dù seed là gì, khoảng 80% tập test đã nằm trong dữ liệu huấn luyện của checkpoint.')
