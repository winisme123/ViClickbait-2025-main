"""Gộp results/{mode}_seed{seed}.json thành results/summary.csv dạng mean ± std."""
import glob
import json
import os
import sys

import pandas as pd

from clickbait.models import MODES

RESULTS_DIR = 'results'
METRICS = [
    'accuracy', 'macro_f1',
    'precision_clickbait', 'recall_clickbait', 'f1_clickbait',
    'precision_non-clickbait', 'recall_non-clickbait', 'f1_non-clickbait',
    'train_time_sec',
]


def load_rows():
  rows = []
  for path in sorted(glob.glob(os.path.join(RESULTS_DIR, '*_seed*.json'))):
    with open(path, encoding='utf-8') as f:
      r = json.load(f)
    row = {'mode': r['mode'], 'seed': r['seed'], 'accuracy': r['accuracy'],
           'macro_f1': r['macro_f1'], 'train_time_sec': r.get('train_time_sec')}
    for cls, m in r['per_class'].items():
      for k in ('precision', 'recall', 'f1'):
        row[f'{k}_{cls}'] = m[k]
    rows.append(row)
  return pd.DataFrame(rows)


def main():
  if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
  df = load_rows()
  if df.empty:
    sys.exit(f'Chưa có file kết quả nào trong {RESULTS_DIR}/')

  summary = []
  for mode in [m for m in MODES if m in set(df['mode'])]:
    g = df[df['mode'] == mode]
    out = {'mode': mode, 'n_seeds': len(g),
           'seeds': ' '.join(str(s) for s in sorted(g['seed']))}
    for metric in METRICS:
      values = g[metric].astype(float)
      # Độ lệch chuẩn mẫu (ddof=1); chỉ có 1 seed thì không tính được std
      std = values.std(ddof=1) if len(values) > 1 else float('nan')
      fmt = '{:.1f} ± {:.1f}' if metric == 'train_time_sec' else '{:.4f} ± {:.4f}'
      out[metric] = fmt.format(values.mean(), std)
    summary.append(out)

  summary = pd.DataFrame(summary)
  out_path = os.path.join(RESULTS_DIR, 'summary.csv')
  summary.to_csv(out_path, index=False, encoding='utf-8-sig')
  print(summary.to_string(index=False))
  print(f'\nĐã lưu {out_path}')


if __name__ == '__main__':
  main()
