"""Baseline cổ điển: TF-IDF + Logistic Regression trên tiêu đề (docs/protocol-v3.md, mục D).

Chạy trên CPU, vài giây mỗi seed:
    python -m scripts.baseline_tfidf --seeds 42 43 44 45 46 47 48 49

C của Logistic Regression được chọn theo Macro-F1 trên tập validation; tập test
chỉ dùng một lần với C đã chọn. Kết quả: results/tfidf/tfidf_seed{seed}.json
và results/tfidf/summary.csv.
"""
import argparse
import json
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from clickbait.dataset import load_split
from clickbait.metrics import binary_metrics
from clickbait.text_utils import segment_text

OUT_DIR = os.path.join('results', 'tfidf')
C_GRID = [0.01, 0.1, 1, 10, 100]


def to_xy(df):
  x = [segment_text(t) for t in df['title'].astype(str)]
  y = (df['label'] == 'clickbait').astype(int).to_numpy()
  return x, y


def run_seed(seed):
  (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (
      to_xy(load_split(seed, s)) for s in ('train', 'val', 'test'))
  # pyvi nối từ ghép bằng '_' nên tách token theo khoảng trắng là đủ
  vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        token_pattern=r'[^\s]+', lowercase=True)
  f_tr = vec.fit_transform(x_tr)
  f_va, f_te = vec.transform(x_va), vec.transform(x_te)

  val_scores = {}
  for c in C_GRID:
    clf = LogisticRegression(C=c, max_iter=2000).fit(f_tr, y_tr)
    val_scores[c] = f1_score(y_va, clf.predict(f_va), average='macro')
  best_c = max(C_GRID, key=lambda c: (val_scores[c], -abs(c - 1)))

  clf = LogisticRegression(C=best_c, max_iter=2000).fit(f_tr, y_tr)
  result = binary_metrics(y_te, clf.predict(f_te))
  result.update({'mode': 'tfidf', 'seed': seed, 'C': best_c,
                 'val_macro_f1_by_C': {str(k): v for k, v in val_scores.items()},
                 'n_features': f_tr.shape[1]})
  return result


def main():
  if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
  args = parser.parse_args()

  os.makedirs(OUT_DIR, exist_ok=True)
  rows = []
  for seed in args.seeds:
    r = run_seed(seed)
    with open(os.path.join(OUT_DIR, f'tfidf_seed{seed}.json'), 'w', encoding='utf-8') as f:
      json.dump(r, f, ensure_ascii=False, indent=2)
    rows.append({'seed': seed, 'C': r['C'], 'accuracy': r['accuracy'], 'macro_f1': r['macro_f1']})
    print(f"seed {seed}: C={r['C']:<5} accuracy={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")

  df = pd.DataFrame(rows)
  df.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
  print(f"Trung bình {len(df)} seed: accuracy {df.accuracy.mean():.4f} ± {df.accuracy.std():.4f}, "
        f"macro_f1 {df.macro_f1.mean():.4f} ± {df.macro_f1.std():.4f}")


if __name__ == '__main__':
  main()
