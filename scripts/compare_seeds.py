"""So sánh hai mode qua nhiều seed (docs/protocol-v3.md, mục A).

    python -m scripts.compare_seeds --a concat --b text --seeds 42 43 44 45 46 47 48 49

- Phân tích chính: Wilcoxon hạng có dấu, hai phía, trên hiệu Macro-F1 (a − b) theo seed.
- Phân tích phụ: số seed a thắng; McNemar (chính xác, hai phía) từng seed và gộp mọi seed.
Đúng/sai từng mẫu lấy từ results/{mode}_seed{seed}_errors.csv và data/splits/.
Kết quả: results/compare_{a}_vs_{b}.json
"""
import argparse
import json
import os
import sys

import pandas as pd
from scipy.stats import binomtest, wilcoxon

from clickbait.dataset import load_split

RESULTS_DIR = 'results'


def correct_ids(mode, seed, test_ids):
  errors = pd.read_csv(os.path.join(RESULTS_DIR, f'{mode}_seed{seed}_errors.csv'), encoding='utf-8-sig')
  wrong = set(errors['id'])
  assert wrong <= test_ids, f'{mode} seed {seed}: errors.csv có id ngoài tập test'
  return test_ids - wrong


def mcnemar(only_a, only_b):
  n = only_a + only_b
  return float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0


def main():
  if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--a', default='concat')
  parser.add_argument('--b', default='text')
  parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46, 47, 48, 49])
  args = parser.parse_args()

  per_seed, diffs = [], []
  pooled_a = pooled_b = 0
  for seed in args.seeds:
    with open(os.path.join(RESULTS_DIR, f'{args.a}_seed{seed}.json'), encoding='utf-8') as f:
      f1_a = json.load(f)['macro_f1']
    with open(os.path.join(RESULTS_DIR, f'{args.b}_seed{seed}.json'), encoding='utf-8') as f:
      f1_b = json.load(f)['macro_f1']
    test_ids = set(load_split(seed, 'test')['id'])
    ok_a, ok_b = correct_ids(args.a, seed, test_ids), correct_ids(args.b, seed, test_ids)
    only_a, only_b = len(ok_a - ok_b), len(ok_b - ok_a)
    pooled_a += only_a
    pooled_b += only_b
    diffs.append(f1_a - f1_b)
    per_seed.append({'seed': seed, f'macro_f1_{args.a}': f1_a, f'macro_f1_{args.b}': f1_b,
                     'diff': f1_a - f1_b, f'only_{args.a}_correct': only_a,
                     f'only_{args.b}_correct': only_b, 'mcnemar_p': mcnemar(only_a, only_b)})

  w = wilcoxon(diffs, alternative='two-sided')
  mean_diff = sum(diffs) / len(diffs)
  result = {
      'a': args.a, 'b': args.b, 'seeds': args.seeds, 'per_seed': per_seed,
      'mean_diff': mean_diff, 'wins_a': sum(d > 0 for d in diffs), 'n_seeds': len(diffs),
      'wilcoxon_statistic': float(w.statistic), 'wilcoxon_p': float(w.pvalue),
      'pooled_mcnemar': {f'only_{args.a}_correct': pooled_a, f'only_{args.b}_correct': pooled_b,
                         'p': mcnemar(pooled_a, pooled_b)},
      'verdict_protocol_v3': 'đạt' if mean_diff > 0 and w.pvalue < 0.05 else 'chưa chứng minh được',
  }
  out = os.path.join(RESULTS_DIR, f'compare_{args.a}_vs_{args.b}.json')
  with open(out, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

  print(pd.DataFrame(per_seed).round(4).to_string(index=False))
  print(f"\nTrung bình hiệu {mean_diff:+.4f}; {args.a} thắng {result['wins_a']}/{len(diffs)} seed")
  print(f"Wilcoxon: W={w.statistic:.1f}, p={w.pvalue:.4f}")
  print(f"McNemar gộp: {pooled_a} vs {pooled_b}, p={result['pooled_mcnemar']['p']:.4f}")
  print(f"Kết luận theo protocol v3: {result['verdict_protocol_v3']}\nĐã lưu {out}")


if __name__ == '__main__':
  main()
