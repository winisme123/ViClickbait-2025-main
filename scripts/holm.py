"""Hiệu chỉnh Holm–Bonferroni cho các kiểm định Wilcoxon chính (docs/protocol-v4.md).

    python -m scripts.holm results/compare_v4_concat_vs_text.json results/compare_v4_concat_vs_image.json

Holm: xếp p tăng dần, so p thứ k (k = 0, 1, ...) với α / (m − k); dừng ở p đầu tiên
không đạt. Một giả thuyết "đạt" khi bị bác bỏ H0 và hiệu trung bình > 0 (α tổng = 0,05).
Kết quả: results/holm_v4.json (đổi bằng --out)
"""
import argparse
import json
import sys


def holm(pvalues, alpha=0.05):
  """Trả về list[bool] bác bỏ H0 theo Holm, cùng thứ tự với pvalues."""
  order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
  reject = [False] * len(pvalues)
  m = len(pvalues)
  for rank, i in enumerate(order):
    if pvalues[i] < alpha / (m - rank):
      reject[i] = True
    else:
      break
  return reject


def main():
  if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('files', nargs='+')
  parser.add_argument('--alpha', type=float, default=0.05)
  parser.add_argument('--out', default='results/holm_v4.json')
  args = parser.parse_args()

  comps = []
  for path in args.files:
    with open(path, encoding='utf-8') as f:
      comps.append(json.load(f))
  reject = holm([c['wilcoxon_p'] for c in comps], args.alpha)
  rows = []
  for c, r in zip(comps, reject):
    verdict = 'đạt' if r and c['mean_diff'] > 0 else 'chưa chứng minh được'
    rows.append({'comparison': f"{c['a']} vs {c['b']}", 'mean_diff': c['mean_diff'],
                 'bootstrap_ci95': c.get('bootstrap_ci95'), 'wins_a': c['wins_a'],
                 'n_seeds': c['n_seeds'], 'wilcoxon_p': c['wilcoxon_p'],
                 'holm_reject': r, 'verdict': verdict})
    print(f"{rows[-1]['comparison']}: hiệu {c['mean_diff']:+.4f}, thắng {c['wins_a']}/{c['n_seeds']}, "
          f"p={c['wilcoxon_p']:.4g}, Holm bác bỏ H0: {r} → {verdict}")
  with open(args.out, 'w', encoding='utf-8') as f:
    json.dump({'alpha': args.alpha, 'method': 'holm', 'results': rows}, f, ensure_ascii=False, indent=2)
  print(f'Đã lưu {args.out}')


if __name__ == '__main__':
  main()
