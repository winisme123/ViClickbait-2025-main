"""Chạy lần lượt train + evaluate cho mọi mode x seed.

Mỗi lần chạy là một tiến trình riêng để giải phóng hoàn toàn bộ nhớ GPU.
Có thể dừng giữa chừng và chạy lại: bỏ qua lần đã có kết quả, và bỏ qua bước
train nếu checkpoint đã huấn luyện xong (có file .json đi kèm).

    python -m scripts.run_all
    python -m scripts.run_all --modes text concat --seeds 42
"""
import argparse
import os
import subprocess
import sys
import time

from clickbait.models import MODES, checkpoint_path

RESULTS_DIR = 'results'
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')


def run(cmd, log_path):
  """Chạy lệnh, ghi toàn bộ output vào log; dừng hẳn nếu lệnh lỗi."""
  print(f'  $ {" ".join(cmd)}  (log: {log_path})', flush=True)
  start = time.perf_counter()
  env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
  with open(log_path, 'w', encoding='utf-8') as log:
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
  print(f'    xong sau {time.perf_counter() - start:.0f} s, mã thoát {proc.returncode}',
        flush=True)
  if proc.returncode != 0:
    sys.exit(f'Lỗi khi chạy {" ".join(cmd)} — xem {log_path}')


def main():
  parser = argparse.ArgumentParser(description='Chạy hàng loạt mode x seed')
  parser.add_argument('--modes', nargs='+', default=MODES, choices=MODES)
  parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
  args = parser.parse_args()

  os.makedirs(LOGS_DIR, exist_ok=True)
  for seed in args.seeds:
    for mode in args.modes:
      name = f'{mode}_seed{seed}'
      print(f'=== {name}', flush=True)
      if os.path.exists(os.path.join(RESULTS_DIR, f'{name}.json')):
        print('  đã có kết quả, bỏ qua')
        continue

      # train.py ghi .json sau .pth, nên có .json nghĩa là đã huấn luyện xong
      ckpt = checkpoint_path(mode, seed)
      if os.path.exists(ckpt.replace('.pth', '.json')):
        print(f'  đã có checkpoint {ckpt}, bỏ qua huấn luyện')
      else:
        run([sys.executable, '-m', 'scripts.train', '--mode', mode, '--seed', str(seed)],
            os.path.join(LOGS_DIR, f'{name}_train.log'))

      run([sys.executable, '-m', 'scripts.evaluate', '--mode', mode, '--seed', str(seed)],
          os.path.join(LOGS_DIR, f'{name}_eval.log'))

  print('Hoàn tất. Chạy python -m scripts.summarize để gộp kết quả.')


if __name__ == '__main__':
  main()
