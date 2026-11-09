import numpy as np

from scripts.compare_seeds import bootstrap_ci
from scripts.holm import holm


def test_holm_step_down():
  assert holm([0.01, 0.04]) == [True, True]      # 0.01 < 0.025, rồi 0.04 < 0.05
  assert holm([0.03, 0.04]) == [False, False]    # 0.03 không < 0.025 → dừng
  assert holm([0.04, 0.001]) == [True, True]
  assert holm([0.2, 0.001]) == [False, True]


def test_bootstrap_ci_contains_mean_and_deterministic():
  d = [0.01, 0.02, -0.005, 0.015, 0.03]
  lo, hi = bootstrap_ci(d)
  assert lo <= np.mean(d) <= hi
  assert bootstrap_ci(d) == (lo, hi)
