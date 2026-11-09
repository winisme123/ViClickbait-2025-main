import pytest

from clickbait.metrics import binary_metrics


def test_binary_metrics_known_values():
  # TN=2, FP=1, FN=1, TP=2
  m = binary_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
  assert m['confusion_matrix'] == [[2, 1], [1, 2]]
  assert m['accuracy'] == pytest.approx(4 / 6)
  assert m['per_class']['clickbait']['precision'] == pytest.approx(2 / 3)
  assert m['per_class']['clickbait']['support'] == 3
  assert m['macro_f1'] == pytest.approx(2 / 3)


def test_binary_metrics_single_class_predictions():
  # Mô hình sụp về một lớp: F1 lớp clickbait = 0, không lỗi chia cho 0
  m = binary_metrics([0, 0, 1, 1], [0, 0, 0, 0])
  assert m['per_class']['clickbait']['f1'] == 0
  assert m['confusion_matrix'] == [[2, 0], [2, 0]]
