"""Chỉ số đánh giá phân loại nhị phân dùng cho scripts/evaluate.py."""
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

CLASS_NAMES = ['non-clickbait', 'clickbait']


def binary_metrics(y_true, y_pred):
  """Accuracy, P/R/F1 từng lớp, Macro-F1, ma trận nhầm lẫn (nhãn 0 = non-clickbait, 1 = clickbait)."""
  precision, recall, f1, support = precision_recall_fscore_support(
      y_true, y_pred, labels=[0, 1], zero_division=0)
  return {
      'n_samples': int(len(y_true)),
      'accuracy': float(accuracy_score(y_true, y_pred)),
      'macro_f1': float(f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0)),
      'per_class': {
          name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                 'f1': float(f1[i]), 'support': int(support[i])}
          for i, name in enumerate(CLASS_NAMES)
      },
      # Hàng = nhãn thật, cột = nhãn dự đoán, thứ tự [non-clickbait, clickbait]
      'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
  }
