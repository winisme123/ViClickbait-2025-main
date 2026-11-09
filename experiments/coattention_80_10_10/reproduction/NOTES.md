# Kiểm chứng kết quả Co-Attention 80/10/10

Ngày 04–05/11/2026, máy RTX 3060 Laptop 6 GB. Thư mục này lưu bằng chứng cho mục "Lưu ý về kết quả" trong `../README.md`.

## 1. Con số 92,93% bị rò rỉ dữ liệu

Bản trước của `run_baselines.py` nhập tay dòng "Multimodal Attention (Đề xuất)": Accuracy 94,15%, Macro-F1 92,93%. Số này đo bằng checkpoint `multimodal_clickbait_model.pth` (bản trên Google Drive của nhóm), kiến trúc `MultimodalClickbaitModel`.

Checkpoint này được tạo bởi `train.py` gốc (commit `efd89c8`, 10/09/2026). Script đó chia `random_split` 80/20 trên **toàn bộ** 3.414 mẫu, không cố định seed. `test.csv` được `split_data.py` tạo sau (commit `c07e2db`, 13/10/2026) bằng một lần chia khác.

| File | Nội dung |
|---|---|
| `eval_drive_ckpt_test.log` | `evaluate.py` trên `test.csv`: Accuracy 92,98%, Macro-F1 91,42% (tái lập được số đã báo) |
| `eval_drive_ckpt_full.log` | `evaluate.py` trên toàn bộ 3.414 mẫu: Macro-F1 90,73% |
| `drive_ckpt_preds.csv` | Xác suất checkpoint dự đoán cho từng mẫu trong 3.414 mẫu |
| `verify_leakage.py` | Tính điểm theo từng tập và mức trùng lặp train/test (không cần GPU) |

Kết quả của `python reproduction/verify_leakage.py` (chạy từ `experiments/coattention_80_10_10`):

```
Tập              n  Accuracy  Macro-F1
train.csv     2731    92.16%    90.29%
val.csv        341    94.72%    93.56%
test.csv       342    92.98%    91.42%
toàn bộ       3414    92.50%    90.73%

1.000 cách chia ngẫu nhiên: trung bình 273.6/342 mẫu (80%), thấp nhất 253, cao nhất 294
```

Điểm trên train, val và test ngang nhau, và khoảng 80% mẫu test đã nằm trong dữ liệu huấn luyện của checkpoint. Vì vậy 92,93% không phải điểm trên tập test độc lập.

Ghi chú kỹ thuật: bản `transformers` mới đổi tên khóa trọng số của ViT (`attention.attention.query` → `attention.q_proj`, …). Để nạp được checkpoint, cần đổi tên 192 khóa (giữ nguyên giá trị) rồi nạp với `strict=True`.

## 2. Đo lại đúng cách

| File | Nội dung |
|---|---|
| `unseeded_train_coattention.log`, `unseeded_eval_coattention.log` | Chạy nguyên `train.py` + `evaluate.py` của commit `c07e2db` (chưa cố định seed): Macro-F1 **76,04%** |
| `train_coattention.log` | `train.py` sau khi cố định seed 42: checkpoint tốt nhất ở epoch 3 (val Macro-F1 77,25%) |
| `run_baselines.log` | `run_baselines.py` sau khi sửa: 3 baseline seed 42 + **đo thật** checkpoint Co-Attention trên `test.csv` |

Kết quả cuối (`../ablation_study_results.csv`), tập test 342 mẫu:

| Mô hình | Accuracy | Macro-F1 |
|---|---|---|
| Text-only (PhoBERT) | 80,41% | 77,50% |
| Image-only (ViT) | 69,01% | 63,58% |
| Late Fusion (Concat) | 81,87% | **79,61%** |
| Multimodal Co-Attention (Đề xuất) | 81,58% | 79,23% |

Chênh lệch giữa hai lần chạy Co-Attention (76,04% không seed và 79,23% có seed) cho thấy với 1 lần chia, kết quả dao động vài điểm theo khởi tạo. Đây là lý do pipeline chính báo cáo trung bình qua 3 seed.
