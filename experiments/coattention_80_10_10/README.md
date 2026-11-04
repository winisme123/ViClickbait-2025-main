# Thử nghiệm Co-Attention, chia 80/10/10

Đây là nhánh thử nghiệm riêng của nhóm, giữ nguyên mã nguồn ban đầu. Pipeline chính của khóa luận (chia 70/15/15, 4 mô hình × 3 seed) nằm ở `clickbait/` và `scripts/`. Xem README ở thư mục gốc và `docs/experiments.md`.

| File | Nội dung |
|---|---|
| `split_data.py` | Chia 80/10/10 phân tầng (seed 42) → `train.csv`, `val.csv`, `test.csv`, `clickbait_dataset_with_splits.csv` |
| `models.py` | `ViClickbaitDataset`, `MultimodalClickbaitModel` (cross-attention một chiều), `MultimodalCoAttentionModel` (cross-attention hai chiều) |
| `train.py` | Huấn luyện `MultimodalCoAttentionModel`: 4 epoch, `pos_weight` cân bằng nhãn, lr 2e-5 cho encoder và 1e-4 cho phần hợp nhất, warmup 10% |
| `evaluate.py` | Đánh giá mô hình Co-Attention |
| `run_baselines.py` | Huấn luyện + đánh giá 3 baseline: Text-only, Image-only, Late Fusion → `ablation_study_results.csv` |
| `app.py` | Bản demo Streamlit ban đầu |
| `download_images.py` | Script tải ảnh ban đầu (đã được thay bằng `scripts/download_images.py`) |

## Chạy

Các script đọc dữ liệu theo đường dẫn tương đối, nên cần chạy từ **trong thư mục này**, với ảnh nằm ở `images/`:

```powershell
cd experiments\coattention_80_10_10
python train.py
python run_baselines.py
```

## Khác biệt so với pipeline chính

| | Thư mục này | Pipeline chính |
|---|---|---|
| Chia tập | 80/10/10, 1 seed, toàn bộ 3.414 mẫu | 70/15/15, 3 seed, 3.197 mẫu hợp lệ |
| Ảnh thiếu | Thay bằng ảnh đen | Loại mẫu khỏi dataset |
| Tiền xử lý tiêu đề | Không tách từ | Tách từ bằng pyvi (chuẩn PhoBERT) |

Do khác cách chia tập và tập mẫu, kết quả ở đây **không so sánh trực tiếp** được với `results/summary.csv`.
