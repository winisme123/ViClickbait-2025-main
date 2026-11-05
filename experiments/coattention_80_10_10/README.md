# Thử nghiệm Co-Attention, chia 80/10/10

Đây là nhánh thử nghiệm riêng của nhóm, giữ nguyên mã nguồn ban đầu. Pipeline chính của khóa luận (chia 70/15/15, 4 mô hình × 3 seed) nằm ở `clickbait/` và `scripts/`. Xem README ở thư mục gốc và `docs/experiments.md`.

| File | Nội dung |
|---|---|
| `split_data.py` | Chia 80/10/10 phân tầng (seed 42) → `train.csv`, `val.csv`, `test.csv`, `clickbait_dataset_with_splits.csv` |
| `models.py` | `ViClickbaitDataset`, `MultimodalClickbaitModel` (cross-attention một chiều), `MultimodalCoAttentionModel` (cross-attention hai chiều) |
| `train.py` | Huấn luyện `MultimodalCoAttentionModel`: 4 epoch, `pos_weight` cân bằng nhãn, lr 2e-5 cho encoder và 1e-4 cho phần hợp nhất, warmup 10% |
| `evaluate.py` | Đánh giá mô hình Co-Attention |
| `run_baselines.py` | Huấn luyện + đánh giá 3 baseline (Text-only, Image-only, Late Fusion) và đo checkpoint Co-Attention của `train.py` trên cùng `test.csv` → `ablation_study_results.csv` |
| `download_images.py` | Script tải ảnh ban đầu (đã được thay bằng `scripts/download_images.py`) |
| `reproduction/` | Log kiểm chứng kết quả và `verify_leakage.py` |

Bản demo Streamlit của thư mục này đã chuyển ra `app.py` ở thư mục gốc và trở thành demo chính.

## Kết quả

`ablation_study_results.csv`, tập test 342 mẫu, seed 42:

| Mô hình | Accuracy | Macro-F1 |
|---|---|---|
| Text-only (PhoBERT) | 80,41% | 77,50% |
| Image-only (ViT) | 69,01% | 63,58% |
| Late Fusion (Concat) | 81,87% | 79,61% |
| Multimodal Co-Attention (Đề xuất) | 81,58% | 79,23% |

## Lưu ý về kết quả

Bản trước của `run_baselines.py` **nhập tay** dòng "Multimodal Attention (Đề xuất)" với Accuracy 94,15% / Macro-F1 92,93%. Con số đó đo bằng checkpoint `multimodal_clickbait_model.pth`, được huấn luyện bằng `train.py` gốc (chia ngẫu nhiên 80/20 trên toàn bộ dữ liệu) **trước khi** `test.csv` được tạo. Vì vậy khoảng 80% mẫu `test.csv` nằm trong dữ liệu huấn luyện của checkpoint đó. Bằng chứng: `reproduction/NOTES.md`.

`run_baselines.py` hiện đo thật checkpoint do `train.py` tạo ra (chỉ học `train.csv`, chọn theo `val.csv`). `train.py` và `run_baselines.py` cố định seed 42.

## Chạy

Các script đọc dữ liệu theo đường dẫn tương đối, nên cần chạy từ **trong thư mục này**, với ảnh nằm ở `images/`:

```powershell
cd experiments\coattention_80_10_10
New-Item -ItemType Junction -Path images -Target ..\..\data\images   # một lần, dùng chung ảnh
python train.py                        # → best_multimodal_coattention_model.pth
python run_baselines.py                # → ablation_study_results.csv
python reproduction\verify_leakage.py  # kiểm tra rò rỉ dữ liệu, không cần GPU
```

## Khác biệt so với pipeline chính

| | Thư mục này | Pipeline chính |
|---|---|---|
| Chia tập | 80/10/10, 1 seed, toàn bộ 3.414 mẫu | 70/15/15, 3 seed, 3.197 mẫu hợp lệ |
| Ảnh thiếu | Thay bằng ảnh đen | Loại mẫu khỏi dataset |
| Tiền xử lý tiêu đề | Không tách từ | Tách từ bằng pyvi (chuẩn PhoBERT) |

Do khác cách chia tập và tập mẫu, kết quả ở đây **không so sánh trực tiếp** được với `results/summary.csv`.
