# ViClickbait Detector — Phát hiện clickbait đa phương thức bằng học sâu

Khóa luận tốt nghiệp **CNTT-KLCN298**: phát hiện tiêu đề báo điện tử tiếng Việt mang tính *clickbait* bằng cách kết hợp **tiêu đề** (PhoBERT) và **ảnh thumbnail** (ViT) theo hướng **hợp nhất muộn** (late fusion), huấn luyện và đánh giá trên bộ dữ liệu **ViClickbait-2025**.

| | |
|---|---|
| Bài toán | Phân loại nhị phân `clickbait` / `non-clickbait` |
| Văn bản | [`vinai/phobert-base`](https://huggingface.co/vinai/phobert-base), tách từ bằng `pyvi`, masked mean pooling |
| Hình ảnh | [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224), mean pooling |
| Hợp nhất | `concat` (nối đặc trưng) và `multimodal` (cross-attention văn bản → ảnh) |
| Dữ liệu | 3.197 mẫu hợp lệ / 3.414 mẫu gốc, chia 70/15/15 phân tầng, 3 seed |
| Đánh giá | Accuracy, Precision, Recall, F1 từng lớp, **Macro-F1**, ma trận nhầm lẫn |
| Demo | Ứng dụng web Streamlit: dán link bài báo → dự đoán + giải thích |

## Kết quả chính

Macro-F1 trên tập **test**, trung bình ± độ lệch chuẩn qua 3 seed (42, 43, 44). Chi tiết: [docs/experiments.md](docs/experiments.md).

| Mô hình | Đầu vào | Accuracy | Macro-F1 | F1 (clickbait) |
|---|---|---|---|---|
| `text` | Tiêu đề | 0,8299 ± 0,0268 | 0,8029 ± 0,0346 | 0,7303 ± 0,0520 |
| `image` | Ảnh | 0,7410 ± 0,0229 | 0,6675 ± 0,0456 | 0,5116 ± 0,0810 |
| `concat` | Tiêu đề + ảnh | **0,8444 ± 0,0182** | **0,8194 ± 0,0209** | **0,7521 ± 0,0289** |
| `multimodal` | Tiêu đề + ảnh (cross-attention) | 0,8410 ± 0,0219 | 0,8087 ± 0,0315 | 0,7304 ± 0,0499 |

## Cấu trúc thư mục

```
ClickBaitKLTN/
├── app.py                     # Demo chính (streamlit run app.py)
├── app_v2.py                  # Demo phân tích: so sánh 3 mô hình, ngưỡng, lịch sử
├── clickbait/                 # Thư viện dùng chung
│   ├── dataset.py             #   mẫu hợp lệ, chia tập 70/15/15, Dataset PyTorch
│   ├── models.py              #   4 kiến trúc: text, image, concat, multimodal
│   ├── inference.py           #   nạp checkpoint, dự đoán, heatmap cho demo
│   ├── metrics.py             #   Accuracy / P / R / F1 / Macro-F1 / ma trận nhầm lẫn
│   ├── text_utils.py          #   tách từ tiếng Việt (pyvi) cho PhoBERT
│   ├── scraper.py             #   lấy tiêu đề + ảnh bìa từ link bài báo
│   └── signals.py             #   dấu hiệu clickbait theo luật (để giải thích trên demo)
├── scripts/                   # Các bước của pipeline (python -m scripts.<tên>)
│   ├── download_images.py     #   1. tải ảnh thumbnail
│   ├── prepare_data.py        #   2. lọc mẫu hợp lệ
│   ├── train.py               #   3. huấn luyện một mode × seed
│   ├── evaluate.py            #   4. đánh giá trên tập test (một lần)
│   ├── run_all.py             #   3+4 cho mọi mode × seed
│   └── summarize.py           #   5. gộp kết quả mean ± std
├── data/
│   ├── text/                  # dataset gốc + valid_samples.csv + exclusion_review.csv
│   ├── splits/                # split_seed{42,43,44}.csv (id → train/val/test)
│   └── images/                # ảnh thumbnail (tự tải, không đưa lên git)
├── checkpoints/               # *.pth (không đưa lên git) + *.json meta huấn luyện
├── results/                   # kết quả test từng lượt, summary.csv
│   └── v1_no_warmup/          # lượt chạy đối chứng không warmup
├── experiments/
│   └── coattention_80_10_10/  # thử nghiệm Co-Attention + 3 baseline, chia 80/10/10
├── tests/                     # kiểm thử tự động (pytest)
└── docs/                      # tài liệu dữ liệu, thực nghiệm, quy trình làm việc
```

## Cài đặt

Yêu cầu **Python 3.11**, GPU NVIDIA khuyến nghị (đã chạy trên RTX 3060 Laptop 6 GB).

```powershell
python -m venv venv
venv\Scripts\activate
# Cài PyTorch bản CUDA trước cho khớp driver
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> Mọi lệnh bên dưới chạy từ **thư mục gốc** của repo.

## Tái lập thực nghiệm

```powershell
# 1. Tải ảnh thumbnail (~3.400 ảnh, lỗi ghi vào data/text/image_download_errors.csv)
python -m scripts.download_images

# 2. Lọc mẫu hợp lệ: xem báo cáo trước, rồi ghi data/text/valid_samples.csv
python -m scripts.prepare_data
python -m scripts.prepare_data --apply

# 3–4. Huấn luyện + đánh giá 4 mode × 3 seed (bỏ qua lượt đã có kết quả)
python -m scripts.run_all --seeds 42 43 44

# 5. Gộp kết quả → results/summary.csv
python -m scripts.summarize
```

Chạy riêng một lượt:

```powershell
python -m scripts.train --mode multimodal --seed 42      # → checkpoints/multimodal_seed42.pth + .json
python -m scripts.evaluate --mode multimodal --seed 42   # → results/multimodal_seed42.json + _errors.csv
python -m scripts.train --mode concat --limit 16 --epochs 1   # smoke test, không lưu checkpoint
```

Siêu tham số mặc định: AdamW lr 2e-5, weight decay 0,01, 3 epoch, batch 4, FP16, warmup tuyến tính 10%, độ dài tiêu đề tối đa 48 token. Checkpoint được chọn theo **Macro-F1 trên tập validation**; tập test chỉ được đánh giá một lần (`evaluate.py` từ chối ghi đè nếu không có `--overwrite`).

## Demo

### Tải checkpoint

Checkpoint không đưa lên git (tổng khoảng 1,4 GB). Tải hai file từ [Google Drive](https://drive.google.com/drive/folders/1BV4ri2JhlKYOuICo7CpByWLPe195iiA1?usp=drive_link) và đặt vào thư mục `checkpoints/`:

| File | Kích thước | SHA-256 |
|---|---|---|
| `multimodal_seed42.pth` | 896.742.579 byte | `84F2011FA0E6AD94D65FEF5D8E19DA88316F9D7547B90D8398A9F32F5ABB2C57` |
| `text_seed42.pth` | 540.868.589 byte | `2672E4EB78AB04A69DC9A2766B5ED90CD7D9C7EAB53C51F4937CBA463256A1A4` |

Kiểm tra file tải về không bị lỗi: `Get-FileHash checkpoints\*.pth -Algorithm SHA256`, kết quả phải khớp bảng trên.

### Chạy

```powershell
streamlit run app.py      # demo chính
streamlit run app_v2.py   # bản phân tích: so sánh 3 mô hình, chỉnh ngưỡng, lịch sử
```

Mở <http://localhost:8501>. Cả hai bản dùng checkpoint seed 42 trong `checkpoints/` (cần có `multimodal_seed42.pth` và `text_seed42.pth`).

**`app.py`, demo chính:**
- 3 cách nhập: dán link bài báo (tự lấy tiêu đề + ảnh bìa), tự nhập tiêu đề + tải ảnh, chọn bài mẫu.
- Bài có ảnh được chấm bằng mô hình `multimodal`. Bài không có ảnh được chấm bằng mô hình `text`, không ghép ảnh giả.
- Kết quả kèm giải thích: tô màu các cụm từ mang dấu hiệu clickbait theo luật ngôn ngữ (`clickbait/signals.py`), heatmap vùng ảnh mô hình chú ý (attention rollout).
- Sidebar và tab "Báo cáo" đọc số liệu từ `results/summary.csv`, không có số nhập tay.

**`app_v2.py`, bản phân tích:** chọn mô hình Text / Image / Multimodal, chế độ **So sánh** chạy cả 3 cùng lúc, chỉnh ngưỡng quyết định, lưu lịch sử vào `analysis_history.json`.

Phần suy diễn dùng chung nằm ở `clickbait/inference.py`, với tiền xử lý giống hệt lúc huấn luyện (có bài kiểm thử đảm bảo xác suất của demo trùng với `scripts/evaluate.py`).

## Kiểm thử

```powershell
python -m pytest              # toàn bộ (cần checkpoints/text_seed42.pth cho 1 test)
python -m pytest -m "not slow"  # chỉ test nhanh, không cần checkpoint
```

Kiểm tra: tập chia không giao nhau, đúng kích thước 2237/480/480, phân tầng, tái tạo được từ seed; masked mean pooling bỏ qua padding; chỉ số đánh giá; checkpoint khớp kiến trúc; luật nhận diện dấu hiệu.

## Tài liệu

- [docs/data.md](docs/data.md): bộ dữ liệu, quy tắc lọc mẫu, chia tập
- [docs/experiments.md](docs/experiments.md): thiết lập, kết quả chi tiết, ablation warmup
- [docs/git-workflow.md](docs/git-workflow.md): quy ước nhánh, commit, pull request của nhóm
- [experiments/coattention_80_10_10/](experiments/coattention_80_10_10/README.md): thử nghiệm Co-Attention hai chiều và 3 baseline với cách chia 80/10/10
- [CHANGELOG.md](CHANGELOG.md): lịch sử thay đổi

## Nhóm thực hiện

| | |
|---|---|
| Sinh viên | Nguyễn Hữu Thiện · Thi Lý Hữu Toàn · Trần Quốc Tuấn |
| Giảng viên hướng dẫn | ThS. Trần Khải Thiện |

## Bộ dữ liệu và giấy phép

ViClickbait-2025: 3.414 tiêu đề từ 8 trang báo (Báo Mới, VnExpress, Thanh Niên, 24h, VietNamNet, SaoStar, Tuổi Trẻ, Kênh14). Dataset đang trong quá trình xét duyệt công bố, chưa có DOI; dự kiến phát hành theo CC BY 4.0. Mã nguồn phục vụ mục đích học thuật, phi thương mại.
