# ViClickbait Detector — Phát hiện clickbait đa phương thức bằng học sâu

Khóa luận tốt nghiệp **CNTT-KLCN298**: phát hiện tiêu đề báo điện tử tiếng Việt mang tính *clickbait* bằng cách kết hợp **tiêu đề** (PhoBERT) và **ảnh thumbnail** (ViT) theo hướng **hợp nhất muộn** (late fusion), huấn luyện và đánh giá trên bộ dữ liệu **ViClickbait-2025**.

| | |
|---|---|
| Bài toán | Phân loại nhị phân `clickbait` / `non-clickbait` |
| Văn bản | [`vinai/phobert-base`](https://huggingface.co/vinai/phobert-base), tách từ bằng `pyvi`, masked mean pooling |
| Hình ảnh | [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224), mean pooling |
| Hợp nhất | `concat`: Late Fusion, nối đặc trưng (**mô hình chính, dùng trong demo**); `multimodal`: cross-attention văn bản → ảnh (phần mở rộng căn chỉnh đặc trưng) |
| Dữ liệu | 3.197 mẫu hợp lệ / 3.414 mẫu gốc, chia 70/15/15 phân tầng theo seed |
| Đánh giá | Accuracy, Precision, Recall, F1 từng lớp, **Macro-F1**, ma trận nhầm lẫn; kiểm định Wilcoxon + Holm theo seed |
| Demo | Ứng dụng web Streamlit: dán link bài báo hoặc nhập tiêu đề + ảnh → nhãn, xác suất và giải thích |

## Kết quả chính

### Lặp lại độc lập trên 20 seed (50–69)

Thí nghiệm chính để trả lời câu hỏi "kết hợp ảnh có tốt hơn từng phương thức riêng không". Quy tắc kiểm định được chốt và đẩy lên git **trước khi chạy** ([docs/protocol-v4.md](docs/protocol-v4.md)). Kết quả trên tập **test**, trung bình ± độ lệch chuẩn:

| Mô hình | Đầu vào | Accuracy | Macro-F1 | F1 (clickbait) |
|---|---|---|---|---|
| `text` | Tiêu đề | 0,8473 ± 0,0162 | **0,8231 ± 0,0184** | 0,7578 ± 0,0256 |
| `image` | Ảnh | 0,7265 ± 0,0143 | 0,6460 ± 0,0214 | 0,4776 ± 0,0397 |
| `concat` | Tiêu đề + ảnh | 0,8436 ± 0,0171 | 0,8187 ± 0,0196 | 0,7517 ± 0,0286 |

| Giả thuyết | Hiệu Macro-F1 TB | Số seed thắng | Bootstrap 95% CI | Wilcoxon p | Kết luận (Holm, α = 0,05) |
|---|---|---|---|---|---|
| H1: `concat` > `text` | −0,0044 | 7/20 | [−0,0103; +0,0012] | 0,20 | **Không đạt** |
| H2: `concat` > `image` | +0,1727 | 20/20 | [+0,1597; +0,1857] | 1,9·10⁻⁶ | **Đạt** |

**Đọc kết quả:** Late Fusion vượt rõ rệt mô hình chỉ dùng ảnh, nhưng **không tốt hơn** mô hình chỉ dùng tiêu đề. Xu hướng `concat` > `text` thấy ở 8 seed trước (7/8 seed, p = 0,078) không lặp lại trên 20 seed mới. Trên ViClickbait-2025, tín hiệu clickbait nằm chủ yếu ở cách viết tiêu đề; ảnh thumbnail phần lớn là ảnh minh họa chung.

### So sánh bốn kiến trúc (seed 42, 43, 44)

| Mô hình | Đầu vào | Accuracy | Macro-F1 | F1 (clickbait) |
|---|---|---|---|---|
| `text` | Tiêu đề | 0,8299 ± 0,0268 | 0,8029 ± 0,0346 | 0,7303 ± 0,0520 |
| `image` | Ảnh | 0,7410 ± 0,0229 | 0,6675 ± 0,0456 | 0,5116 ± 0,0810 |
| `concat` | Tiêu đề + ảnh | 0,8444 ± 0,0182 | 0,8194 ± 0,0209 | 0,7521 ± 0,0289 |
| `multimodal` | Tiêu đề + ảnh (cross-attention) | 0,8410 ± 0,0219 | 0,8087 ± 0,0315 | 0,7304 ± 0,0499 |

Cross-attention không vượt Late Fusion (McNemar p = 0,70). Kết quả 8 seed, các thí nghiệm bổ sung (đoạn mở đầu, chọn ngưỡng, baseline TF-IDF) và ablation warmup: [docs/experiments.md](docs/experiments.md).

## Cấu trúc thư mục

```
ClickBaitKLTN/
├── app.py                     # Demo chính (streamlit run app.py), mô hình Late Fusion
├── app_v2.py                  # Demo phân tích: so sánh text / image / multimodal, ngưỡng, lịch sử
├── clickbait/                 # Thư viện dùng chung
│   ├── dataset.py             #   mẫu hợp lệ, chia tập 70/15/15, Dataset PyTorch
│   ├── models.py              #   4 kiến trúc: text, image, concat, multimodal
│   ├── inference.py           #   nạp checkpoint, dự đoán, đóng góp từng từ (Shapley), heatmap
│   ├── explain.py             #   ghép đóng góp của mô hình với dấu hiệu ngôn ngữ thành lời giải thích
│   ├── signals.py             #   dấu hiệu clickbait theo luật (để giải thích trên demo)
│   ├── metrics.py             #   Accuracy / P / R / F1 / Macro-F1 / ma trận nhầm lẫn
│   ├── text_utils.py          #   chuẩn hóa và tách từ tiếng Việt (pyvi) cho PhoBERT
│   ├── scraper.py             #   lấy tiêu đề + ảnh bìa từ link bài báo
│   └── ui_paste.py            #   ô dán ảnh bằng Ctrl+V cho Streamlit
├── scripts/                   # Các bước của pipeline (python -m scripts.<tên>)
│   ├── download_images.py     #   1. tải ảnh thumbnail
│   ├── prepare_data.py        #   2. lọc mẫu hợp lệ
│   ├── train.py               #   3. huấn luyện một mode × seed
│   ├── evaluate.py            #   4. đánh giá trên tập test (một lần)
│   ├── run_all.py             #   3+4 cho mọi mode × seed
│   ├── summarize.py           #   5. gộp kết quả mean ± std
│   ├── compare_seeds.py       #   so sánh hai mode qua nhiều seed: Wilcoxon, McNemar, bootstrap CI
│   ├── holm.py                #   hiệu chỉnh Holm–Bonferroni cho nhiều giả thuyết
│   ├── tune_threshold.py      #   chọn ngưỡng quyết định trên validation
│   └── baseline_tfidf.py      #   baseline TF-IDF + Logistic Regression
├── data/
│   ├── text/                  # dataset gốc + valid_samples.csv + exclusion_review.csv
│   ├── splits/                # split_seed{42..69}.csv (id → train/val/test)
│   └── images/                # ảnh thumbnail (tự tải, không đưa lên git)
├── checkpoints/               # *.pth (không đưa lên git) + *.json meta huấn luyện
├── results/                   # kết quả test từng lượt, summary*.csv, compare_*.json, holm_v4.json
├── experiments/
│   └── coattention_80_10_10/  # thử nghiệm Co-Attention + 3 baseline, chia 80/10/10
├── tests/                     # kiểm thử tự động (pytest)
└── docs/                      # dữ liệu, thực nghiệm, quy tắc chốt trước (protocol), quy trình git
```

## Cài đặt

Yêu cầu **Python 3.11**, GPU NVIDIA khuyến nghị (đã chạy trên RTX 3060 Laptop 6 GB). Phiên bản đã kiểm thử: PyTorch 2.11.0 (CUDA 12.8), Transformers 5.22.1, Streamlit 1.66.1; danh sách đầy đủ trong `requirements-lock.txt`.

```powershell
python -m venv venv
venv\Scripts\activate
# Cài PyTorch bản CUDA trước cho khớp driver
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Hoặc cài đúng phiên bản đã kiểm thử: pip install -r requirements-lock.txt
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

Thí nghiệm lặp lại 20 seed (protocol v4, khoảng 20 phút mỗi seed trên RTX 3060; `--delete-checkpoints` xóa file `.pth` sau khi đánh giá để tiết kiệm ổ đĩa):

```powershell
python -m scripts.run_all --modes text image concat --seeds 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 --delete-checkpoints
python -m scripts.compare_seeds --a concat --b text  --seeds 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 --out results/compare_v4_concat_vs_text.json
python -m scripts.compare_seeds --a concat --b image --seeds 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 --out results/compare_v4_concat_vs_image.json
python -m scripts.holm results/compare_v4_concat_vs_text.json results/compare_v4_concat_vs_image.json
python -m scripts.summarize --modes text image concat --seeds 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 --out results/summary_v4.csv
```

Chạy riêng một lượt:

```powershell
python -m scripts.train --mode concat --seed 42      # → checkpoints/concat_seed42.pth + .json
python -m scripts.evaluate --mode concat --seed 42   # → results/concat_seed42.json + _errors.csv
python -m scripts.train --mode concat --limit 16 --epochs 1   # smoke test, không lưu checkpoint
```

Siêu tham số mặc định: AdamW lr 2e-5, weight decay 0,01, 3 epoch, batch 4, FP16, warmup tuyến tính 10%, độ dài tiêu đề tối đa 48 token. Checkpoint được chọn theo **Macro-F1 trên tập validation**; tập test chỉ được đánh giá một lần (`evaluate.py` từ chối ghi đè nếu không có `--overwrite`). Lượt nào lỗi giữa chừng thì chạy lại đúng lệnh cũ: `run_all` bỏ qua các lượt đã có kết quả.

## Demo

### Tải checkpoint

Checkpoint không đưa lên git. Tải từ [Google Drive](https://drive.google.com/drive/folders/1BV4ri2JhlKYOuICo7CpByWLPe195iiA1?usp=drive_link) và đặt vào thư mục `checkpoints/`:

| File | Dùng cho | Kích thước | SHA-256 |
|---|---|---|---|
| `concat_seed42.pth` | `app.py` (bài có ảnh) | 887.289.967 byte | `36CE9A9FC0081F5CBC05768CC639E7B80CCFBD37824216288F474E5A37046C4F` |
| `text_seed42.pth` | `app.py`, `app_v2.py` (bài không có ảnh) | 540.868.589 byte | `2672E4EB78AB04A69DC9A2766B5ED90CD7D9C7EAB53C51F4937CBA463256A1A4` |
| `multimodal_seed42.pth` | `app_v2.py` | 896.742.579 byte | `84F2011FA0E6AD94D65FEF5D8E19DA88316F9D7547B90D8398A9F32F5ABB2C57` |

Kiểm tra file tải về không bị lỗi: `Get-FileHash checkpoints\*.pth -Algorithm SHA256`, kết quả phải khớp bảng trên.

### Chạy

```powershell
streamlit run app.py      # demo chính
streamlit run app_v2.py   # bản phân tích: so sánh 3 mô hình, chỉnh ngưỡng, lịch sử
```

Mở <http://localhost:8501>. Khi GPU đang bận (ví dụ đang huấn luyện), đặt `$env:CLICKBAIT_DEVICE = "cpu"` trước khi chạy để demo dùng CPU.

**`app.py`, demo chính:**
- 3 cách nhập: dán link bài báo (tự lấy tiêu đề + ảnh bìa), tự nhập tiêu đề + tải ảnh hoặc **dán ảnh bằng Ctrl+V**, chọn bài mẫu.
- Bài có ảnh được chấm bằng mô hình **Late Fusion** (`concat` seed 42). Bài không có ảnh được chấm bằng mô hình `text`, không ghép ảnh giả.
- Kết quả gồm nhãn, xác suất clickbait, thời gian xử lý và mục **"Vì sao mô hình kết luận như vậy?"**: tóm tắt bằng lời; tiêu đề tô màu theo đóng góp của từng từ (giá trị Shapley đo trên mô hình); các điểm giống clickbait / giống tin chính thống; ảnh làm xác suất thay đổi bao nhiêu (so với ảnh xám trung tính); heatmap vùng ảnh ViT chú ý.
- Thanh bên và tab "Báo cáo" đọc số liệu từ `results/summary.csv` và `results/summary_v4.csv`, không có số nhập tay.

**`app_v2.py`, bản phân tích:** chọn mô hình Text / Image / Multimodal, chế độ **So sánh** chạy cả 3 cùng lúc, chỉnh ngưỡng quyết định, lưu lịch sử vào `analysis_history.json`.

Phần suy diễn dùng chung nằm ở `clickbait/inference.py`, với tiền xử lý giống hệt lúc huấn luyện (có bài kiểm thử bảo đảm xác suất của demo trùng với `scripts/evaluate.py`). Thời gian dự đoán một bài trên RTX 3060: khoảng 0,03–0,04 s (CPU: khoảng 0,26 s), chưa tính thời gian tải trang báo.

## Kiểm thử

```powershell
python -m pytest                # toàn bộ 47 test (một số test cần checkpoint seed 42)
python -m pytest -m "not slow"  # chỉ test nhanh, không cần checkpoint
```

Kiểm tra: tập chia không giao nhau, đúng kích thước 2237/480/480, phân tầng, tái tạo được từ seed; masked mean pooling bỏ qua padding; chỉ số đánh giá; checkpoint khớp kiến trúc; demo cho cùng xác suất với đường đánh giá; đóng góp từng từ thỏa tính chất Shapley; dùng lại đặc trưng ảnh cho kết quả như chạy đầy đủ; hiệu chỉnh Holm và bootstrap CI; luật nhận diện dấu hiệu; scraper.

## Tài liệu

- [docs/data.md](docs/data.md): bộ dữ liệu, quy tắc lọc mẫu, chia tập
- [docs/experiments.md](docs/experiments.md): thiết lập, kết quả chi tiết, ablation, thí nghiệm bổ sung v3 và lặp lại v4
- [docs/protocol-v3.md](docs/protocol-v3.md), [docs/protocol-v4.md](docs/protocol-v4.md): quy tắc thí nghiệm, chốt và đẩy lên git trước khi chạy
- [docs/git-workflow.md](docs/git-workflow.md): quy ước nhánh, commit, pull request của nhóm
- [experiments/coattention_80_10_10/](experiments/coattention_80_10_10/README.md): thử nghiệm Co-Attention hai chiều và 3 baseline với cách chia 80/10/10
- [CHANGELOG.md](CHANGELOG.md): lịch sử thay đổi

## Nhóm thực hiện

| Họ tên | MSSV | Lớp |
|---|---|---|
| Nguyễn Hữu Thiện | 2001230925 | 14DHTH12 |
| Thi Lý Hữu Toàn | 2001230828 | 14DHTH12 |
| Trần Quốc Tuấn | 2001230865 | 14DHTH13 |

Giảng viên hướng dẫn: ThS. Trần Khải Thiện. Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP. Hồ Chí Minh.

## Bộ dữ liệu và giấy phép

ViClickbait-2025: 3.414 tiêu đề từ 8 trang báo (Báo Mới, VnExpress, Thanh Niên, 24h, VietNamNet, SaoStar, Tuổi Trẻ, Kênh14). Bộ dữ liệu chỉ gồm văn bản và nhãn; ảnh thumbnail được nhóm tải lại từ bài gốc (`scripts/download_images.py`). Khi dùng bộ dữ liệu, trích dẫn bài báo gốc:

> D. P. Nguyen, T. K. Tran, Y. M. Nguyen, B. Vo. *ViClickbait-2025: A comprehensive dataset for Vietnamese clickbait detection*. Data in Brief, vol. 63, art. 112164, 2025. <https://doi.org/10.1016/j.dib.2025.112164>

Mã nguồn phục vụ mục đích học thuật, phi thương mại.
