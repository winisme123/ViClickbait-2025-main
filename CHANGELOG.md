# Lịch sử thay đổi

## [0.8.1] - 2026-11-09

### Sửa
- `requirements.txt` thiếu `scipy` (dùng trong `scripts/compare_seeds.py`).
- README: bộ dữ liệu ViClickbait-2025 đã công bố (Data in Brief, DOI 10.1016/j.dib.2025.112164); thêm MSSV và lớp của nhóm.
- Chuẩn hóa comment và docstring trong mã nguồn (bỏ ghi chú mang tính lịch sử sửa đổi, sửa mô tả mô hình Co-Attention trong `experiments/`).

### Thêm
- `requirements-lock.txt`: phiên bản chính xác của môi trường đã chạy thực nghiệm.

## [0.8.0] - 2026-11-07

### Thêm
- Thí nghiệm lặp lại độc lập theo `docs/protocol-v4.md`: 20 seed mới (50–69) × `text` / `image` / `concat`. Kết quả: `concat` vượt `image` (20/20 seed, p = 1,9·10⁻⁶) nhưng **không** vượt `text` (7/20 seed, hiệu −0,0044, p = 0,20). Tóm tắt trong `docs/experiments.md`; số liệu trong `results/summary_v4.csv`, `results/compare_v4_*.json`, `results/holm_v4.json`.
- `scripts/holm.py`: hiệu chỉnh Holm–Bonferroni; `scripts/compare_seeds.py`: tham số `--out` và khoảng tin cậy bootstrap 95% theo seed.
- `tests/test_holm.py`; `tests/test_inference.py` kiểm tra dùng lại đặc trưng ảnh cho cả `concat` và `multimodal`.

### Thay đổi
- Demo `app.py` dùng mô hình **Late Fusion** (`concat` seed 42) thay cho `multimodal`, vì đây là mô hình chính theo yêu cầu đề tài. Tab "Báo cáo" thêm kết quả 20 seed và bỏ nhận xét "kết hợp ảnh giúp cải thiện".
- README cập nhật kết quả chính, checkpoint cần tải, lệnh tái lập v4.

### Ghi chú
- Lượt `concat` seed 63 lỗi thiếu RAM hệ thống ở lần chạy đầu; đã chạy lại đúng seed với cùng cấu hình theo protocol.

## [0.7.0] - 2026-11-06

### Thêm
- Demo: dán ảnh thumbnail bằng **Ctrl+V** (Cách 2, và Cách 1 khi trang báo không cho lấy ảnh), `clickbait/ui_paste.py`.
- Demo: mục **"Vì sao mô hình kết luận như vậy?"**: tóm tắt bằng lời; tiêu đề tô màu theo đóng góp của từng từ; từng điểm giống clickbait / giống tin chính thống bấm mở xem chi tiết; ảnh hưởng của ảnh thumbnail; bảng đóng góp đầy đủ.
- `clickbait/inference.word_importance`: đóng góp của từng từ đo trực tiếp trên mô hình (giá trị Shapley theo hoán vị, đơn vị log-odds); `image_contribution`: so ảnh thật với ảnh xám trung tính. Đặc trưng ViT được tính một lần rồi dùng lại cho mọi biến thể tiêu đề.
- `clickbait/explain.py`: ghép đóng góp của mô hình với dấu hiệu của bộ luật ngôn ngữ thành các điểm giải thích.
- Biến môi trường `CLICKBAIT_DEVICE=cpu` để ép demo chạy CPU.
- `docs/protocol-v4.md`: lặp lại độc lập trên 20 seed mới (chốt trước khi chạy); `run_all --delete-checkpoints`.

## [0.6.0] - 2026-11-05

### Thêm
- Thí nghiệm bổ sung theo `docs/protocol-v3.md` (chốt trước khi chạy): 8 seed cho Text-only/Late Fusion, mode `text_lead`/`concat_lead` (tiêu đề + đoạn mở đầu), chọn ngưỡng trên validation, baseline TF-IDF. Kết quả tóm tắt trong `docs/experiments.md`.
- `scripts/compare_seeds.py` (Wilcoxon + McNemar), `scripts/tune_threshold.py`, `scripts/baseline_tfidf.py`.
- `scripts/summarize.py`: tham số `--modes/--seeds/--out`; mặc định vẫn gộp seed 42–44 nên `results/summary.csv` không đổi.
- `clickbait/text_utils.normalize_text`: chuẩn hóa NFC và bỏ ký tự vô hình cho đầu vào demo.

### Quyết định
- Demo giữ đầu vào chỉ tiêu đề và ngưỡng 0,5, vì đoạn mở đầu và ngưỡng chọn trên validation đều không cải thiện nhất quán.

## [0.5.0] - 2026-11-05

### Thay đổi
- Web của Toàn trở thành demo chính `app.py`; web phân tích cũ chuyển thành `app_v2.py`.
- `app.py` nạp mô hình `multimodal` seed 42 (tập test độc lập) thay cho `multimodal_clickbait_model.pth`, đọc số liệu từ `results/summary.csv` thay cho các số nhập tay. Bài không có ảnh được chấm bằng mô hình `text` thay vì ghép ảnh đen.
- `app.py` có thêm phần giải thích: tô dấu hiệu ngôn ngữ trong tiêu đề, heatmap attention trên ảnh.
- Phần suy diễn dùng chung của hai bản web tách ra `clickbait/inference.py`.

### Sửa lỗi
- `experiments/coattention_80_10_10/run_baselines.py`: dòng mô hình đề xuất (92,93%) trước đây nhập tay từ checkpoint bị rò rỉ dữ liệu, nay được đo thật trên `test.csv` từ checkpoint của `train.py`.
- `experiments/coattention_80_10_10/train.py`, `run_baselines.py`: cố định seed để chạy lại cho cùng kết quả.
- `clickbait/scraper.py`: link Báo Mới trước đây cho "tiêu đề" là chính đường link và ảnh là logo chung; nay lấy tiêu đề từ thẻ `<title>` và ảnh thật từ `__NEXT_DATA__`.

### Thêm
- `experiments/coattention_80_10_10/reproduction/`: log kiểm chứng và `verify_leakage.py`.
- `tests/test_inference.py`: demo cho cùng xác suất với đường đánh giá. `tests/test_scraper.py`: xử lý trang Báo Mới.
- Mục "Kiểm soát rò rỉ dữ liệu" trong `docs/experiments.md`. Phần thành viên nhóm trong README.

## [0.4.0] - 2026-11-04

### Thay đổi
- Gộp pipeline v2 vào repo chính. Thử nghiệm Co-Attention 80/10/10 (`train.py`, `run_baselines.py`, `split_data.py`, ...) được giữ nguyên nội dung và chuyển vào `experiments/coattention_80_10_10/`.

## [0.3.0] - 2026-11-04

### Thay đổi
- Tổ chức lại mã nguồn: thư viện chung đặt trong `clickbait/`, các bước pipeline đặt trong `scripts/` (chạy bằng `python -m scripts.<tên>`), kiểm thử đặt trong `tests/`, tài liệu đặt trong `docs/`.
- Tách chỉ số đánh giá ra `clickbait/metrics.py`.
- Chuyển kết quả đối chứng không warmup vào `results/v1_no_warmup/`.

### Sửa lỗi
- `signals.py`: mẫu "không thể bỏ **qua**" trước đây bị gõ nhầm thành "bỏ quen" nên không bao giờ khớp.

### Thêm
- Bộ kiểm thử pytest cho chia tập, mô hình, chỉ số đánh giá và luật nhận diện dấu hiệu.
- README, `docs/data.md`, `docs/experiments.md`, `docs/git-workflow.md`.

## [0.2.0] - 2026-11-03

### Thêm
- Tải lại ảnh thumbnail từ `og:image` / `__NEXT_DATA__` (Báo Mới), ghi log lỗi.
- Lọc mẫu hợp lệ (`prepare_data.py`): còn 3.197/3.414 mẫu.
- Chia 70/15/15 phân tầng, cố định theo seed 42/43/44.
- Mô hình `concat` làm baseline hợp nhất muộn. Masked mean pooling cho văn bản.
- Huấn luyện: chọn checkpoint theo Macro-F1 trên validation, FP16, warmup tuyến tính 10%, cảnh báo khi mô hình sụp về một lớp, lưu meta `.json`.
- Đánh giá trên tập test: P/R/F1 từng lớp, ma trận nhầm lẫn, kết quả theo nguồn báo, file mẫu dự đoán sai.
- `run_all.py`, `summarize.py`: chạy 4 mode × 3 seed và gộp kết quả mean ± std.

## [0.1.0] - 2026-11-01

### Thêm
- Ứng dụng Streamlit: dán link bài báo, 3 chế độ, heatmap attention, giải thích theo luật, lịch sử.
- `scraper.py`, `signals.py`, `text_utils.py`.

## [0.0.1] - 2026-09-10

### Thêm
- Bộ dữ liệu ViClickbait-2025, mã huấn luyện và đánh giá ban đầu.
