# Lịch sử thay đổi

## [0.5.0] - 2026-11-05

### Thay đổi
- Web của Toàn trở thành demo chính `app.py`; web phân tích cũ chuyển thành `app_v2.py`.
- `app.py` nạp mô hình `multimodal` seed 42 (tập test độc lập) thay cho `multimodal_clickbait_model.pth`, đọc số liệu từ `results/summary.csv` thay cho các số nhập tay. Bài không có ảnh được chấm bằng mô hình `text` thay vì ghép ảnh đen.
- `app.py` có thêm phần giải thích: tô dấu hiệu ngôn ngữ trong tiêu đề, heatmap attention trên ảnh.
- Phần suy diễn dùng chung của hai bản web tách ra `clickbait/inference.py`.

### Sửa lỗi
- `experiments/coattention_80_10_10/run_baselines.py`: dòng mô hình đề xuất (92,93%) trước đây nhập tay từ checkpoint bị rò rỉ dữ liệu, nay được đo thật trên `test.csv` từ checkpoint của `train.py`.
- `experiments/coattention_80_10_10/train.py`, `run_baselines.py`: cố định seed để chạy lại cho cùng kết quả.

### Thêm
- `experiments/coattention_80_10_10/reproduction/`: log kiểm chứng và `verify_leakage.py`.
- `tests/test_inference.py`: demo cho cùng xác suất với đường đánh giá.
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
