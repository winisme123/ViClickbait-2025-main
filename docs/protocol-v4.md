# Quy tắc thí nghiệm lặp lại độc lập (v4)

*Viết và commit ngày 06/11/2026, **trước khi** chạy. Thí nghiệm này lặp lại độc lập câu hỏi chính của đề tài trên 20 cách chia dữ liệu mới. Không gộp với các seed cũ để tính kiểm định chính. Mọi kết quả đều được báo cáo, kể cả khi không đạt.*

## Bối cảnh

Đề tài yêu cầu mô hình kết hợp văn bản và ảnh "nâng cao hiệu quả so với các phương pháp sử dụng văn bản hoặc hình ảnh riêng lẻ".

Thí nghiệm v3 (8 seed, `docs/protocol-v3.md`) cho kết quả: Late Fusion hơn Text-only +1,07 điểm Macro-F1, thắng 7/8 seed, nhưng Wilcoxon p = 0,078, nên kết luận "chưa chứng minh được".

Mô phỏng từ chính 8 seed đó cho thấy:
- Với 8 seed, kiểm định chỉ có khoảng **39%** khả năng phát hiện một chênh lệch cỡ 1 điểm.
- Với 20 seed, khả năng này khoảng **89%**.

Vì vậy nhóm chạy một thí nghiệm **mới, độc lập**, với cỡ mẫu được chốt trước.

## Thiết kế

- **Seed mới:** 50, 51, …, 69 (đúng **20 seed**; không chạy thêm, không dừng sớm). Mỗi seed là một cách chia 70/15/15 phân tầng mới.
- **Mô hình:** `text` (PhoBERT), `image` (ViT), `concat` (Late Fusion PhoBERT + ViT).
- **Cấu hình:** giữ nguyên v2: 3 epoch, batch 4, lr 2e-5, warmup 10%, FP16, `max_length` 48, chọn checkpoint theo Macro-F1 trên validation, ngưỡng 0,5.
- Nếu một lượt lỗi (ví dụ thiếu bộ nhớ GPU), chạy lại **đúng seed đó với cùng cấu hình**. Không bỏ seed, không thay seed.
- Do giới hạn ổ đĩa, file trọng số `.pth` bị xóa ngay sau khi đánh giá (`run_all --delete-checkpoints`). Vẫn giữ: thông tin huấn luyện (`checkpoints/*.json`), kết quả (`results/*.json`), dự đoán sai (`results/*_errors.csv`), file chia tập và log.

## Giả thuyết và kiểm định

| | Giả thuyết | Kiểm định chính |
|---|---|---|
| H1 | Late Fusion tốt hơn Text-only | Wilcoxon hạng có dấu, hai phía, trên 20 hiệu Macro-F1 (`concat − text`) |
| H2 | Late Fusion tốt hơn Image-only | Wilcoxon hạng có dấu, hai phía, trên 20 hiệu Macro-F1 (`concat − image`) |

- Hiệu chỉnh đa so sánh **Holm–Bonferroni** cho hai giả thuyết, α tổng = 0,05: giá trị p nhỏ hơn so với 0,025, giá trị p còn lại so với 0,05.
- **Kết luận "đạt"** cho một giả thuyết khi hiệu trung bình > 0 **và** vượt ngưỡng Holm.
- **Phân tích phụ (chỉ để mô tả):** số seed thắng, mean ± std, McNemar từng seed, hiệu trung bình kèm khoảng tin cậy bootstrap 95% theo seed.
- Kết quả v3 (seed 42–49) được báo cáo song song nhưng **không gộp** vào kiểm định chính.

## Lệnh chạy

```
python -m scripts.run_all --modes text image concat --seeds 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 --delete-checkpoints
python -m scripts.compare_seeds --a concat --b text  --seeds 50 ... 69
python -m scripts.compare_seeds --a concat --b image --seeds 50 ... 69
```
