# Quy tắc thí nghiệm bổ sung (v3)

*Viết và commit ngày 05/11/2026, **trước khi** chạy bất kỳ thí nghiệm nào dưới đây. Mọi lựa chọn được chốt bằng dữ liệu train/validation. Tập test của mỗi cấu hình chỉ được đánh giá **một lần**, và **mọi kết quả đều được báo cáo**, kể cả khi kém hơn.*

Kết quả chính (4 mô hình × seed 42/43/44, `results/summary.csv`) giữ nguyên, không chạy lại.

## A. Lặp lại thêm 5 seed: Text-only và Late Fusion

**Câu hỏi:** Late Fusion (`concat`) có tốt hơn Text-only (`text`) một cách nhất quán không? Đây là mục tiêu ghi trong đề cương.

- Seed mới: **45, 46, 47, 48, 49**. Cách chia 70/15/15 phân tầng như cũ.
- Cấu hình **giữ nguyên** v2: 3 epoch, batch 4, lr 2e-5, warmup 10%, FP16, `max_length` 48, chọn checkpoint theo Macro-F1 trên validation.
- Báo cáo **cả 8 seed** (42–49), không bỏ seed nào.
- **Phân tích chính:** kiểm định Wilcoxon hạng có dấu, hai phía, α = 0,05, trên 8 cặp hiệu Macro-F1 (`concat − text`) theo seed.
- **Phân tích phụ:** số seed `concat` thắng, kiểm định McNemar từng seed, độ lệch chuẩn qua 8 seed.
- **Kết luận "đạt"** khi trung bình hiệu > 0 **và** Wilcoxon p < 0,05. Ngược lại, báo cáo là "chưa chứng minh được".

## B. Thêm đoạn mở đầu bài báo (`lead_paragraph`)

**Câu hỏi:** đưa thêm đoạn mở đầu (văn bản ngữ cảnh) vào cùng tiêu đề có giúp phát hiện clickbait không?

- Mode mới: `text_lead` (PhoBERT) và `concat_lead` (PhoBERT + ViT, late fusion).
- Đầu vào văn bản: cặp câu PhoBERT `<s> tiêu đề </s></s> đoạn mở đầu </s>`, cả hai đều tách từ bằng pyvi. Nếu quá dài thì **chỉ cắt đoạn mở đầu** (`truncation='only_second'`).
- `max_length` = **80**, theo quy tắc chốt trước: phân vị 95 của độ dài token (tiêu đề + đoạn mở đầu) trên tập **train seed 42** là 73,2, làm tròn lên bội của 16.
- Các siêu tham số khác **giống hệt** v2. Seed 42, 43, 44, so sánh với `text` và `concat` trên cùng cách chia.
- Báo cáo mean ± std, và Wilcoxon/McNemar như mục A (với 3 seed, chủ yếu báo cáo hướng và độ lớn của chênh lệch).
- Demo chỉ chuyển sang dùng đoạn mở đầu nếu kết quả trên test tốt hơn **và** nhóm đồng ý. Quyết định này ghi lại trong CHANGELOG.

## C. Chọn ngưỡng quyết định trên validation

**Câu hỏi:** dữ liệu lệch nhãn (31,8% clickbait). Ngưỡng 0,5 có phải là tốt nhất không?

- Với **mỗi** checkpoint (mọi mode, mọi seed): tính xác suất trên tập **validation**, chọn ngưỡng trong lưới 0,05–0,95 (bước 0,01) có Macro-F1 cao nhất. Nếu bằng nhau, chọn ngưỡng gần 0,5 nhất.
- Áp dụng ngưỡng đó **một lần** lên tập test. Báo cáo song song với kết quả ở ngưỡng 0,5, không thay thế.

## D. Baseline cổ điển: TF-IDF + Logistic Regression

**Câu hỏi:** dùng PhoBERT có đáng so với một mô hình cổ điển không?

- Đặc trưng: TF-IDF trên tiêu đề đã tách từ (pyvi), n-gram từ 1–2, `min_df=2`, `sublinear_tf=True`.
- Bộ phân loại: `LogisticRegression(max_iter=2000)`. Hệ số `C` chọn trong {0,01; 0,1; 1; 10; 100} theo Macro-F1 trên **validation**.
- Chạy trên cả 8 cách chia (seed 42–49), báo cáo mean ± std.

## Không làm

- Không chọn seed, không bỏ kết quả xấu.
- Không chỉnh siêu tham số sau khi thấy kết quả test.
- Không đổi bảng kết quả chính v2.
