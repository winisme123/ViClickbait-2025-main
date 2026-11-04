# Dữ liệu

## Bộ dữ liệu gốc

`data/text/clickbait_dataset_vietnamese.csv` (bản `.jsonl` cùng nội dung): **3.414** tiêu đề, 2.349 `non-clickbait` và 1.065 `clickbait` (31,2%).

Các cột chính: `id`, `url`, `title`, `lead_paragraph`, `category`, `publish_datetime`, `source`, `thumbnail_url`, `label`.

| Nguồn | Số mẫu |
|---|---|
| Báo Mới | 731 |
| VnExpress | 608 |
| Thanh Niên | 576 |
| 24h | 564 |
| VietNamNet | 356 |
| SaoStar | 286 |
| Tuổi Trẻ | 241 |
| Kênh14 | 52 |

## Tải ảnh (`scripts/download_images.py`)

Cột `thumbnail_url` chỉ chứa đường dẫn file đích, không phải URL ảnh. Vì vậy ảnh được lấy lại từ trang gốc (cột `url`):

- Thông thường lấy thẻ `og:image`.
- Báo Mới: `og:image` là logo chung nên ảnh được lấy từ `__NEXT_DATA__`, khớp theo ID bài trong URL.
- Ảnh được kiểm tra bằng PIL rồi lưu vào `data/images/`. Bài bị gỡ hoặc lỗi mạng được ghi vào `data/text/image_download_errors.csv`.

## Lọc mẫu hợp lệ (`scripts/prepare_data.py`)

Một mẫu bị loại nếu có ít nhất một lý do sau:

| Lý do | Ý nghĩa | Số dòng |
|---|---|---|
| `thieu_anh` | Không tải được ảnh | 187 |
| `thieu_anh;url_khong_phai_bai` | Trang chuyên mục, không có ảnh | 15 |
| `url_khong_phai_bai` | URL không chứa ID bài (≥ 5 chữ số), là trang chuyên mục | 7 |
| `url_khong_phai_bai;tieu_de_<=2_tu` | Trang chuyên mục, tiêu đề ≤ 2 từ | 5 |
| `anh_giu_cho` | Ảnh giữ chỗ (logo xám VnExpress, md5 `4db13668…`) | 3 |
| **Tổng** | | **217** |

Kết quả còn **3.197 mẫu hợp lệ** (2.181 `non-clickbait`, 1.016 `clickbait`, 31,8%). Danh sách id nằm trong `data/text/valid_samples.csv`, danh sách loại kèm lý do nằm trong `data/text/exclusion_review.csv`. Cả bốn mô hình dùng chung đúng tập mẫu này để so sánh công bằng.

## Chia tập (`clickbait/dataset.py`)

- Chia **70/15/15 phân tầng theo nhãn** bằng `sklearn.model_selection.train_test_split`: tách 30% ra trước, sau đó chia đôi phần này thành val và test.
- Dùng 3 seed **42, 43, 44**. Mỗi seed được lưu trong `data/splits/split_seed{seed}.csv` (`id`, `split`).
- Mỗi seed có train **2.237**, val **480** và test **480** mẫu. Tỉ lệ clickbait giữa các tập chênh nhau dưới 1 điểm phần trăm.
- `tests/test_dataset.py` kiểm tra các file chia tập tái tạo đúng từ seed và không giao nhau.

```powershell
python -m clickbait.dataset --seed 42   # tạo lại file chia tập và in thống kê
```
