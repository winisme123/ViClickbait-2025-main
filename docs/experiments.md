# Thực nghiệm

## Mô hình (`clickbait/models.py`)

| Mode | Kiến trúc |
|---|---|
| `text` | PhoBERT → masked mean pooling → MLP |
| `image` | ViT → mean pooling trên các token ảnh → MLP |
| `concat` | [masked mean PhoBERT ; mean ViT] → MLP. Hợp nhất muộn, không có cross-attention |
| `multimodal` | Cross-attention 8 head (query là token văn bản, key/value là token ảnh), sau đó [masked mean văn bản ; masked mean đầu ra attention] → MLP |

Cả bốn mode dùng chung MLP `Linear(d, 256) → ReLU → Dropout(0,3) → Linear(256, 1)`. Hàm mất mát là `BCEWithLogitsLoss`.

## Thiết lập huấn luyện (`scripts/train.py`)

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW, lr 2e-5, weight decay 0,01 |
| Lịch học | Warmup tuyến tính 10% số bước (168/1.680), sau đó giảm tuyến tính về 0 |
| Epoch, batch | 3 epoch, batch 4, không tích lũy gradient |
| Độ chính xác | FP16 (autocast + GradScaler) |
| Độ dài tiêu đề | 48 token (dài nhất trên tập hợp lệ là 38) |
| Chọn checkpoint | Epoch có Macro-F1 trên validation cao nhất |
| Phần cứng | RTX 3060 Laptop 6 GB, Windows 11 |

Mỗi checkpoint có file `checkpoints/{mode}_seed{seed}.json` đi kèm, lưu siêu tham số, lịch sử từng epoch, thời gian và VRAM.

## Kết quả trên tập test

Các con số dưới đây lấy từ `results/summary.csv`, tính trung bình ± độ lệch chuẩn mẫu (ddof = 1) qua 3 seed.

| Mode | Accuracy | Macro-F1 | P (cb) | R (cb) | F1 (cb) | F1 (non-cb) | Thời gian train (s) |
|---|---|---|---|---|---|---|---|
| text | 0,8299 ± 0,0268 | 0,8029 ± 0,0346 | 0,7354 | 0,7277 | 0,7303 | 0,8756 | 182 |
| image | 0,7410 ± 0,0229 | 0,6675 ± 0,0456 | 0,6347 | 0,4336 | 0,5116 | 0,8234 | 429 |
| concat | 0,8444 ± 0,0182 | 0,8194 ± 0,0209 | 0,7649 | 0,7407 | 0,7521 | 0,8866 | 572 |
| multimodal | 0,8410 ± 0,0219 | 0,8087 ± 0,0315 | 0,7905 | 0,6819 | 0,7304 | 0,8871 | 584 |

Macro-F1 theo từng seed:

| Mode | 42 | 43 | 44 | Epoch tốt nhất | VRAM tối đa |
|---|---|---|---|---|---|
| text | 0,7938 | 0,8412 | 0,7739 | 3 / 2 / 2 | 3,0 GB |
| image | 0,6520 | 0,7188 | 0,6316 | 3 / 3 / 3 | 2,0 GB |
| concat | 0,8047 | 0,8433 | 0,8101 | 3 / 3 / 2 | 5,0 GB |
| multimodal | 0,8068 | 0,8411 | 0,7783 | 2 / 2 / 2 | 5,0 GB |

Mỗi lượt có kết quả chi tiết trong `results/{mode}_seed{seed}.json`: ma trận nhầm lẫn, P/R/F1 từng lớp, kết quả theo nguồn báo và siêu tham số. Các mẫu dự đoán sai nằm trong `results/{mode}_seed{seed}_errors.csv`.

### Nhận xét

- Ảnh đứng một mình yếu hơn hẳn (Macro-F1 0,67). Recall lớp clickbait chỉ đạt 0,43.
- Thêm ảnh vào văn bản cải thiện được một chút: `concat` tăng khoảng 1,7 điểm Macro-F1 so với `text`, độ lệch chuẩn cũng nhỏ hơn.
- `multimodal` (cross-attention) **không vượt** `concat`. Chênh lệch giữa ba mô hình có văn bản đều nằm trong khoảng một độ lệch chuẩn, nên 3 seed chưa đủ để khẳng định mô hình nào tốt hơn có ý nghĩa thống kê.
- So với `concat`, `multimodal` có precision lớp clickbait cao hơn nhưng recall thấp hơn.

## Ablation: warmup learning rate

Lượt đầu (v1) **không dùng warmup**, mọi thiết lập khác giữ nguyên. Kết quả lưu ở `results/v1_no_warmup/`.

| Mode | v1 không warmup (42 / 43 / 44) | v1 Macro-F1 | v2 warmup 10% Macro-F1 |
|---|---|---|---|
| text | 0,7931 / 0,8269 / **0,4052** | 0,6751 ± 0,2343 | 0,8029 ± 0,0346 |
| image | 0,6447 / 0,6873 / 0,6092 | 0,6470 ± 0,0391 | 0,6675 ± 0,0456 |
| concat | 0,7845 / 0,8230 / 0,7933 | 0,8003 ± 0,0201 | 0,8194 ± 0,0209 |
| multimodal | 0,7782 / 0,8367 / 0,7925 | 0,8025 ± 0,0305 | 0,8087 ± 0,0315 |

Ở v1, mô hình `text` seed 44 bị **sụp về dự đoán một lớp**: loss dừng ở khoảng 0,625, đúng bằng entropy của phân phối nhãn, và Macro-F1 trên validation không đổi qua các epoch. Thêm warmup tuyến tính 10% khắc phục được hiện tượng này và cải thiện cả bốn mode. Vì vậy v2 là cấu hình chính. Từ v2, `train.py` in cảnh báo khi Macro-F1 trên validation không đổi qua mọi epoch. Không lượt nào của v2 bị cảnh báo.

## Kiểm soát rò rỉ dữ liệu

Bản thử nghiệm Co-Attention ban đầu (`experiments/coattention_80_10_10/`) từng báo Macro-F1 92,93%. Con số này đo bằng checkpoint `multimodal_clickbait_model.pth`, được huấn luyện bằng `train.py` gốc (commit `efd89c8`): script đó chia ngẫu nhiên 80/20 trên **toàn bộ** 3.414 mẫu. `test.csv` được `split_data.py` tạo **sau đó** bằng một lần chia khác, nên khoảng 80% mẫu test (trung bình 273,6/342 trên 1.000 cách chia ngẫu nhiên) đã nằm trong dữ liệu huấn luyện.

| Tập (chia 80/10/10) | Macro-F1 của checkpoint |
|---|---|
| `train.csv` | 90,29% |
| `val.csv` | 93,56% |
| `test.csv` | 91,42% |

Điểm trên ba tập ngang nhau. Một mô hình chỉ học `train.csv` không cho kết quả như vậy. Bằng chứng và script kiểm tra lại (không cần GPU) nằm trong `experiments/coattention_80_10_10/reproduction/`.

Pipeline chính tránh lỗi này bằng các biện pháp sau:

- Tập test được tách phân tầng **trước** khi huấn luyện, lưu thành file `data/splits/split_seed{seed}.csv`.
- `tests/test_dataset.py` kiểm tra train/val/test không giao nhau và tái tạo đúng từ seed.
- Checkpoint được chọn chỉ theo tập validation. `scripts/evaluate.py` từ chối ghi đè kết quả test đã có.
- Demo và tài liệu đọc số liệu từ `results/`, không nhập tay. `tests/test_inference.py` kiểm tra demo cho cùng xác suất với đường đánh giá.

## Thí nghiệm bổ sung v3 (quy tắc chốt trước: `docs/protocol-v3.md`)

| Thí nghiệm | Kết quả (Macro-F1) | Kết luận |
|---|---|---|
| A. Text-only vs Late Fusion, 8 seed (42–49) | 0,8085 ± 0,0239 vs **0,8192 ± 0,0143**; concat thắng 7/8; Wilcoxon p = 0,078; McNemar gộp p = 0,030 | Tốt hơn nhất quán về hướng, **chưa chứng minh được** theo tiêu chí chính (Wilcoxon p < 0,05) |
| B. Thêm đoạn mở đầu (seed 42–44) | text 0,8029 → 0,8180 (2/3 seed tăng); concat 0,8194 → 0,8038 (0/3) | Không cải thiện nhất quán, demo giữ chỉ tiêu đề |
| C. Ngưỡng chọn trên validation | concat 0,8192 → 0,8160; text 0,8085 → 0,8094; multimodal 0,8087 → 0,8128; image 0,6675 → 0,6676 | Không cải thiện nhất quán, giữ ngưỡng 0,5 |
| D. TF-IDF + Logistic Regression, 8 seed | 0,7556 ± 0,0125 | PhoBERT cao hơn khoảng 5 điểm |

File: `results/summary_8seeds.csv`, `results/summary_lead.csv`, `results/compare_*.json`, `results/threshold/`, `results/tfidf/`.

## Tái lập

```powershell
python -m scripts.run_all --seeds 42 43 44
python -m scripts.summarize
```

Lượt đã có `results/{mode}_seed{seed}.json` sẽ được bỏ qua. Muốn chạy lại thì chuyển thư mục `results/` và `checkpoints/` sang chỗ khác trước.
