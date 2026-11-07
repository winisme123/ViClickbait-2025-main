import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from torch.amp import autocast

# Hỗ trợ hiển thị tiếng Việt UTF-8 chuẩn trên Windows
sys.stdout.reconfigure(encoding='utf-8')

from models import ViClickbaitDataset, MultimodalClickbaitModel, MultimodalCoAttentionModel

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phát hiện Clickbait trên tập Test")
    parser.add_argument('--weights', type=str, default='multimodal_clickbait_model.pth', help='Đường dẫn file checkpoint .pth')
    parser.add_argument('--model_type', type=str, default='legacy', choices=['legacy', 'coattention'], help='Loại kiến trúc: legacy hoặc coattention')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='File dữ liệu test độc lập')
    args = parser.parse_args()

    repo_dir = '.'
    test_path = os.path.join(repo_dir, args.test_csv)
    images_dir = os.path.join(repo_dir, 'images')

    if not os.path.exists(test_path):
        print(f"LỖI: Không tìm thấy file {test_path}! Hãy chạy `split_data.py` trước.")
        return

    test_df = pd.read_csv(test_path)
    print("====================================================================")
    print("      HỆ THỐNG ĐÁNH GIÁ MÔ HÌNH PHÁT HIỆN CLICKBAIT ĐA PHƯƠNG THỨC    ")
    print("====================================================================")
    print(f" Tập dữ liệu kiểm thử (Test): {test_path} ({len(test_df)} mẫu)")
    print(f" Kiến trúc mô hình:           {args.model_type.upper()}")
    print(f" Tệp trọng số mô hình:        {args.weights}")

    # 1. Khởi tạo Tokenizer và Image Processor
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 2. Tạo Dataset & DataLoader cho tập Test
    test_dataset = ViClickbaitDataset(test_df, images_dir, tokenizer, image_processor, max_length=128)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 3. Khởi tạo mô hình & Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Thiết bị suy diễn:           {device}")
    if torch.cuda.is_available():
        print(f" Tên GPU:                     {torch.cuda.get_device_name(0)}")

    if args.model_type == 'coattention':
        model = MultimodalCoAttentionModel().to(device)
    else:
        model = MultimodalClickbaitModel().to(device)

    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
        print(" Đã tải trọng số mô hình thành công!")
    else:
        print(f" CẢNH BÁO: Chưa tìm thấy file trọng số '{args.weights}'! Mô hình sẽ chạy với trọng số ngẫu nhiên.")

    # 4. Thực hiện suy diễn & đo thời gian (KPI: < 2.5s / mẫu)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            if torch.cuda.is_available():
                with autocast('cuda'):
                    outputs = model(input_ids, attention_mask, pixel_values)
            else:
                outputs = model(input_ids, attention_mask, pixel_values)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_infer_time = time.time() - start_time
    avg_latency = total_infer_time / len(test_df)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 5. Tính toán các chỉ số nghiệm thu theo đề cương khóa luận
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    pre_macro = precision_score(all_labels, all_preds, average='macro')
    rec_macro = recall_score(all_labels, all_preds, average='macro')
    diff_pre_rec = abs(pre_macro - rec_macro)

    cb_f1 = f1_score(all_labels, all_preds, pos_label=1.0)
    cb_pre = precision_score(all_labels, all_preds, pos_label=1.0)
    cb_rec = recall_score(all_labels, all_preds, pos_label=1.0)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n======================= KẾT QUẢ ĐÁNH GIÁ CHI TIẾT =======================")
    print(classification_report(all_labels, all_preds, target_names=['Non-Clickbait (0)', 'Clickbait (1)'], digits=4))
    print("-------------------------------------------------------------------------")
    print(f" 🎯 Accuracy:                         {acc*100:.2f}%")
    print(f" ⭐ Macro F1-Score (KPI chính):        {macro_f1*100:.2f}%")
    print(f" 📊 Macro Precision:                  {pre_macro*100:.2f}%")
    print(f" 📊 Macro Recall:                     {rec_macro*100:.2f}%")
    print(f" ⚖️ Độ lệch |Precision - Recall|:     {diff_pre_rec*100:.2f}%  (KPI yêu cầu <= 5.0%)")
    print(f" ⚡ Thời gian phản hồi trung bình:    {avg_latency:.4f}s / mẫu (KPI yêu cầu < 2.5s)")
    print("-------------------------------------------------------------------------")
    print(" MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print(f"  [TN={tn:3d}  |  FP={fp:3d}]  (Thực tế Non-clickbait: {tn+fp})")
    print(f"  [FN={fn:3d}  |  TP={tp:3d}]  (Thực tế Clickbait:     {fn+tp})")
    print("=========================================================================\n")

if __name__ == '__main__':
    main()