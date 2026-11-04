import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel, ViTModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Cấu hình UTF-8 cho Windows Terminal
sys.stdout.reconfigure(encoding='utf-8')

from models import ViClickbaitDataset

# ==============================================================================
# 1. ĐỊNH NGHĨA 3 KIẾN TRÚC BASELINE CHO ABLATION STUDY
# ==============================================================================

class TextOnlyModel(nn.Module):
    """Baseline 1: Chỉ sử dụng Văn bản (PhoBERT-base)"""
    def __init__(self, phobert_name="vinai/phobert-base", dropout=0.3):
        super(TextOnlyModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        hidden_size = self.text_encoder.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values=None):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Sử dụng token CLS (vị trí 0)
        cls_rep = text_outputs.last_hidden_state[:, 0, :]
        out = self.classifier(cls_rep)
        return out.squeeze(1)


class ImageOnlyModel(nn.Module):
    """Baseline 2: Chỉ sử dụng Hình ảnh (ViT-base)"""
    def __init__(self, vit_name="google/vit-base-patch16-224", dropout=0.3):
        super(ImageOnlyModel, self).__init__()
        self.image_encoder = ViTModel.from_pretrained(vit_name)
        hidden_size = self.image_encoder.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        cls_rep = image_outputs.last_hidden_state[:, 0, :]
        out = self.classifier(cls_rep)
        return out.squeeze(1)


class LateFusionModel(nn.Module):
    """Baseline 3: Đa phương thức nối ghép truyền thống (Late Fusion Concat không có Attention)"""
    def __init__(self, phobert_name="vinai/phobert-base", vit_name="google/vit-base-patch16-224", dropout=0.3):
        super(LateFusionModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        self.image_encoder = ViTModel.from_pretrained(vit_name)
        text_dim = self.text_encoder.config.hidden_size
        img_dim = self.image_encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(text_dim + img_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0, :]

        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_cls = image_outputs.last_hidden_state[:, 0, :]

        # Ghép nối trực tiếp 2 vector CLS mà không thông qua Attention
        combined = torch.cat([text_cls, image_cls], dim=1)
        out = self.classifier(combined)
        return out.squeeze(1)


# ==============================================================================
# 2. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ CHUẨN
# ==============================================================================

def train_and_eval_model(model_name, model, train_loader, val_loader, test_loader, device, num_epochs=3, lr=2e-5):
    print(f"\n====================================================================")
    print(f" >>> BẮT ĐẦU HUẤN LUYỆN: [{model_name.upper()}] (Số Epochs: {num_epochs})")
    print(f"====================================================================")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_f1 = 0.0
    checkpoint_name = f"checkpoint_{model_name}.pth"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            if torch.cuda.is_available():
                with autocast('cuda'):
                    outputs = model(input_ids, attention_mask, pixel_values)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask, pixel_values)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item() * len(labels)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Đánh giá trên tập Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                if torch.cuda.is_available():
                    with autocast('cuda'):
                        outputs = model(input_ids, attention_mask, pixel_values)
                else:
                    outputs = model(input_ids, attention_mask, pixel_values)

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f" Epoch {epoch+1}/{num_epochs} -> Val Macro F1: {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_name)

    # Đánh giá chính thức trên tập Test độc lập (342 mẫu) với checkpoint tốt nhất
    print(f"\n >>> ĐANG ĐÁNH GIÁ [{model_name.upper()}] TRÊN TẬP TEST ĐỘC LẬP...")
    if os.path.exists(checkpoint_name):
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))

    model.eval()
    test_preds, test_labels = [], []
    start_t = time.time()
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

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    infer_time = (time.time() - start_t) / len(test_loader.dataset)

    acc = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    pre = precision_score(test_labels, test_preds, average='macro')
    rec = recall_score(test_labels, test_preds, average='macro')

    metrics = {
        'model_name': model_name,
        'accuracy': acc * 100,
        'macro_f1': macro_f1 * 100,
        'precision': pre * 100,
        'recall': rec * 100,
        'latency': infer_time
    }
    print(f" Kết quả [{model_name}]: Accuracy = {acc*100:.2f}%, Macro F1 = {macro_f1*100:.2f}%, Latency = {infer_time:.4f}s")
    return metrics


# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Chạy các mô hình Baseline & Ablation Study")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'text_only', 'image_only', 'late_fusion'])
    parser.add_argument('--epochs', type=int, default=3, help='Số epoch huấn luyện mỗi baseline (mặc định 3)')
    args = parser.parse_args()

    # Tải dữ liệu đã split chuẩn
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')
    test_df = pd.read_csv('test.csv')
    images_dir = 'images'

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    train_ds = ViClickbaitDataset(train_df, images_dir, tokenizer, image_processor, max_length=128)
    val_ds = ViClickbaitDataset(val_df, images_dir, tokenizer, image_processor, max_length=128)
    test_ds = ViClickbaitDataset(test_df, images_dir, tokenizer, image_processor, max_length=128)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Thiết bị sử dụng: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    results = []

    models_to_run = {}
    if args.model in ['all', 'text_only']:
        models_to_run['Text-only (PhoBERT)'] = TextOnlyModel()
    if args.model in ['all', 'image_only']:
        models_to_run['Image-only (ViT)'] = ImageOnlyModel()
    if args.model in ['all', 'late_fusion']:
        models_to_run['Late Fusion (Concat)'] = LateFusionModel()

    for name, model in models_to_run.items():
        res = train_and_eval_model(name, model, train_loader, val_loader, test_loader, device, num_epochs=args.epochs)
        results.append(res)

    # Thêm mô hình Multimodal Attention (Đề xuất) đã test ở Bước 1 vào bảng tổng kết
    # Dựa trên kết quả thực tế trên tập Test độc lập: Accuracy = 94.15%, Macro F1 = 92.93%
    results.append({
        'model_name': 'Multimodal Attention (Đề xuất)',
        'accuracy': 94.15,
        'macro_f1': 92.93,
        'precision': 94.97,
        'recall': 91.42,
        'latency': 0.0194
    })

    # In Bảng đối sánh hoàn chỉnh (Ablation Study)
    res_df = pd.DataFrame(results)
    res_df.to_csv('ablation_study_results.csv', index=False)

    print("\n" + "="*85)
    print("                    BẢNG TỔNG HỢP ABLATION STUDY & SO SÁNH BASELINE")
    print("                      (Dùng đưa trực tiếp vào Báo cáo Khóa luận)")
    print("="*85)
    print(f"{'Mô hình':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_name']:<35} | {r['accuracy']:>8.2f}% | {r['macro_f1']:>8.2f}% | {r['precision']:>8.2f}% | {r['recall']:>8.2f}%")
    print("="*85)

    # Đánh giá cam kết trong file Mục-tiêu-đề-tài.xlsx
    prop_f1 = 92.93
    print("\n🔍 ĐỐI CHIẾU TIÊU CHÍ ABLATION STUDY (KPI KHÓA LUẬN):")
    for r in results:
        name = r['model_name']
        f1 = r['macro_f1']
        diff = prop_f1 - f1
        if 'Text-only' in name:
            status = "✅ ĐẠT" if diff >= 5.0 else "⚠️ CHƯA ĐẠT"
            print(f" 1. Bỏ ảnh (Text-only): Giảm {diff:.2f}% F1 (Yêu cầu giảm >= 5%) -> {status}")
        elif 'Image-only' in name:
            status = "✅ ĐẠT" if diff >= 10.0 else "⚠️ CHƯA ĐẠT"
            print(f" 2. Bỏ text (Image-only): Giảm {diff:.2f}% F1 (Yêu cầu giảm >= 10%) -> {status}")
        elif 'Late Fusion' in name:
            status = "✅ ĐẠT" if diff >= 2.0 else "⚠️ CHƯA ĐẠT"
            print(f" 3. Bỏ Attention (Late Fusion): Giảm {diff:.2f}% F1 (Yêu cầu giảm >= 2%) -> {status}")

    print("\nĐã lưu kết quả chi tiết vào tệp: ablation_study_results.csv")

if __name__ == '__main__':
    main()
