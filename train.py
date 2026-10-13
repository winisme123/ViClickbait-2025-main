import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, ViTImageProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.amp import GradScaler, autocast

# Đảm bảo in tiếng Việt chuẩn trên console Windows
sys.stdout.reconfigure(encoding='utf-8')

from models import ViClickbaitDataset, MultimodalCoAttentionModel

def train():
    # 1. Đường dẫn dữ liệu
    repo_dir = '.'
    train_csv = os.path.join(repo_dir, 'train.csv')
    val_csv = os.path.join(repo_dir, 'val.csv')
    images_dir = os.path.join(repo_dir, 'images')

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print("LỖI: Chưa tìm thấy train.csv hoặc val.csv. Vui lòng chạy split_data.py trước!")
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print("====================================================================")
    print(f" Số lượng mẫu tập Huấn luyện (Train):      {len(train_df)}")
    print(f" Số lượng mẫu tập Kiểm định (Validation): {len(val_df)}")
    print("====================================================================")

    # 2. Khởi tạo Tokenizer và Image Processor
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 3. Tạo Dataset & DataLoader
    train_dataset = ViClickbaitDataset(train_df, images_dir, tokenizer, image_processor, max_length=128)
    val_dataset = ViClickbaitDataset(val_df, images_dir, tokenizer, image_processor, max_length=128)

    # Batch size 4 hoặc 8 phù hợp VRAM 8GB của RTX 4060
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Cấu hình thiết bị & Khởi tạo mô hình
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Thiết bị huấn luyện: {device}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")

    model = MultimodalCoAttentionModel().to(device)

    # 5. Xử lý mất cân bằng lớp (Class Imbalance)
    # Tỷ lệ nhãn âm (0) / nhãn dương (1)
    num_pos = (train_df['label'] == 'clickbait').sum()
    num_neg = len(train_df) - num_pos
    pos_weight_val = num_neg / max(1, num_pos)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
    print(f" Trọng số cân bằng nhãn (pos_weight cho BCE): {pos_weight_val:.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 6. Cấu hình Tối ưu hóa (Optimizer & Scheduler)
    # Tách learning rate: encoder học chậm (2e-5), classifier học nhanh hơn (1e-4)
    encoder_params = list(model.text_encoder.parameters()) + list(model.image_encoder.parameters())
    fusion_params = (
        list(model.cross_attn_text_to_img.parameters()) + 
        list(model.cross_attn_img_to_text.parameters()) + 
        list(model.classifier.parameters())
    )

    optimizer = AdamW([
        {'params': encoder_params, 'lr': 2e-5, 'weight_decay': 0.01},
        {'params': fusion_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])

    num_epochs = 4
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # 7. Hàm đánh giá trên tập Validation
    def evaluate(model, dataloader):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                if torch.cuda.is_available():
                    with autocast('cuda'):
                        outputs = model(input_ids, attention_mask, pixel_values)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(input_ids, attention_mask, pixel_values)
                    loss = criterion(outputs, labels)

                total_loss += loss.item() * len(labels)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        clickbait_f1 = f1_score(all_labels, all_preds, pos_label=1.0)
        return avg_loss, acc, macro_f1, clickbait_f1

    # 8. Huấn luyện chính
    best_val_macro_f1 = 0.0
    best_model_path = "best_multimodal_coattention_model.pth"

    print("\n>>> BẮT ĐẦU TIẾN TRÌNH HUẤN LUYỆN ĐA PHƯƠNG THỨC VỚI CO-ATTENTION <<<")
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

        epoch_train_loss = train_loss / len(train_loader.dataset)
        val_loss, val_acc, val_macro_f1, val_cb_f1 = evaluate(model, val_loader)

        print(f"\n--- Kết quả Epoch {epoch+1}/{num_epochs} ---")
        print(f" Train Loss:     {epoch_train_loss:.4f}")
        print(f" Val Loss:       {val_loss:.4f}")
        print(f" Val Accuracy:   {val_acc*100:.2f}%")
        print(f" Val Macro F1:   {val_macro_f1*100:.2f}%")
        print(f" Val CB F1:      {val_cb_f1*100:.2f}%")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(f" => Đã lưu checkpoint mới tốt nhất (Val Macro F1 = {val_macro_f1*100:.2f}%)!")

    print(f"\nHuấn luyện hoàn tất! Trọng số tốt nhất đã được lưu tại: {best_model_path}")

if __name__ == '__main__':
    train()