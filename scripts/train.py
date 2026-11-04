import os
import sys
import copy
import json
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, ViTImageProcessor, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from clickbait.dataset import MAX_LENGTH, ViClickbaitDataset, load_split
from clickbait.models import MODEL_CLASSES, MODES, PHOBERT_NAME, VIT_NAME, checkpoint_path

# Đảm bảo in ký tự tiếng Việt không bị lỗi bảng mã trên Windows console
if sys.platform == 'win32':
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr is not None:
        sys.stderr.reconfigure(encoding='utf-8')

# 0. Tham số dòng lệnh
parser = argparse.ArgumentParser(description='Huấn luyện mô hình clickbait')
parser.add_argument(
    '--mode', type=str, default='multimodal', choices=MODES,
    help='Chế độ mô hình: text (chỉ PhoBERT), image (chỉ ViT), '
         'multimodal (PhoBERT + ViT + cross-attention), concat (nối đặc trưng, không cross-attention)'
)
parser.add_argument('--epochs', type=int, default=3, help='Số epoch huấn luyện')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--grad-accum', type=int, default=1, help='Số bước tích lũy gradient')
parser.add_argument('--warmup-ratio', type=float, default=0.1,
                    help='Tỉ lệ số bước warmup tuyến tính trên tổng số bước, sau đó giảm tuyến tính về 0')
parser.add_argument('--seed', type=int, default=42, help='Seed chia tập và khởi tạo')
parser.add_argument('--limit', type=int, default=None,
                    help='Smoke test: chỉ dùng N mẫu train và N mẫu val, không lưu checkpoint')
args = parser.parse_args()
MODE = args.mode


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

# 1. Dữ liệu: danh sách mẫu hợp lệ + chia 70/15/15 phân tầng theo seed
train_df = load_split(args.seed, 'train')
val_df = load_split(args.seed, 'val')
if args.limit:
    train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=args.seed)
    val_df = val_df.sample(n=min(args.limit, len(val_df)), random_state=args.seed)
print(f"Chế độ huấn luyện: {MODE.upper()} | seed {args.seed}")

# 2. Khởi tạo Tokenizer và Image Processor
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)
image_processor = ViTImageProcessor.from_pretrained(VIT_NAME)

train_dataset = ViClickbaitDataset(train_df, tokenizer, image_processor, MODE)
val_dataset = ViClickbaitDataset(val_df, tokenizer, image_processor, MODE)

# num_workers=0 để tránh lỗi đa tiến trình trên Windows
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(args.seed))
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print(f"Số lượng mẫu Train: {len(train_dataset)} | Số lượng mẫu Validation: {len(val_dataset)}")


def model_inputs(batch, device):
    """Chỉ chuyển lên GPU các đầu vào mà mode hiện tại có dùng."""
    return {k: batch[k].to(device) for k in ('input_ids', 'attention_mask', 'pixel_values') if k in batch}


# Đoạn mã huấn luyện chính
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL_CLASSES[MODE]().to(device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Không tìm thấy GPU'
    print(f"Đã khởi tạo mô hình trên thiết bị: {device} | Tên GPU: {gpu_name}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Tổng số bước cập nhật optimizer (đã tính grad-accum)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"LR scheduler: warmup tuyến tính {warmup_steps}/{total_steps} bước, sau đó giảm tuyến tính về 0")
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")

    def evaluate_model(model, dataloader, device):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                labels = batch['label'].to(device)

                with autocast("cuda"):
                    outputs = model(**model_inputs(batch, device))
                    loss = criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                probs = torch.sigmoid(outputs)
                all_preds.extend((probs >= 0.5).float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return total_loss / len(all_labels), acc, macro_f1

    num_epochs = args.epochs
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN MODE {MODE.upper()} ({num_epochs} epochs, "
          f"batch {args.batch_size} x grad-accum {args.grad_accum}) ---")

    # Theo dõi trọng số epoch tốt nhất (dựa trên Val Macro-F1)
    best_val_f1 = -1.0
    best_epoch = -1
    best_model_wts = None
    history = []
    step_times = []
    train_start = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"[{MODE}] Epoch {epoch+1}/{num_epochs}")
        step_start = time.perf_counter()
        for step, batch in enumerate(progress_bar):
            labels = batch['label'].to(device)

            with autocast("cuda"):
                outputs = model(**model_inputs(batch, device))
                loss = criterion(outputs, labels)

            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * labels.size(0)
            # outputs là logit: logit >= 0 tương đương sigmoid >= 0.5
            preds = (outputs >= 0).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            # Thời gian 1 bước (gồm cả nạp dữ liệu)
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
            step_start = step_end

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        val_start = time.perf_counter()
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device)
        val_time = time.perf_counter() - val_start
        history.append({'epoch': epoch + 1, 'train_loss': epoch_train_loss, 'train_acc': epoch_train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc, 'val_macro_f1': val_f1,
                        'val_time_sec': val_time})

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Hoàn tất:")
        print(f" - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f" - Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}"
              f" | Val time: {val_time:.1f} s\n")

        # Cập nhật trọng số tốt nhất nếu Val Macro-F1 cải thiện
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  ★ Trọng số tốt nhất mới tại epoch {best_epoch} (Val Macro-F1: {best_val_f1:.4f})\n")

    train_time = time.perf_counter() - train_start
    # Bỏ 3 bước đầu (khởi động CUDA/cuDNN) khi tính thời gian trung bình
    avg_step = float(np.mean(step_times[3:] if len(step_times) > 3 else step_times))

    print(f"Huấn luyện mode {MODE.upper()} hoàn thành!")
    print(f"Epoch tốt nhất: {best_epoch}/{num_epochs} với Val Macro-F1 = {best_val_f1:.4f}")
    print(f"Thời gian huấn luyện (gồm validation): {train_time:.1f} s | Trung bình mỗi bước train: {avg_step:.3f} s")

    # Val Macro-F1 không đổi qua mọi epoch thường là dấu hiệu mô hình sụp về một lớp
    val_f1_constant = num_epochs > 1 and len({round(h['val_macro_f1'], 6) for h in history}) == 1
    if val_f1_constant:
        print(f"CẢNH BÁO: Val Macro-F1 không đổi qua cả {num_epochs} epoch ({best_val_f1:.4f}) — "
              f"có thể mô hình đã sụp về dự đoán một lớp. Kiểm tra log trước khi dùng kết quả.")
    if torch.cuda.is_available():
        print(f"VRAM cấp phát tối đa: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    if args.limit:
        print("Smoke test (--limit): không lưu checkpoint.")
    else:
        ckpt_name = checkpoint_path(MODE, args.seed)
        os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
        torch.save(best_model_wts, ckpt_name)
        # Siêu tham số + lịch sử huấn luyện đi kèm checkpoint (evaluate.py đọc lại)
        meta = {
            'mode': MODE, 'seed': args.seed, 'epochs': num_epochs, 'batch_size': args.batch_size,
            'grad_accum': args.grad_accum, 'lr': args.lr, 'weight_decay': args.weight_decay,
            'scheduler': 'linear_warmup', 'warmup_ratio': args.warmup_ratio, 'warmup_steps': warmup_steps,
            'total_steps': total_steps, 'val_f1_constant': val_f1_constant,
            'max_length': MAX_LENGTH, 'fp16': True, 'n_train': len(train_dataset), 'n_val': len(val_dataset),
            'best_epoch': best_epoch, 'best_val_macro_f1': best_val_f1,
            'train_time_sec': train_time, 'avg_step_sec': avg_step,
            'max_memory_allocated_gb': (torch.cuda.max_memory_allocated() / 1024**3
                                        if torch.cuda.is_available() else None),
            'history': history,
        }
        with open(ckpt_name.replace('.pth', '.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu trọng số TỐT NHẤT (epoch {best_epoch}) thành {ckpt_name}!")
