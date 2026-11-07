import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor
from sklearn.metrics import classification_report
from torch.amp import autocast

from clickbait.dataset import ViClickbaitDataset, load_split
from clickbait.metrics import CLASS_NAMES, binary_metrics
from clickbait.models import ALL_MODES, MODEL_CLASSES, PHOBERT_NAME, VIT_NAME, checkpoint_path

# Đảm bảo in ký tự tiếng Việt không bị lỗi bảng mã trên Windows console
if sys.platform == 'win32':
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr is not None:
        sys.stderr.reconfigure(encoding='utf-8')

RESULTS_DIR = 'results'

# 0. Tham số dòng lệnh
parser = argparse.ArgumentParser(description='Đánh giá mô hình clickbait trên tập TEST')
parser.add_argument(
    '--mode', type=str, default='multimodal', choices=ALL_MODES,
    help='Chế độ mô hình: text, image, multimodal, concat'
)
parser.add_argument('--batch-size', type=int, default=8, help='Batch size khi đánh giá')
parser.add_argument('--seed', type=int, default=42, help='Seed của file chia tập và checkpoint')
parser.add_argument('--overwrite', action='store_true',
                    help='Cho phép đánh giá lại khi đã có kết quả (tập TEST chỉ nên dùng một lần)')
args = parser.parse_args()
MODE = args.mode

result_path = os.path.join(RESULTS_DIR, f'{MODE}_seed{args.seed}.json')
errors_path = os.path.join(RESULTS_DIR, f'{MODE}_seed{args.seed}_errors.csv')
if os.path.exists(result_path) and not args.overwrite:
    sys.exit(f"Đã có {result_path} — tập TEST chỉ đánh giá một lần. Dùng --overwrite nếu thật sự cần.")


# 1. Dữ liệu: tập TEST của file chia tập theo seed (khớp với lúc train)
test_df = load_split(args.seed, 'test')

# 2. Khởi tạo Tokenizer và Processor
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)
image_processor = ViTImageProcessor.from_pretrained(VIT_NAME)

test_dataset = ViClickbaitDataset(test_df, tokenizer, image_processor, MODE)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL_CLASSES[MODE]().to(device)

    model_weights_path = checkpoint_path(MODE, args.seed)
    if not os.path.exists(model_weights_path):
        sys.exit(f"Không tìm thấy {model_weights_path}! Hãy chạy python -m scripts.train --mode {MODE} --seed {args.seed} trước.")
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    print(f"Đã tải trọng số từ {model_weights_path}!")

    # Siêu tham số + thời gian huấn luyện do train.py ghi kèm checkpoint
    meta_path = model_weights_path.replace('.pth', '.json')
    train_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            train_meta = json.load(f)

    model.eval()
    all_probs = []
    all_labels = []

    print(f"Đang đánh giá mode {MODE.upper()} trên tập Test ({len(test_dataset)} mẫu)...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask', 'pixel_values') if k in batch}

            with autocast("cuda"):
                outputs = model(**inputs)
            all_probs.extend(torch.sigmoid(outputs.float()).cpu().numpy())
            all_labels.extend(batch['label'].numpy())

    probs = np.array(all_probs)
    y_true = np.array(all_labels).astype(int)
    y_pred = (probs >= 0.5).astype(int)

    # 3. Tính toán các chỉ số đánh giá theo yêu cầu khóa luận
    print(f"\n================ KẾT QUẢ ĐÁNH GIÁ [{MODE.upper()}] ================")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=['Non-Clickbait (0)', 'Clickbait (1)'], digits=4, zero_division=0))
    overall = binary_metrics(y_true, y_pred)
    print(f"Macro F1-score chính: {overall['macro_f1']:.4f}")
    print("\nConfusion Matrix (Ma trận nhầm lẫn):")
    print(np.array(overall['confusion_matrix']))

    # Macro-F1 và F1 lớp clickbait theo từng nguồn báo
    sources = test_df['source'].to_numpy()
    by_source = {}
    for source in sorted(set(sources)):
        idx = sources == source
        m = binary_metrics(y_true[idx], y_pred[idx])
        by_source[source] = {
            'n_samples': m['n_samples'],
            'n_clickbait': int(y_true[idx].sum()),
            'accuracy': m['accuracy'],
            'macro_f1': m['macro_f1'],
            'f1_clickbait': m['per_class']['clickbait']['f1'],
        }
    print("\nTheo nguồn báo:")
    print(pd.DataFrame(by_source).T.to_string(float_format=lambda x: f'{x:.4f}'))
    print("================================================================")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        'mode': MODE,
        'seed': args.seed,
        'split': 'test',
        'threshold': 0.5,
        'checkpoint': model_weights_path,
        **overall,
        'by_source': by_source,
        'hyperparameters': {k: train_meta.get(k) for k in (
            'epochs', 'batch_size', 'grad_accum', 'lr', 'weight_decay', 'scheduler', 'warmup_ratio',
            'warmup_steps', 'total_steps', 'val_f1_constant', 'max_length', 'fp16',
            'n_train', 'n_val', 'best_epoch', 'best_val_macro_f1')},
        'train_time_sec': train_meta.get('train_time_sec'),
        'avg_step_sec': train_meta.get('avg_step_sec'),
        'max_memory_allocated_gb': train_meta.get('max_memory_allocated_gb'),
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả vào {result_path}")

    # Các mẫu dự đoán sai
    wrong = y_pred != y_true
    errors = pd.DataFrame({
        'id': test_df['id'].to_numpy()[wrong],
        'source': sources[wrong],
        'prob_clickbait': probs[wrong].round(4),
        'true_label': np.array(CLASS_NAMES)[y_true[wrong]],
        'pred_label': np.array(CLASS_NAMES)[y_pred[wrong]],
        'title': test_df['title'].to_numpy()[wrong],
    })
    errors.to_csv(errors_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu {len(errors)} mẫu dự đoán sai vào {errors_path}")
