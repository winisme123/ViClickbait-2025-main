import pandas as pd
from sklearn.model_selection import train_test_split
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. Đọc bộ dữ liệu gốc
csv_path = 'clickbait_dataset_vietnamese.csv'
df = pd.read_csv(csv_path)

print(f"Tổng số mẫu gốc: {len(df)}")
print(f"Phân phối nhãn:\n{df['label'].value_counts()}")

# 2. Chia tập Stratified: 80% Train, 10% Validation, 10% Test
# Đảm bảo giữ nguyên tỷ lệ clickbait / non-clickbait trên cả 3 tập
train_df, temp_df = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['label']
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    random_state=42, 
    stratify=temp_df['label']
)

print(f"\nSố lượng mẫu:")
print(f"- Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"- Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"- Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# Gán cột 'split' vào bộ dữ liệu gốc để dễ quản lý
df.loc[train_df.index, 'split'] = 'train'
df.loc[val_df.index, 'split'] = 'val'
df.loc[test_df.index, 'split'] = 'test'

df.to_csv('clickbait_dataset_with_splits.csv', index=False)
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("\nĐã tạo thành công các tệp: clickbait_dataset_with_splits.csv, train.csv, val.csv, test.csv!")
