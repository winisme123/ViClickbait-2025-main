"""Dữ liệu dùng chung cho train / evaluate: danh sách mẫu hợp lệ, chia tập, Dataset.

Chạy trực tiếp để tạo file chia tập và in thống kê:
    python -m clickbait.dataset --seed 42
"""
import argparse
import os
import sys

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .text_utils import segment_text

CSV_PATH = os.path.join('data', 'text', 'clickbait_dataset_vietnamese.csv')
VALID_PATH = os.path.join('data', 'text', 'valid_samples.csv')
IMAGES_DIR = os.path.join('data', 'images')
SPLITS_DIR = os.path.join('data', 'splits')

# Độ dài token lớn nhất của tiêu đề (sau pyvi + PhoBERT tokenizer) trên các mẫu
# hợp lệ là 38, làm tròn lên bội của 16.
MAX_LENGTH = 48
# Mode *_lead (tiêu đề + đoạn mở đầu): phân vị 95 độ dài token trên train seed 42
# là 73,2, làm tròn lên bội của 16 (docs/protocol-v3.md, mục B).
LEAD_MAX_LENGTH = 80


def max_length_for(mode):
  return LEAD_MAX_LENGTH if mode.endswith('_lead') else MAX_LENGTH


def image_path_for(row):
  # Khớp logic đặt tên file với download_images.py
  clean_url = str(row['thumbnail_url']).split('?')[0]
  img_name = os.path.basename(clean_url)
  if not img_name or not img_name.lower().endswith(
      ('.jpg', '.jpeg', '.png', '.webp', '.jfif')
  ):
    img_name = f'thumb_{row.name}.jpg'
  return os.path.join(IMAGES_DIR, img_name)


def split_path(seed):
  return os.path.join(SPLITS_DIR, f'split_seed{seed}.csv')


def make_split(seed):
  """Chia 70/15/15 phân tầng theo nhãn trên các mẫu hợp lệ."""
  df = pd.read_csv(CSV_PATH)
  valid_ids = set(pd.read_csv(VALID_PATH)['id'])
  df = df[df['id'].isin(valid_ids)]

  train, rest = train_test_split(
      df, test_size=0.30, stratify=df['label'], random_state=seed)
  val, test = train_test_split(
      rest, test_size=0.50, stratify=rest['label'], random_state=seed)

  split = pd.concat([
      pd.DataFrame({'id': part['id'].to_numpy(), 'split': name})
      for name, part in (('train', train), ('val', val), ('test', test))
  ])
  os.makedirs(SPLITS_DIR, exist_ok=True)
  split.to_csv(split_path(seed), index=False)
  return split


def load_split(seed, split):
  """Trả về DataFrame các mẫu thuộc tập `split` (train/val/test) của seed."""
  path = split_path(seed)
  splits = pd.read_csv(path) if os.path.exists(path) else make_split(seed)
  ids = set(splits.loc[splits['split'] == split, 'id'])
  df = pd.read_csv(CSV_PATH)
  return df[df['id'].isin(ids)]


class ViClickbaitDataset(Dataset):
  """Mode text không nạp ảnh, mode image không tokenize tiêu đề.

  Mode có hậu tố _lead đưa cặp câu (tiêu đề, đoạn mở đầu) vào PhoBERT; nếu quá dài
  thì chỉ cắt đoạn mở đầu.
  """

  def __init__(self, dataframe, tokenizer, processor, mode, max_length=None):
    self.df = dataframe
    self.tokenizer = tokenizer
    self.processor = processor
    self.use_lead = mode.endswith('_lead')
    base = mode.removesuffix('_lead')
    self.use_text = base != 'image'
    self.use_image = base != 'text'
    self.max_length = max_length or max_length_for(mode)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    item = {
        'label': torch.tensor(
            1.0 if row['label'] == 'clickbait' else 0.0, dtype=torch.float)
    }

    if self.use_text:
      # PhoBERT nhận tiêu đề đã tách từ
      title = segment_text(str(row['title']))
      if self.use_lead:
        lead = row['lead_paragraph']
        lead = segment_text(str(lead)) if isinstance(lead, str) else ''
        text_encoding = self.tokenizer(
            title, lead, max_length=self.max_length, padding='max_length',
            truncation='only_second', return_tensors='pt')
      else:
        text_encoding = self.tokenizer(
            title, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt')
      item['input_ids'] = text_encoding['input_ids'].squeeze(0)
      item['attention_mask'] = text_encoding['attention_mask'].squeeze(0)

    if self.use_image:
      # Danh sách mẫu hợp lệ đảm bảo ảnh tồn tại, lỗi mở ảnh không bị bỏ qua
      image = Image.open(image_path_for(row)).convert('RGB')
      item['pixel_values'] = self.processor(
          images=image, return_tensors='pt')['pixel_values'].squeeze(0)

    return item


def print_split_stats(seed):
  for name in ('train', 'val', 'test'):
    df = load_split(seed, name)
    n_cb = (df['label'] == 'clickbait').sum()
    print(f'{name:5s}: {len(df):4d} mẫu | clickbait {n_cb:4d} '
          f'({n_cb / len(df):.2%})')


if __name__ == '__main__':
  if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
  parser = argparse.ArgumentParser(description='Tạo file chia tập 70/15/15')
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  make_split(args.seed)
  print(f'Đã lưu {split_path(args.seed)}')
  print_split_stats(args.seed)
