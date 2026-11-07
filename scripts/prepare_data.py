"""Tạo danh sách mẫu hợp lệ dùng chung cho mọi mô hình.

Loại mẫu khi:
  - không có ảnh trong data/images (tên file theo cột thumbnail_url)
  - là trang chuyên mục: url không chứa ID bài (>= 5 chữ số) hoặc tiêu đề <= 2 từ
  - ảnh là ảnh giữ chỗ của trang báo (không phải thumbnail thật)

Mặc định chỉ in báo cáo + ghi data/text/exclusion_review.csv để duyệt.
Chạy với --apply để ghi data/text/valid_samples.csv.
"""
import argparse
import hashlib
import os

import pandas as pd

from clickbait.dataset import CSV_PATH as csv_path, VALID_PATH as valid_path, image_path_for

review_path = os.path.join('data', 'text', 'exclusion_review.csv')

# md5 của ảnh giữ chỗ (logo xám) mà trang báo trả về khi bài không có ảnh
PLACEHOLDER_MD5 = {
    '4db13668019d8a429878b824522f5c04',  # VnExpress
}


def exclusion_reasons(row):
  reasons = []
  if not os.path.exists(row['image_path']):
    reasons.append('thieu_anh')
  elif md5(row['image_path']) in PLACEHOLDER_MD5:
    reasons.append('anh_giu_cho')
  if not pd.Series([row['url']]).str.contains(r'\d{5,}').iloc[0]:
    reasons.append('url_khong_phai_bai')
  if len(str(row['title']).split()) <= 2:
    reasons.append('tieu_de_<=2_tu')
  return ';'.join(reasons)


def summarize(df, name):
  n = len(df)
  n_cb = (df['label'] == 'clickbait').sum()
  print(f'{name}: {n} mẫu, clickbait {n_cb} ({n_cb / n:.2%})')


def md5(path):
  with open(path, 'rb') as f:
    return hashlib.md5(f.read()).hexdigest()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--apply', action='store_true',
                      help='Ghi danh sách mẫu hợp lệ ra valid_samples.csv')
  args = parser.parse_args()

  df = pd.read_csv(csv_path)
  df['image_path'] = df.apply(image_path_for, axis=1)
  df['reason'] = df.apply(exclusion_reasons, axis=1)
  excluded = df[df['reason'] != '']
  kept = df[df['reason'] == '']

  print('=== CÁC DÒNG SẼ LOẠI (trừ lý do chỉ thiếu ảnh) ===')
  category = excluded[excluded['reason'] != 'thieu_anh']
  with pd.option_context('display.width', 250, 'display.max_colwidth', 80,
                         'display.max_rows', None):
    print(category[['id', 'source', 'label', 'reason', 'url', 'title']]
          .to_string(index=False))
  only_missing = excluded[excluded['reason'] == 'thieu_anh']
  print(f'\nThiếu ảnh (không kèm lý do khác): {len(only_missing)} dòng, '
        f'theo nguồn: {only_missing["source"].value_counts().to_dict()}')
  excluded[['id', 'source', 'label', 'reason', 'url', 'title']].to_csv(
      review_path, index=False, encoding='utf-8-sig')
  print(f'Toàn bộ {len(excluded)} dòng loại được ghi tại {review_path}')

  print('\n=== TRƯỚC / SAU ===')
  summarize(df, 'Trước')
  summarize(kept, 'Sau  ')
  by_source = pd.DataFrame({
      'truoc': df['source'].value_counts(),
      'sau': kept['source'].value_counts(),
  }).fillna(0).astype(int)
  by_source['cb_truoc_%'] = (
      df.groupby('source')['label'].apply(lambda s: (s == 'clickbait').mean())
      * 100).round(1)
  by_source['cb_sau_%'] = (
      kept.groupby('source')['label'].apply(lambda s: (s == 'clickbait').mean())
      * 100).round(1)
  print(by_source.sort_values('truoc', ascending=False).to_string())

  print('\n=== NHÓM ẢNH TRÙNG MD5 (trong các mẫu giữ lại, chỉ liệt kê) ===')
  kept = kept.assign(md5=kept['image_path'].map(md5))
  dup = kept[kept['md5'].duplicated(keep=False)].sort_values(['md5', 'id'])
  for h, g in dup.groupby('md5'):
    print(f'{h} ({len(g)} ảnh):')
    for _, r in g.iterrows():
      print(f'   {r["id"]} [{r["source"]}, {r["label"]}] {r["title"][:70]}')
  if dup.empty:
    print('Không có.')

  if args.apply:
    kept[['id']].to_csv(valid_path, index=False)
    print(f'\nĐã ghi {len(kept)} mẫu hợp lệ vào {valid_path}')
  else:
    print('\nChưa ghi danh sách mẫu hợp lệ (chạy lại với --apply sau khi duyệt).')


if __name__ == '__main__':
  main()
