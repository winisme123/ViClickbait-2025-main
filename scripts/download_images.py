import io
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

from clickbait.scraper import HEADERS, _meta_content

# Cột thumbnail_url trong dataset chỉ chứa đường dẫn file đích
# (vd: data/images/article_0001_image.png), không phải URL ảnh.
# Vì vậy URL ảnh được lấy lại từ thẻ og:image của trang gốc (cột url).

# 1. Đọc dataset
csv_path = os.path.join('data', 'text', 'clickbait_dataset_vietnamese.csv')
df = pd.read_csv(csv_path)
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
if limit:
  df = df.head(limit)

# 2. Tạo thư mục images nếu chưa có
images_dir = os.path.join('data', 'images')
os.makedirs(images_dir, exist_ok=True)
errors_path = os.path.join('data', 'text', 'image_download_errors.csv')


def image_path_for(row):
  # Khớp logic đặt tên file với train.py / evaluate.py
  clean_url = str(row['thumbnail_url']).split('?')[0]
  img_name = os.path.basename(clean_url)
  if not img_name or not img_name.lower().endswith(
      ('.jpg', '.jpeg', '.png', '.webp', '.jfif')
  ):
    img_name = f'thumb_{row.name}.jpg'
  return os.path.join(images_dir, img_name)


def find_baomoi_image(page_url, html):
  # Báo Mới đặt og:image là logo chung; ảnh thật nằm trong __NEXT_DATA__,
  # đường dẫn có chứa ID bài (vd: ...-c52570002.epi -> .../2025_06_23_20_52570002/...)
  article_id = re.search(r'-[a-z](\d+)\.epi', page_url)
  if not article_id:
    return None
  candidates = re.findall(r'"(?:imageUrl|thumb)":"([^"]+)"', html)
  for url in candidates:
    if f'_{article_id.group(1)}/' in url:
      return url
  return None


def find_image_url(page_url):
  resp = requests.get(page_url, headers=HEADERS, timeout=15)
  resp.raise_for_status()
  soup = BeautifulSoup(resp.text, 'html.parser')
  image_url = _meta_content(
      soup,
      {'property': 'og:image'}, {'name': 'og:image'},
      {'property': 'twitter:image'}, {'name': 'twitter:image'},
      {'name': 'twitter:image:src'},
  )
  if 'baomoi.com' in page_url:
    image_url = find_baomoi_image(page_url, resp.text)
    if not image_url:
      raise ValueError(
          f'Báo Mới: không tìm thấy ảnh của bài (trang cuối: {resp.url})'
      )
  if not image_url:
    raise ValueError('Trang không có og:image/twitter:image')
  if image_url.startswith('//'):
    image_url = 'https:' + image_url
  return image_url


def download_image(row):
  img_path = image_path_for(row)

  # Nếu đã tải rồi thì bỏ qua để tiết kiệm thời gian
  if os.path.exists(img_path):
    return None

  image_url = None
  try:
    image_url = find_image_url(row['url'])
    resp = requests.get(
        image_url, headers={**HEADERS, 'Referer': row['url']}, timeout=15
    )
    resp.raise_for_status()
    # Kiểm tra nội dung đúng là ảnh, rồi lưu đúng định dạng theo đuôi file
    image = Image.open(io.BytesIO(resp.content)).convert('RGB')
    image.save(img_path)
    return None
  except Exception as e:
    return {
        'id': row['id'],
        'url': row['url'],
        'image_url': image_url,
        'error': f'{type(e).__name__}: {e}',
    }


print(f'Bắt đầu crawl ảnh cho {len(df)} mẫu dữ liệu...')

# Sử dụng đa luồng (ThreadPoolExecutor) để tải song song
with ThreadPoolExecutor(max_workers=16) as executor:
  results = list(
      executor.map(download_image, [row for _, row in df.iterrows()])
  )

errors = [r for r in results if r]
if errors:
  pd.DataFrame(errors).to_csv(errors_path, index=False, encoding='utf-8-sig')
  print(f'{len(errors)} mẫu lỗi, chi tiết tại {errors_path}')
  for err in errors[:10]:
    print(f"  {err['id']}: {err['error']}")

# Kiểm tra kết quả
found = sum(os.path.exists(image_path_for(row)) for _, row in df.iterrows())
print(f'Tải hoàn tất! Số mẫu đã có ảnh: {found}/{len(df)}')
