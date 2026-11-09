import math
import os

import pytest
import torch

from clickbait.dataset import ViClickbaitDataset, load_split
from clickbait.inference import image_contribution, load_pipeline, predict_proba, word_importance
from clickbait.models import checkpoint_path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(checkpoint_path('text', 42)),
                       reason='chưa có checkpoints/text_seed42.pth'),
]


@pytest.fixture(scope='module')
def text_pipeline():
  return load_pipeline('text', device=torch.device('cpu'))


def test_demo_loads_checkpoint(text_pipeline):
  assert text_pipeline['loaded']


def test_demo_matches_evaluation_preprocessing(text_pipeline):
  # Demo phải cho cùng xác suất với đường đánh giá (Dataset của scripts/evaluate.py)
  row_df = load_split(42, 'test').head(3)
  ds = ViClickbaitDataset(row_df, text_pipeline['tokenizer'], None, 'text')
  model = text_pipeline['model']
  for i in range(len(row_df)):
    item = ds[i]
    with torch.no_grad():
      expected = torch.sigmoid(model(item['input_ids'][None], item['attention_mask'][None])).item()
    got = predict_proba(text_pipeline, str(row_df.iloc[i]['title']), None)
    assert got == pytest.approx(expected, abs=1e-6)


def test_image_mode_requires_image(text_pipeline):
  # Không ghép ảnh giả: mode cần ảnh mà thiếu ảnh thì báo lỗi rõ ràng
  pipeline = dict(text_pipeline, mode='multimodal')
  with pytest.raises(ValueError):
    predict_proba(pipeline, 'tiêu đề', None)


def test_word_importance_consistent_with_prediction(text_pipeline):
  title = 'SỐC! Sự thật đằng sau vẻ đẹp của hot girl khiến ai cũng ngã ngửa'
  base, contrib = word_importance(text_pipeline, title)
  assert base == pytest.approx(predict_proba(text_pipeline, title, None), abs=1e-4)
  assert [w for w, _ in contrib][:2] == ['SỐC', '!']
  # Tính chất hiệu quả của Shapley: tổng đóng góp = logit(tiêu đề) - logit(tiêu đề rỗng)
  logit = lambda q: math.log(q / (1 - q))
  empty = predict_proba(text_pipeline, '', None)
  assert sum(d for _, d in contrib) == pytest.approx(logit(base) - logit(empty), abs=1e-2)
  # Chạy lại cho cùng kết quả (seed cố định)
  assert word_importance(text_pipeline, title)[1] == contrib


def test_image_contribution_none_for_text_mode(text_pipeline):
  assert image_contribution(text_pipeline, 'tiêu đề', None) is None


@pytest.mark.parametrize('mode', ['concat', 'multimodal'])
def test_cached_image_features_match_full_forward(mode):
  # Dùng lại đặc trưng ảnh phải cho cùng kết quả với chạy đầy đủ mô hình
  from PIL import Image
  from clickbait.inference import _batch_proba
  if not os.path.exists(checkpoint_path(mode, 42)):
    pytest.skip(f'chưa có checkpoints/{mode}_seed42.pth')
  mm = load_pipeline(mode, device=torch.device('cpu'))
  assert mm['loaded']
  img = Image.new('RGB', (300, 200), (200, 30, 30))
  titles = ['Sân bay Vinh đóng cửa 6 tháng', 'SỐC! Sự thật kinh hoàng']
  pix = mm['processor'](images=img, return_tensors='pt')['pixel_values']
  cached = _batch_proba(mm, [t for t in titles], pix)
  for t, c in zip(titles, cached):
    with torch.no_grad():
      ids = mm['tokenizer'](t, max_length=48, padding='max_length', truncation=True, return_tensors='pt')
      full = torch.sigmoid(mm['model'](ids['input_ids'], ids['attention_mask'], pix)).item()
    assert c == pytest.approx(full, abs=1e-5)
  assert not hasattr(mm['model'].image_encoder, 'hidden')  # ViT gốc được trả lại sau khi giải thích
