import os

import pytest
import torch

from clickbait.dataset import ViClickbaitDataset, load_split
from clickbait.inference import load_pipeline, predict_proba
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
