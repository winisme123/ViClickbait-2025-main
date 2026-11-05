"""Kiểm tra danh sách mẫu hợp lệ và file chia tập 70/15/15."""
import os

import pandas as pd
import pytest

from clickbait import dataset

SEEDS = [42, 43, 44]


@pytest.fixture(scope='module')
def valid_ids():
  return set(pd.read_csv(dataset.VALID_PATH)['id'])


def test_valid_samples_excludes_reviewed_rows(valid_ids):
  excluded = set(pd.read_csv(os.path.join('data', 'text', 'exclusion_review.csv'))['id'])
  n_total = len(pd.read_csv(dataset.CSV_PATH))
  assert not valid_ids & excluded
  assert len(valid_ids) + len(excluded) == n_total


@pytest.mark.parametrize('seed', SEEDS)
def test_split_sizes_and_disjoint(seed, valid_ids):
  parts = {name: set(dataset.load_split(seed, name)['id']) for name in ('train', 'val', 'test')}
  assert {k: len(v) for k, v in parts.items()} == {'train': 2237, 'val': 480, 'test': 480}
  assert not parts['train'] & parts['val']
  assert not parts['train'] & parts['test']
  assert not parts['val'] & parts['test']
  assert parts['train'] | parts['val'] | parts['test'] == valid_ids


@pytest.mark.parametrize('seed', SEEDS)
def test_split_is_stratified(seed):
  ratios = [(dataset.load_split(seed, name)['label'] == 'clickbait').mean()
            for name in ('train', 'val', 'test')]
  assert max(ratios) - min(ratios) < 0.01


@pytest.mark.parametrize('seed', SEEDS)
def test_split_file_is_reproducible(seed, tmp_path, monkeypatch):
  # Tạo lại file chia tập vào thư mục tạm, phải trùng khớp file đã dùng để huấn luyện
  saved = pd.read_csv(dataset.split_path(seed))
  monkeypatch.setattr(dataset, 'SPLITS_DIR', str(tmp_path))
  regenerated = dataset.make_split(seed).reset_index(drop=True)
  pd.testing.assert_frame_equal(regenerated, saved)


def test_image_path_for_fallback():
  row = pd.Series({'thumbnail_url': 'https://example.com/anh?size=1'}, name=7)
  assert dataset.image_path_for(row) == os.path.join(dataset.IMAGES_DIR, 'thumb_7.jpg')
  row = pd.Series({'thumbnail_url': 'data/images/article_0001_image.png'}, name=0)
  assert dataset.image_path_for(row) == os.path.join(dataset.IMAGES_DIR, 'article_0001_image.png')


class _FakeTokenizer:
  """Ghi lại tham số gọi tokenizer, trả tensor rỗng đúng độ dài."""

  def __init__(self):
    self.calls = []

  def __call__(self, *texts, max_length, truncation, **kwargs):
    import torch
    self.calls.append({'texts': texts, 'max_length': max_length, 'truncation': truncation})
    return {'input_ids': torch.zeros(1, max_length, dtype=torch.long),
            'attention_mask': torch.ones(1, max_length, dtype=torch.long)}


def test_max_length_for_modes():
  assert dataset.max_length_for('text') == dataset.MAX_LENGTH == 48
  assert dataset.max_length_for('concat') == 48
  assert dataset.max_length_for('text_lead') == dataset.LEAD_MAX_LENGTH == 80


def test_lead_mode_uses_sentence_pair_and_truncates_lead_only():
  df = pd.DataFrame({'title': ['Tiêu đề A', 'Tiêu đề B'], 'lead_paragraph': ['Đoạn mở đầu', float('nan')],
                     'label': ['clickbait', 'non-clickbait']})
  tok = _FakeTokenizer()
  ds = dataset.ViClickbaitDataset(df, tok, None, 'text_lead')
  assert ds.use_text and not ds.use_image
  item = ds[0]
  assert item['input_ids'].shape[0] == 80
  assert len(tok.calls[0]['texts']) == 2 and tok.calls[0]['truncation'] == 'only_second'
  ds[1]  # đoạn mở đầu bị thiếu → chuỗi rỗng, không lỗi
  assert tok.calls[1]['texts'][1] == ''


def test_title_mode_unchanged():
  df = pd.DataFrame({'title': ['Tiêu đề'], 'lead_paragraph': ['x'], 'label': ['clickbait']})
  tok = _FakeTokenizer()
  dataset.ViClickbaitDataset(df, tok, None, 'text')[0]
  assert len(tok.calls[0]['texts']) == 1 and tok.calls[0]['max_length'] == 48
  assert tok.calls[0]['truncation'] is True
