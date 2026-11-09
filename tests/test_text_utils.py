import unicodedata

from clickbait.text_utils import normalize_text, segment_text


def test_normalize_text_unifies_decomposed_diacritics():
  composed = 'Sân bay Vinh đóng cửa'
  decomposed = unicodedata.normalize('NFD', composed)
  assert decomposed != composed
  assert normalize_text(decomposed) == composed
  # Sau chuẩn hóa, hai cách gõ dấu cho cùng kết quả tách từ
  assert segment_text(normalize_text(decomposed)) == segment_text(composed)


def test_normalize_text_strips_invisible_and_extra_spaces():
  assert normalize_text('﻿  Bạn​ sẽ   không tin  \n') == 'Bạn sẽ không tin'


def test_normalize_text_keeps_clean_text_unchanged():
  title = 'SỐC! Bật mí sự thật kinh hoàng đằng sau vẻ đẹp của hot girl'
  assert normalize_text(title) == title
