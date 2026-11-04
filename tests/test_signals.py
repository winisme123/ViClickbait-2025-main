import pytest

from clickbait.signals import extract_signals


def categories(title):
  return [(s.text, s.category) for s in extract_signals(title)]


@pytest.mark.parametrize('title', ['Không thể bỏ qua món ăn này', 'khong the bo qua mon an nay'])
def test_khong_the_bo_qua_with_and_without_accents(title):
  assert categories(title) == [(title[:16], 'sensational')]


def test_curiosity_and_emphasis():
  assert categories('Bạn sẽ không tin điều này!!') == [
      ('Bạn sẽ không tin', 'curiosity'), ('!!', 'emphasis')]


@pytest.mark.parametrize('title', ['Giá vàng hôm nay tăng mạnh',
                                   'Thủ tướng chủ trì hội nghị phát triển kinh tế'])
def test_neutral_titles_have_no_signals(title):
  assert extract_signals(title) == []


def test_empty_title():
  assert extract_signals('') == []
