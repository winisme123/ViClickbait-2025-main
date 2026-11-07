from bs4 import BeautifulSoup

from clickbait.scraper import _meta_content, find_baomoi_image


def test_meta_content_skips_url_values():
  # Báo Mới đặt og:title bằng chính URL bài viết
  soup = BeautifulSoup(
      '<meta property="og:title" content="https://baomoi.com/a-c123.epi">'
      '<meta name="twitter:title" content="Tiêu đề thật">', 'html.parser')
  assert _meta_content(soup, {'property': 'og:title'}) == 'https://baomoi.com/a-c123.epi'
  assert _meta_content(soup, {'property': 'og:title'}, {'name': 'twitter:title'},
                       skip_urls=True) == 'Tiêu đề thật'


def test_find_baomoi_image_matches_article_id():
  html = ('{"thumb":"https://x/2025_06_23_1_99999/other.jpg",'
          '"imageUrl":"https://x/2025_06_23_20_52570002/real.jpg"}')
  url = 'https://baomoi.com/tieu-de-r52570002.epi'
  assert find_baomoi_image(url, html) == 'https://x/2025_06_23_20_52570002/real.jpg'


def test_find_baomoi_image_none_without_match():
  assert find_baomoi_image('https://baomoi.com/khong-co-id', '{}') is None
  assert find_baomoi_image('https://baomoi.com/a-c1.epi', '{"thumb":"https://x/logo.png"}') is None
