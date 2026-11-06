from clickbait.explain import build_points, render_contributions, summary_sentence
from clickbait.signals import extract_signals


def test_rule_points_always_kept_and_phrases_merged():
  title = 'SỐC! Sự thật khiến ai cũng ngã ngửa'
  contrib = [('SỐC', 0.05), ('!', -0.1), ('Sự thật', 0.2), ('khiến', 0.1),
             ('ai', 0.4), ('cũng', 0.5), ('ngã ngửa', 0.9)]
  cb, normal = build_points(contrib, extract_signals(title))
  texts = [p.text for p in cb]
  # Mọi dấu hiệu của bộ luật đều có mặt, kể cả khi mô hình cho điểm thấp
  assert {'!', 'Sự thật'} <= set(texts)
  # Các từ liền nhau cùng chiều được gộp thành một cụm
  assert 'ai cũng ngã ngửa' in texts
  assert normal == []


def test_non_clickbait_points_and_summary():
  contrib = [('Sân bay', -0.9), ('Vinh', -0.2), ('đóng cửa', -0.6), ('6', 0.05), ('tháng', -0.1)]
  cb, normal = build_points(contrib, [])
  assert cb == []
  assert normal[0].text == 'Sân bay Vinh đóng cửa' and normal[0].effect < 0
  text = summary_sentence(0.03, contrib, [], image_effect=(0.03, 0.025))
  assert 'tin chính thống' in text and 'gần như không ảnh hưởng' in text


def test_render_escapes_html():
  html = render_contributions([('<b>x</b>', 0.5), ('y', -0.5)])
  assert '<b>x</b>' not in html and '&lt;b&gt;' in html
