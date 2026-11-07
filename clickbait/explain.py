"""Ghép kết quả giải thích cho demo: đóng góp của từng từ (từ mô hình) + dấu hiệu theo luật.

- Đóng góp từng từ: `inference.word_importance` (Shapley theo hoán vị, đơn vị log-odds).
  Đây là lý do *của mô hình*.
- Dấu hiệu theo luật: `signals.extract_signals`. Đây là bộ luật ngôn ngữ để người
  đọc dễ hiểu, không phải lý do bên trong mô hình.
"""
from dataclasses import dataclass, field

from .signals import CATEGORIES, _escape

# Đóng góp (log-odds) tối thiểu để một từ được nêu thành "điểm" riêng
MIN_WORD_EFFECT = 0.3
MAX_POINTS = 5
# Dấu câu tự nó mang tín hiệu (các dấu câu khác không nêu thành điểm riêng)
MEANINGFUL_PUNCT = {'?', '!', '...', '…'}

TIPS = {
    'curiosity': 'Tin chính thống thường nói thẳng nội dung chính ngay trong tiêu đề, '
                 'không giấu thông tin để buộc người đọc nhấp vào.',
    'sensational': 'Tin chính thống dùng từ trung tính, mô tả sự việc thay vì phóng đại cảm xúc.',
    'question': 'Câu hỏi hoặc dấu chấm than khiến người đọc phải nhấp vào mới biết câu trả lời.',
    'listicle': 'Tiêu đề dạng danh sách ("N cách", "top N") hứa hẹn nội dung dễ đọc để thu hút lượt nhấp.',
    'address': 'Gọi thẳng người đọc tạo cảm giác liên quan cá nhân, kéo người đọc nhấp vào.',
    'emphasis': 'Viết hoa cả từ hoặc lặp dấu câu để gây chú ý mạnh; hiếm gặp ở báo chính thống.',
}


@dataclass
class Point:
  """Một "điểm" giải thích hiển thị thành một mục bấm mở được."""
  text: str                 # cụm từ trong tiêu đề
  direction: str            # 'clickbait' hoặc 'non-clickbait'
  effect: float             # tổng đóng góp log-odds của các từ thuộc cụm (từ mô hình)
  label: str                # tên loại dấu hiệu
  reason: str               # vì sao cụm này giống clickbait / tin chính thống
  tip: str = ''
  from_rules: bool = False  # True nếu bộ luật ngôn ngữ cũng nhận ra cụm này
  words: list = field(default_factory=list)


def _is_punct(word):
  return not any(ch.isalnum() for ch in word)


def _words_in(span_text, contributions):
  """Các từ (theo pyvi) nằm trong một cụm của bộ luật, khớp không phân biệt hoa thường."""
  span = span_text.lower()
  return [(i, w, d) for i, (w, d) in enumerate(contributions) if w.lower() in span]


def _model_phrases(contributions, used):
  """Gộp các từ liền nhau cùng chiều (|đóng góp| ≥ MIN_WORD_EFFECT / 2) thành cụm.

  Bỏ các từ đã thuộc dấu hiệu của bộ luật và dấu câu không mang tín hiệu.
  Trả về list[(cụm, tổng đóng góp, các từ)].
  """
  phrases, run = [], []

  def flush():
    if run:
      words = [w for _, w, _ in run]
      phrases.append((' '.join(words), sum(d for _, _, d in run), words))
      run.clear()

  for i, (word, d) in enumerate(contributions):
    keep = (i not in used and abs(d) >= MIN_WORD_EFFECT / 2
            and not (_is_punct(word) and word not in MEANINGFUL_PUNCT))
    if keep and run and run[-1][0] == i - 1 and (run[-1][2] > 0) == (d > 0):
      run.append((i, word, d))
    else:
      flush()
      if keep:
        run.append((i, word, d))
  flush()
  return [p for p in phrases if abs(p[1]) >= MIN_WORD_EFFECT]


def build_points(contributions, signals):
  """Trả về (điểm giống clickbait, điểm giống tin chính thống).

  - Mọi dấu hiệu của bộ luật đều thành một điểm clickbait (không bị cắt), kèm
    đóng góp thật của mô hình cho các từ trong cụm (có thể âm: luật nhận ra
    nhưng mô hình không coi trọng).
  - Thêm các cụm từ mô hình coi là quan trọng mà bộ luật không nhận ra, cho tới
    khi đủ MAX_POINTS điểm mỗi phía.
  """
  used = set()
  rule_points = []
  for sig in signals:
    matched = _words_in(sig.text, contributions)
    used.update(i for i, _, _ in matched)
    rule_points.append(Point(
        text=sig.text, direction='clickbait', effect=sum(d for _, _, d in matched),
        label=CATEGORIES[sig.category][1], reason=sig.explanation,
        tip=TIPS.get(sig.category, ''), from_rules=True, words=[w for _, w, _ in matched]))
  rule_points.sort(key=lambda p: -p.effect)

  model_cb, normal_points = [], []
  for text, effect, words in _model_phrases(contributions, used):
    if effect > 0:
      model_cb.append(Point(
          text=text, direction='clickbait', effect=effect, label='Mô hình học được từ dữ liệu',
          reason='Trong dữ liệu huấn luyện, cách dùng từ này hay xuất hiện ở tiêu đề clickbait, '
                 'nên mô hình coi đây là dấu hiệu giật tít.', words=words))
    else:
      normal_points.append(Point(
          text=text, direction='non-clickbait', effect=effect, label='Văn phong tin tức',
          reason='Trong dữ liệu huấn luyện, cách dùng từ này hay xuất hiện ở tiêu đề tin chính thống '
                 '(thông tin cụ thể, trung tính), nên kéo kết quả về phía chính thống.',
          words=words))
  model_cb.sort(key=lambda p: -p.effect)
  normal_points.sort(key=lambda p: p.effect)
  room = max(MAX_POINTS - len(rule_points), 0)
  return rule_points + model_cb[:room], normal_points[:MAX_POINTS]


def summary_sentence(prob, contributions, signals, image_effect=None):
  """Một đoạn tóm tắt lý do bằng lời (markdown)."""
  is_cb = prob >= 0.5
  ranked = sorted(contributions, key=lambda wd: -wd[1] if is_cb else wd[1])
  key_words = [w for w, d in ranked
               if (d > 0 if is_cb else d < 0) and not (_is_punct(w) and w not in MEANINGFUL_PUNCT)][:3]
  quoted = ', '.join(f'«{w}»' for w in key_words)
  if is_cb:
    text = f'Mô hình nghiêng về **clickbait** ({prob*100:.1f}%)'
    text += f', chủ yếu do: {quoted}.' if quoted else '.'
    if signals:
      labels = list(dict.fromkeys(CATEGORIES[s.category][1].lower() for s in signals))
      text += f' Tiêu đề có {len(signals)} dấu hiệu giật tít quen thuộc ({", ".join(labels)}).'
  else:
    text = f'Mô hình nghiêng về **tin chính thống** ({(1-prob)*100:.1f}%)'
    text += f', chủ yếu do các từ mang văn phong tin tức: {quoted}.' if quoted else '.'
    if signals:
      text += (f' Bộ luật vẫn thấy {len(signals)} dấu hiệu giống clickbait, nhưng phần còn lại '
               'của tiêu đề kéo kết quả về phía chính thống.')
    else:
      text += ' Không có dấu hiệu giật tít quen thuộc nào.'
  if image_effect is not None:
    with_img, without = image_effect
    diff = (with_img - without) * 100
    if abs(diff) < 2:
      text += ' Ảnh thumbnail gần như không ảnh hưởng tới kết luận.'
    else:
      text += (f' Ảnh thumbnail làm xác suất clickbait {"tăng" if diff > 0 else "giảm"} '
               f'{abs(diff):.1f} điểm phần trăm.')
  return text


def render_contributions(contributions):
  """HTML tiêu đề, mỗi từ tô màu theo đóng góp: đỏ = kéo về clickbait, xanh = về chính thống."""
  peak = max((abs(d) for _, d in contributions), default=0) or 1.0
  parts = []
  for word, d in contributions:
    alpha = min(abs(d) / peak, 1.0) * 0.75
    rgb = '230,57,70' if d > 0 else '29,111,220'
    side = 'clickbait' if d > 0 else 'chính thống'
    tooltip = _escape(f'{word}: kéo về {side} ({d:+.2f})')
    parts.append(f'<span title="{tooltip}" style="background:rgba({rgb},{alpha:.2f}); '
                 f'border-radius:4px; padding:1px 3px; margin:0 1px; cursor:help;">{_escape(word)}</span>')
  return ' '.join(parts)
