import base64
import io

from PIL import Image

from clickbait.ui_paste import MAX_BYTES, decode_data_url


def _data_url(fmt='PNG', mime='image/png', size=(40, 30)):
  buf = io.BytesIO()
  Image.new('RGBA', size, (10, 20, 30, 255)).save(buf, format=fmt)
  return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode()


def test_decode_valid_image_to_rgb():
  img = decode_data_url(_data_url())
  assert img.size == (40, 30) and img.mode == 'RGB'


def test_decode_rejects_bad_input():
  assert decode_data_url(None) is None
  assert decode_data_url('') is None
  assert decode_data_url('data:text/plain;base64,aGVsbG8=') is None
  assert decode_data_url('data:image/png;base64,@@@không-phải-base64') is None
  assert decode_data_url('data:image/png;base64,' + base64.b64encode(b'not an image').decode()) is None


def test_decode_rejects_oversized(monkeypatch):
  monkeypatch.setattr('clickbait.ui_paste.MAX_BYTES', 10)
  assert decode_data_url(_data_url()) is None
  assert MAX_BYTES == 10 * 1024 * 1024
