"""Ô dán ảnh bằng Ctrl+V cho demo Streamlit (st.components.v2).

Người dùng copy ảnh (chuột phải → Sao chép hình ảnh, hoặc chụp màn hình bằng
Win+Shift+S) rồi nhấn Ctrl+V ở bất kỳ đâu trên trang. JavaScript bắt sự kiện
paste, thu nhỏ ảnh (cạnh dài tối đa 1280 px, JPEG) rồi gửi về Python dưới dạng
data URL. Dán chữ vào ô tiêu đề vẫn hoạt động bình thường vì chỉ chặn khi
clipboard có ảnh.
"""
import base64
import binascii
import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024

_HTML = """
<div class="paste-box" tabindex="0">
  <div class="paste-title">📋 Dán ảnh bằng <b>Ctrl+V</b></div>
  <div class="paste-hint">Copy ảnh (chuột phải → <i>Sao chép hình ảnh</i>) hoặc chụp màn hình
  bằng <b>Win+Shift+S</b>, rồi nhấn Ctrl+V ở bất kỳ đâu trên trang.</div>
  <div class="paste-status"></div>
</div>
"""

_CSS = """
.paste-box {
  border: 2px dashed rgba(128, 128, 128, 0.55);
  border-radius: 10px;
  padding: 12px 14px;
  cursor: pointer;
  font-family: inherit;
  color: var(--st-text-color, inherit);
}
.paste-box:focus, .paste-box.active { border-color: #ff4b4b; outline: none; }
.paste-title { font-size: 1rem; margin-bottom: 4px; }
.paste-hint { font-size: 0.85rem; opacity: 0.75; }
.paste-status { font-size: 0.85rem; margin-top: 6px; font-weight: 600; }
"""

_JS = """
export default function (component) {
  const { setStateValue, parentElement } = component;
  const box = parentElement.querySelector('.paste-box');
  const status = parentElement.querySelector('.paste-status');

  function shrink(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, 1280 / Math.max(img.width, img.height));
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        resolve(canvas.toDataURL('image/jpeg', 0.92));
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('decode')); };
      img.src = url;
    });
  }

  function onPaste(event) {
    const items = (event.clipboardData && event.clipboardData.items) || [];
    for (const item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        event.preventDefault();
        status.textContent = 'Đang xử lý ảnh…';
        shrink(item.getAsFile())
          .then((dataUrl) => { status.textContent = 'Đã nhận ảnh ✓'; setStateValue('image', dataUrl); })
          .catch(() => { status.textContent = 'Không đọc được ảnh trong clipboard.'; });
        return;
      }
    }
    // Không có ảnh: để trình duyệt dán chữ như bình thường
    if (box.contains(event.target) || event.target === box) {
      status.textContent = 'Clipboard không có ảnh. Hãy copy một hình ảnh trước.';
    }
  }

  box.addEventListener('click', () => box.focus());
  box.addEventListener('focus', () => box.classList.add('active'));
  box.addEventListener('blur', () => box.classList.remove('active'));
  document.addEventListener('paste', onPaste);
  return () => document.removeEventListener('paste', onPaste);
}
"""

_component = st.components.v2.component('clickbait_paste_image', html=_HTML, css=_CSS, js=_JS)


def decode_data_url(data_url):
  """data URL ảnh → PIL.Image RGB; None nếu rỗng, sai định dạng hoặc quá lớn."""
  if not data_url or not isinstance(data_url, str) or not data_url.startswith('data:image/'):
    return None
  try:
    _, payload = data_url.split(',', 1)
    raw = base64.b64decode(payload, validate=True)
  except (ValueError, binascii.Error):
    return None
  if len(raw) > MAX_BYTES:
    return None
  try:
    return Image.open(io.BytesIO(raw)).convert('RGB')
  except (UnidentifiedImageError, OSError):
    return None


def paste_image(key='paste_image'):
  """Hiện ô dán ảnh, trả về PIL.Image đã dán gần nhất hoặc None.

  Nút "Xóa ảnh đã dán" tạo instance mới của component (đổi key) để về trạng thái rỗng.
  """
  gen_key = f'{key}_generation'
  st.session_state.setdefault(gen_key, 0)
  result = _component(key=f'{key}_{st.session_state[gen_key]}', default={'image': None},
                      on_image_change=lambda: None)
  image = decode_data_url(getattr(result, 'image', None))
  if image is not None and st.button('🗑️ Xóa ảnh đã dán', key=f'{key}_clear'):
    st.session_state[gen_key] += 1
    st.rerun()
  return image
