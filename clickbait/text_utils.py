"""Tiện ích xử lý văn bản dùng chung cho train / evaluate / app.

PhoBERT được pretrain trên văn bản ĐÃ TÁCH TỪ (word-segmented, các từ ghép nối
bằng dấu '_'). Vì vậy trước khi tokenize cần tách từ bằng pyvi để khớp phân phối
dữ liệu lúc pretrain.
"""
import re
import unicodedata

from pyvi import ViTokenizer


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản người dùng nhập vào demo.

    - Unicode NFC: chữ có dấu gõ kiểu tổ hợp ("e" + dấu sắc) và kiểu dựng sẵn ("é")
      thành cùng một dạng, khớp với tokenizer của PhoBERT.
    - Bỏ ký tự vô hình (zero-width, BOM), gộp khoảng trắng thừa.

    Dataset huấn luyện vốn đã sạch (chỉ 6/3.197 tiêu đề không ở dạng NFC), nên bước này
    chỉ áp dụng khi suy diễn để không làm thay đổi kết quả đã đánh giá.
    """
    text = unicodedata.normalize('NFC', str(text))
    text = re.sub(r'[​-‍﻿]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def segment_text(text: str) -> str:
    """Tách từ tiếng Việt: "sân bay" → "sân_bay" (định dạng PhoBERT mong đợi)."""
    return ViTokenizer.tokenize(str(text))
