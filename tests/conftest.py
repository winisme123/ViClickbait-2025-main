import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
  # Đường dẫn dữ liệu trong clickbait/dataset.py là tương đối so với thư mục gốc
  monkeypatch.chdir(ROOT)
