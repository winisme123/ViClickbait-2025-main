import os

import pytest
import torch

from clickbait.models import MODEL_CLASSES, checkpoint_path, make_classifier, masked_mean


def test_masked_mean_ignores_padding():
  hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
  mask = torch.tensor([[1, 1, 0]])
  assert torch.allclose(masked_mean(hidden, mask), torch.tensor([[2.0, 3.0]]))


def test_masked_mean_all_padding_is_finite():
  hidden = torch.ones(1, 3, 4)
  mask = torch.zeros(1, 3, dtype=torch.long)
  assert torch.isfinite(masked_mean(hidden, mask)).all()


def test_classifier_outputs_one_logit():
  assert make_classifier(768)(torch.randn(5, 768)).shape == (5, 1)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(checkpoint_path('text', 42)),
                    reason='chưa có checkpoints/text_seed42.pth')
def test_text_checkpoint_matches_architecture():
  model = MODEL_CLASSES['text']()
  state = torch.load(checkpoint_path('text', 42), map_location='cpu', weights_only=True)
  model.load_state_dict(state)  # strict=True: lỗi nếu kiến trúc lệch với checkpoint
  model.eval()
  ids = torch.tensor([[0, 1000, 2000, 2, 1, 1]])
  mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
  with torch.no_grad():
    assert model(ids, mask).shape == (1,)
