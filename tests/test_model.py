"""
Basic unit tests for EmotionCNN. Run with:
    pytest tests/
"""

import torch
from src.model import EmotionCNN


def test_output_shape():
    model = EmotionCNN(num_classes=7)
    dummy_input = torch.randn(8, 1, 48, 48)
    output = model(dummy_input)
    assert output.shape == (8, 7)


def test_output_is_finite():
    model = EmotionCNN()
    dummy_input = torch.randn(2, 1, 48, 48)
    output = model(dummy_input)
    assert torch.isfinite(output).all()


def test_parameter_count_is_reasonable():
    model = EmotionCNN()
    n_params = sum(p.numel() for p in model.parameters())
    # Sanity bound: the model should be compact (well under 5M params)
    # given the small 48x48 input.
    assert 0 < n_params < 5_000_000


def test_different_batch_sizes():
    model = EmotionCNN()
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 1, 48, 48)
        out = model(x)
        assert out.shape == (batch_size, 7)
