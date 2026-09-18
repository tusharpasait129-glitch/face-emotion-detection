"""
Small shared helpers used by the training, evaluation and detection scripts.
Keeping these in one place avoids repeating the same constants everywhere.
"""

import random
import os
import numpy as np
import torch

# FER2013 uses these seven emotion classes, in this exact label order (0-6).
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# A colour per emotion (BGR, since we draw with OpenCV) purely to make the
# live webcam overlay easier to read at a glance.
EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 140, 255),
    "Fear": (255, 0, 255),
    "Happy": (0, 255, 0),
    "Sad": (255, 0, 0),
    "Surprise": (0, 255, 255),
    "Neutral": (200, 200, 200),
}

IMG_SIZE = 48  # FER2013 images are 48x48 grayscale


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across numpy / torch / python's random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device: CUDA GPU, Apple MPS, or CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
