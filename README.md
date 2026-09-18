# Real-Time Face & Emotion Detection

A computer vision pipeline that detects faces in a webcam feed, video, or
image, and classifies the facial expression into one of seven emotions
(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using a convolutional
neural network trained from scratch on the FER2013 dataset.

Face detection uses OpenCV's Haar cascade classifier; emotion classification
uses a custom CNN (`EmotionCNN`, in `src/model.py`) built and trained in
PyTorch — no pretrained emotion model is used.

## Project Structure

```
face-emotion-detection/
├── src/
│   ├── model.py       # EmotionCNN architecture
│   ├── dataset.py      # FER2013 CSV loader
│   ├── train.py         # Training loop (CLI)
│   ├── evaluate.py     # Test-set evaluation + confusion matrix
│   ├── detect.py         # Real-time webcam / video / image inference (CLI)
│   └── utils.py           # Shared constants and helpers
├── tests/
│   └── test_model.py    # Unit tests for the model
├── data/
│   └── README.md          # How to obtain the FER2013 dataset
├── models/                    # Trained checkpoints get saved here
├── assets/                     # Training curves / confusion matrix get saved here
├── report/                      # Project report
├── requirements.txt
└── README.md
```

## 1. Environment Setup

Requires **Python 3.9+**.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd face-emotion-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Get the Dataset

The FER2013 CSV is not bundled in this repo (see `data/README.md` for
download links and details). Download it and place it at:

```
data/fer2013.csv
```

## 3. Train the Model

```bash
python -m src.train --data data/fer2013.csv --epochs 40 --batch-size 64
```

Key options (`python -m src.train --help` for the full list):

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 40 | Max training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--lr` | 1e-3 | Learning rate (Adam) |
| `--patience` | 7 | Early-stopping patience |

This saves the best checkpoint (by validation accuracy) to
`models/best_model.pt` and a loss/accuracy plot to
`assets/training_curves.png`.

Training on a laptop CPU is slow (a GPU is recommended); reduce `--epochs`
for a quick smoke test.

## 4. Evaluate on the Test Set

```bash
python -m src.evaluate --data data/fer2013.csv --checkpoint models/best_model.pt
```

Prints a per-class precision/recall/F1 report and saves a confusion matrix
image to `assets/confusion_matrix.png`.

## 5. Run Real-Time Detection

**Webcam (default camera):**
```bash
python -m src.detect --checkpoint models/best_model.pt
```
Press `q` to quit the display window.

**Video file:**
```bash
python -m src.detect --checkpoint models/best_model.pt --video path/to/clip.mp4 --save-out out.mp4
```

**Single image:**
```bash
python -m src.detect --checkpoint models/best_model.pt --image path/to/photo.jpg
```
Saves an annotated copy to `detected_output.jpg` (or `--save-out <path>`).

> Note: `detect.py` opens an OpenCV display window (`cv2.imshow`) when run
> on webcam/video without `--save-out` on a headless machine (no display),
> pass `--save-out` so it writes the annotated video/image to disk instead
> of trying to open a window.

## 6. Run Tests

```bash
pytest tests/
```

## Method Summary

- **Face detection:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`,
  bundled with `opencv-python`) locates face bounding boxes in each frame.
- **Preprocessing:** each detected face is cropped, converted to grayscale,
  resized to 48x48, and normalised to match the training distribution.
- **Classification:** the crop is passed through `EmotionCNN`
  (3 convolutional blocks with batch norm + max pooling, followed by two
  fully connected layers with dropout) to produce a probability distribution
  over the 7 emotion classes.
- **Output:** a bounding box and the predicted label with confidence are
  drawn on the frame in real time.

See `report/Project_Report.docx` for the full methodology, dataset details,
and results.

## Known Limitations

- FER2013 labels are inherently noisy (even human agreement on the dataset
  is reported around 65-70%), which caps achievable accuracy regardless of
  model quality.
- Haar cascades are fast but less robust than deep-learning face detectors
  under extreme pose, occlusion, or low light — a DNN-based face detector
  (e.g. OpenCV's `res10` SSD) could be swapped in for improved robustness.
- The model is trained only on FER2013's demographic and lighting
  distribution and may not generalise perfectly to very different camera
  setups.

## License

MIT — see `LICENSE`.
