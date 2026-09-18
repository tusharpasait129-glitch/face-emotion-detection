"""
Runs face detection (OpenCV Haar cascade) followed by emotion classification
(our trained EmotionCNN) on a webcam feed, a video file, or a single image.

Examples
--------
    # Live webcam (default camera index 0), press 'q' to quit
    python -m src.detect --checkpoint models/best_model.pt

    # A specific video file, saving the annotated output
    python -m src.detect --checkpoint models/best_model.pt --video path/to/clip.mp4 --save-out out.mp4

    # A single image
    python -m src.detect --checkpoint models/best_model.pt --image path/to/photo.jpg
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.model import EmotionCNN
from src.utils import EMOTIONS, EMOTION_COLORS, IMG_SIZE, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Real-time face + emotion detection")
    p.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--video", type=str, default=None, help="Path to a video file instead of webcam")
    p.add_argument("--image", type=str, default=None, help="Path to a single image instead of video/webcam")
    p.add_argument("--save-out", type=str, default=None, help="Optional path to save annotated output")
    p.add_argument("--min-face-size", type=int, default=60)
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> EmotionCNN:
    model = EmotionCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_face(gray_face: np.ndarray, device: torch.device) -> torch.Tensor:
    """Resize a grayscale face crop to 48x48 and normalise it to match training."""
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    tensor = torch.from_numpy(face).unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)
    return tensor.to(device)


def annotate_frame(frame, face_cascade, model, device, min_face_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        tensor = preprocess_face(face_roi, device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = EMOTIONS[pred_idx]
        confidence = probs[pred_idx]
        color = EMOTION_COLORS[label]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {confidence * 100:.0f}%"
        cv2.putText(frame, text, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def run_image(args, face_cascade, model, device):
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    frame = annotate_frame(frame, face_cascade, model, device, args.min_face_size)
    out_path = args.save_out or "detected_output.jpg"
    cv2.imwrite(out_path, frame)
    print(f"Saved annotated image to {out_path}")


def run_stream(args, face_cascade, model, device):
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_out, fourcc, fps, (width, height))

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = annotate_frame(frame, face_cascade, model, device, args.min_face_size)
        cv2.imshow("Face + Emotion Detection", frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")

    if args.image:
        run_image(args, face_cascade, model, device)
    else:
        run_stream(args, face_cascade, model, device)


if __name__ == "__main__":
    main()
