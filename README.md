# Real-Time Face & Emotion Detection

A computer vision pipeline that detects faces from a webcam, video, or image and classifies facial expressions into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The emotion classification model is a custom Convolutional Neural Network (EmotionCNN) built and trained from scratch using PyTorch on the FER2013 (Facial Expression Recognition 2013) dataset.

Face detection is performed using OpenCV's Haar Cascade classifier. No pretrained emotion-recognition model is used.

## Project Structure

face-emotion-detection/
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── detect.py
│   └── utils.py
├── tests/
│   └── test_model.py
├── report/
│   ├── Project_Report.docx
│   └── build_report.js
├── models/
├── assets/
├── requirements.txt
├── LICENSE
└── README.md

## 1. Environment Setup

Requires Python 3.9 or later.

Clone the repository:

git clone https://github.com/tusharpsait129-glitch/face-emotion-detection.git

cd face-emotion-detection

Create a virtual environment.

Windows:

python -m venv venv
venv\Scripts\activate

Linux/macOS:

python3 -m venv venv
source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

## 2. Get the Dataset

This project uses the FER2013 (Facial Expression Recognition 2013) dataset.

The dataset contains 35,887 grayscale 48x48 facial images labelled with seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The FER2013 CSV file is not included in this repository because of its large size.

Download the FER2013 dataset from Kaggle and place the CSV file at:

data/fer2013.csv

The CSV file should contain these columns:

emotion
pixels
Usage

The Usage column contains:

Training
PublicTest
PrivateTest

## 3. Train the Model

After placing fer2013.csv inside the data directory, run:

python -m src.train --data data/fer2013.csv --epochs 40 --batch-size 64

Important training options:

--epochs      Maximum number of training epochs
--batch-size  Mini-batch size
--lr          Learning rate
--patience    Early-stopping patience

For all available options:

python -m src.train --help

The best model checkpoint is saved as:

models/best_model.pt

Training curves are saved as:

assets/training_curves.png

Training on a CPU may be slow. A GPU is recommended for faster training.

## 4. Evaluate the Model

After training, evaluate the model using:

python -m src.evaluate --data data/fer2013.csv --checkpoint models/best_model.pt

The evaluation provides classification results including precision, recall, and F1-score.

A confusion matrix is saved as:

assets/confusion_matrix.png

## 5. Real-Time Face and Emotion Detection

### Webcam

To use the default webcam:

python -m src.detect --checkpoint models/best_model.pt

Press q to quit the display window.

### Video File

To process a video:

python -m src.detect --checkpoint models/best_model.pt --video path/to/clip.mp4 --save-out out.mp4

### Single Image

To process an image:

python -m src.detect --checkpoint models/best_model.pt --image path/to/photo.jpg

The annotated image is saved as:

detected_output.jpg

A custom output path can be specified using:

--save-out path/to/output.jpg

The detection system displays detected face bounding boxes, predicted emotions, and confidence values.

## 6. Run Tests

Run the project tests using:

pytest tests/

The tests verify important model-related functionality.

## 7. Methodology

### Face Detection

OpenCV's Haar Cascade classifier is used to detect faces in each input frame.

The classifier used is:

haarcascade_frontalface_default.xml

### Preprocessing

For each detected face:

1. The face is cropped.
2. The crop is converted to grayscale.
3. The image is resized to 48x48 pixels.
4. Pixel values are normalized.
5. The processed image is passed to the CNN.

### Emotion Classification

The processed face is passed through the custom EmotionCNN model.

The CNN uses convolutional layers, batch normalization, ReLU activation, max pooling, dropout, and fully connected layers.

The model predicts one of seven emotion classes:

Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral

### Real-Time Output

For each detected face, the system displays a bounding box, predicted emotion, and confidence score.

## 8. Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pytest

## 9. Dataset

The project uses the FER2013 (Facial Expression Recognition 2013) dataset.

FER2013 consists of grayscale 48x48 facial images classified into seven emotion categories.

The original CSV dataset is not included in this repository because of its size.

## 10. Known Limitations

FER2013 contains some noisy or ambiguous labels, which can affect classification performance.

Haar Cascade detection is fast and lightweight but may be less robust under extreme head poses, occlusion, poor lighting, or unusual camera angles.

The model is trained on FER2013 data, so performance may vary when used with different camera conditions, lighting environments, poses, or demographic distributions.

## 11. Future Improvements

Possible future improvements include:

- Replacing Haar Cascade with a deep-learning-based face detector.
- Improving the CNN architecture.
- Using additional data augmentation.
- Addressing class imbalance using weighted loss.
- Experimenting with different optimizers and learning-rate schedules.
- Improving robustness to lighting and pose variations.
- Optimizing inference speed for real-time applications.
- Deploying the system through a web or desktop interface.

## 12. Project Report

The complete project report is available in:

report/Project_Report.docx

The report contains the project methodology, implementation details, experiments, and results.

## 13. License

This project is licensed under the MIT License.

See the LICENSE file for details.

## 14. Author

Tushar Psait

GitHub:
https://github.com/tusharpsait129-glitch/face-emotion-detection
