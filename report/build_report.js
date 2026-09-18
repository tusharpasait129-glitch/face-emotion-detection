const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, PageBreak,
} = require("docx");

const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840; // US Letter, DXA

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, italics: opts.italics || false, bold: opts.bold || false })],
    spacing: { after: 150 },
    alignment: opts.align || AlignmentType.LEFT,
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function note(text) {
  return new Paragraph({
    children: [new TextRun({ text: "[TO FILL IN AFTER TRAINING] ", bold: true, color: "C0392B" }),
               new TextRun({ text, italics: true, color: "C0392B" })],
    spacing: { after: 150 },
  });
}

function simpleTable(headerRow, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  const mkCell = (text, bold, width) => new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: bold ? { type: ShadingType.CLEAR, fill: "2C3E50" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold, color: bold ? "FFFFFF" : "000000" })] })],
  });
  const header = new TableRow({ children: headerRow.map((t, i) => mkCell(t, true, colWidths[i])) });
  const body = rows.map(r => new TableRow({ children: r.map((t, i) => mkCell(t, false, colWidths[i])) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: colWidths, rows: [header, ...body] });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: PAGE_WIDTH, height: PAGE_HEIGHT } } },
    children: [
      new Paragraph({ text: "Real-Time Face & Emotion Detection Using a Custom CNN", heading: HeadingLevel.TITLE, spacing: { after: 200 } }),
      p("Computer Vision — Evaluated Project", { italics: true }),
      p("Course: Computer Vision", { italics: true }),
      p("Submission Date: September 18, 2026", { italics: true }),
      new Paragraph({ children: [new PageBreak()] }),

      h1("Abstract"),
      p("This project presents an end-to-end system for detecting human faces in images and video streams and classifying the expressed emotion into one of seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The pipeline combines classical computer vision (a Haar cascade face detector) with a convolutional neural network trained from scratch on the FER2013 dataset. The system runs in real time from a webcam feed and is fully operable from the command line, with separate scripts for training, evaluation, and inference. This report describes the dataset, model architecture, training procedure, evaluation methodology, and results, and discusses the limitations of the approach."),

      h1("1. Introduction"),
      p("Automatic facial expression recognition (FER) has applications in human-computer interaction, driver-monitoring systems, mental-health screening tools, and customer-experience analytics. The task is challenging because facial expressions vary continuously in intensity, are often ambiguous even to human annotators, and must be recognised reliably under varying pose, lighting, and occlusion."),
      p("The goal of this project is to build a complete, reproducible pipeline — not just an isolated classifier — that takes a raw video frame, locates faces within it, and reports the most likely emotion for each detected face in real time. The project deliberately avoids using a pretrained emotion-recognition model; the classification network is designed and trained from scratch as part of this work, so that the architectural and training choices below reflect this project's own contribution."),

      h1("2. Related Work"),
      p("Facial expression recognition has been studied extensively since the release of benchmark datasets such as CK+, JAFFE, and FER2013. Early approaches relied on handcrafted features such as Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG) combined with classical classifiers like SVMs. Since the FER2013 Kaggle challenge (2013), convolutional neural networks have become the dominant approach, with entries ranging from shallow custom CNNs to deep architectures such as VGG and ResNet variants fine-tuned for the task. This project follows the CNN-based line of work, using a compact architecture sized appropriately for FER2013's relatively small, low-resolution image set rather than a large pretrained backbone."),

      h1("3. Dataset"),
      p("FER2013 consists of 35,887 grayscale images, each 48x48 pixels, labelled with one of seven emotions. The dataset ships with a predefined split via its Usage column:"),
      simpleTable(
        ["Split", "Usage value", "Approx. size", "Purpose"],
        [
          ["Training", "Training", "28,709 images", "Model training"],
          ["Validation", "PublicTest", "3,589 images", "Hyperparameter tuning / early stopping"],
          ["Test", "PrivateTest", "3,589 images", "Final, held-out evaluation"],
        ],
        [2200, 2400, 2400, 3000]
      ),
      new Paragraph({ text: "", spacing: { after: 150 } }),
      p("The class distribution in FER2013 is known to be imbalanced (the 'Disgust' class in particular has very few examples), which is discussed further in Section 7."),

      h1("4. Methodology"),
      h2("4.1 Face Detection"),
      p("Faces are located using OpenCV's Haar Cascade classifier (haarcascade_frontalface_default.xml). This detector was chosen over a deep-learning-based face detector for its speed and zero additional dependency footprint (it ships with opencv-python), which keeps the real-time pipeline lightweight and easy to run on modest hardware. Each detected bounding box is cropped, converted to grayscale, and resized to 48x48 pixels to match the classifier's expected input."),

      h2("4.2 Preprocessing"),
      bullet("Convert frame to grayscale for face detection."),
      bullet("Crop the detected face region."),
      bullet("Resize the crop to 48x48 pixels."),
      bullet("Normalise pixel values to the [-1, 1] range (mean 0.5, std 0.5), matching the normalisation used during training."),
      bullet("During training only: apply light data augmentation — random horizontal flip, random rotation (±10°), and random crop with padding — to reduce overfitting on the relatively small training set."),

      h2("4.3 Model Architecture — EmotionCNN"),
      p("A custom convolutional network, EmotionCNN, was designed for this task (full implementation in src/model.py). It consists of three convolutional blocks followed by a fully connected classifier head:"),
      simpleTable(
        ["Block", "Layers", "Output size"],
        [
          ["Input", "—", "1 x 48 x 48"],
          ["Block 1", "Conv(32) - BN - ReLU - Conv(32) - BN - ReLU - MaxPool", "32 x 24 x 24"],
          ["Block 2", "Conv(64) - BN - ReLU - Conv(64) - BN - ReLU - MaxPool", "64 x 12 x 12"],
          ["Block 3", "Conv(128) - BN - ReLU - MaxPool", "128 x 6 x 6"],
          ["Head", "Flatten - FC(256) - ReLU - Dropout(0.4) - FC(7)", "7 (logits)"],
        ],
        [1800, 5500, 2700]
      ),
      new Paragraph({ text: "", spacing: { after: 150 } }),
      p("Design rationale: the network is intentionally shallow relative to backbones like ResNet. FER2013's small resolution (48x48) and limited training set (~28.7k images) mean a very deep or wide network overfits quickly; batch normalisation stabilises training, and dropout in the classifier head provides additional regularisation. The full parameter count is printed by running python src/model.py directly."),

      h2("4.4 Training Procedure"),
      bullet("Loss function: Cross-entropy loss over the 7 classes."),
      bullet("Optimizer: Adam, initial learning rate 1e-3, weight decay 1e-4."),
      bullet("LR schedule: ReduceLROnPlateau, halving the learning rate when validation accuracy plateaus for 3 epochs."),
      bullet("Early stopping: training halts if validation accuracy does not improve for 7 consecutive epochs (configurable via --patience)."),
      bullet("Checkpointing: the model weights are saved only when validation accuracy improves, so models/best_model.pt always reflects the best-performing epoch on the validation split."),
      bullet("Batch size: 64 (configurable)."),

      h1("5. Experimental Setup"),
      p("Training and evaluation were run using the scripts in src/ (train.py, evaluate.py). Hardware and exact hyperparameters used for the reported run:"),
      note("Fill in your actual training hardware (CPU/GPU model), total training time, number of epochs actually run before early stopping, and final hyperparameters if different from the defaults."),

      h1("6. Results"),
      h2("6.1 Training Curves"),
      note("After running `python -m src.train`, insert the generated assets/training_curves.png image here and briefly describe the trend (e.g., whether train/val loss converged, any signs of overfitting)."),
      h2("6.2 Test Set Performance"),
      p("After running python -m src.evaluate on the PrivateTest split, record the overall accuracy and per-class precision/recall/F1 below."),
      simpleTable(
        ["Emotion", "Precision", "Recall", "F1-score"],
        [
          ["Angry", "TBD", "TBD", "TBD"],
          ["Disgust", "TBD", "TBD", "TBD"],
          ["Fear", "TBD", "TBD", "TBD"],
          ["Happy", "TBD", "TBD", "TBD"],
          ["Sad", "TBD", "TBD", "TBD"],
          ["Surprise", "TBD", "TBD", "TBD"],
          ["Neutral", "TBD", "TBD", "TBD"],
          ["Overall Accuracy", "—", "—", "TBD"],
        ],
        [3200, 2600, 2600, 2600]
      ),
      new Paragraph({ text: "", spacing: { after: 150 } }),
      note("Insert the generated assets/confusion_matrix.png image here and discuss which emotion pairs are most frequently confused (commonly Fear/Sad and Angry/Disgust are hard to separate on FER2013)."),
      h2("6.3 Qualitative Real-Time Results"),
      note("Include 2-3 annotated screenshots from running `python -m src.detect --image ...` or a webcam session, showing correctly (and if relevant, incorrectly) classified expressions."),

      h1("7. Discussion & Limitations"),
      bullet("FER2013 labels are noisy: human-level agreement on this dataset is commonly cited around 65-70%, which caps the accuracy any model can realistically achieve."),
      bullet("Class imbalance: the 'Disgust' class has very few training examples relative to others, typically leading to lower recall for that class."),
      bullet("Haar cascades can miss faces under extreme pose, partial occlusion, or poor lighting; a deep-learning-based detector (e.g., an SSD/MTCNN face detector) would likely improve detection robustness at the cost of extra inference time and a heavier dependency."),
      bullet("The classifier is trained on posed/static FER2013 images; expressions in natural, in-the-wild video may differ in dynamics and intensity from the training distribution."),

      h1("8. Conclusion & Future Work"),
      p("This project implements a complete, from-scratch pipeline for real-time facial emotion recognition, covering data loading, model design, training with regularisation and early stopping, quantitative evaluation, and real-time CLI-based inference on webcam, video, and image inputs. Future improvements could include: replacing the Haar cascade with a deep-learning face detector for greater robustness, experimenting with class-weighted loss to address FER2013's class imbalance, and extending the temporal dimension by smoothing predictions across consecutive video frames to reduce flicker in the live overlay."),

      h1("References"),
      p("1. Goodfellow, I. et al. \"Challenges in Representation Learning: A Report on Three Machine Learning Contests.\" (FER2013 dataset origin), 2013."),
      p("2. Viola, P., Jones, M. \"Rapid Object Detection using a Boosted Cascade of Simple Features.\" CVPR 2001. (Haar cascade face detection)"),
      p("3. He, K. et al. \"Deep Residual Learning for Image Recognition.\" CVPR 2016. (Architectural reference for CNN design choices)"),
      p("4. PyTorch documentation: https://pytorch.org/docs/"),
      p("5. OpenCV documentation: https://docs.opencv.org/"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  require("fs").writeFileSync("Project_Report.docx", buf);
  console.log("Report written.");
});
