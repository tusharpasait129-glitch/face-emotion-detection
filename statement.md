# Project Statement

## Problem Statement

Facial expressions provide important information about a person's emotional state. However, manually identifying emotions from images or video is difficult to perform consistently and in real time.

This project develops a real-time computer vision system that detects human faces and classifies facial expressions into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Project Scope

The project covers face detection and facial emotion classification from webcam streams, video files, and individual images.

The system uses OpenCV Haar Cascade for face detection and a custom Convolutional Neural Network (EmotionCNN) implemented in PyTorch for emotion classification. The model is trained from scratch using the FER2013 dataset.

## Target Users

The project can be useful for:

- Students learning computer vision and deep learning.
- Researchers experimenting with facial emotion recognition.
- Developers building emotion-aware computer vision applications.
- Educational demonstrations of real-time image classification.

## High-Level Features

1. Real-time face detection using a webcam.
2. Facial emotion classification into seven emotion categories.
3. Processing of video files.
4. Processing of individual images.
5. Confidence scores for predicted emotions.
6. Model training and evaluation using FER2013.
7. Classification metrics and confusion matrix generation.
8. Automated model testing using Pytest.

## Functional Requirements

1. The system shall detect faces from input images, videos, and webcam streams.
2. The system shall classify detected faces into seven emotion categories.
3. The system shall display the predicted emotion and confidence score.
4. The system shall support model evaluation using the FER2013 test data.

## Non-Functional Requirements

1. The system should provide near real-time inference for webcam input.
2. The system should be reproducible using the provided requirements file.
3. The system should maintain modular and maintainable source code.
4. The system should provide reliable evaluation results using standard classification metrics.
