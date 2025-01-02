# Cough-COVID19
CoughCOVID: AI-Assisted Prediction of COVID-19 Through Cough

## Overview
CoughCOVID is a deep learning-based project designed to predict COVID-19 infection based on cough sounds. By leveraging signal processing technology and deep learning models, the project explores the potential of audio-based health diagnostics.

## Key Features

- **Objective**: Predict COVID-19 infection from cough sound recordings.
- **Audio Processing**: Converts recorded WAV audio files into spectrogram representations, specifically Short-Time Fourier Transform (STFT) or Mel spectrograms, for feature extraction.
- **Deep Learning Model**: Utilizes Convolutional Neural Networks (CNNs) to analyze spectrogram features and classify the likelihood of COVID-19 infection.
- **Hardware Integration**: Incorporates a Raspberry Pi device for real-time cough sound input, preprocessing, and prediction.

## Workflow

### Data Collection
- Record cough sounds in WAV format.
- Label the collected audio files for training and testing.

### Preprocessing
- Convert WAV files to STFT or Mel spectrogram images.
- Apply normalization and augmentation techniques to enhance model robustness.

[]

### Model Training
- Train a CNN model on spectrogram data to classify cough sounds.
- Optimize hyperparameters and evaluate performance using appropriate metrics.

### Real-Time Inference
- Deploy the trained model on a Raspberry Pi for real-time cough detection and analysis.

## Application Areas
This project aims to contribute to the development of accessible and scalable health screening tools. Potential applications include:

- COVID-19 screening in public or healthcare settings.
- Integration with IoT devices for continuous health monitoring.
- Early detection and intervention during pandemics.
