# Deepfake Video Detection using CNN–LSTM

This project implements a deep learning–based approach for detecting deepfake videos by modeling both spatial facial features and temporal inconsistencies across video frames. A pretrained ResNeXt-50 CNN is used for feature extraction, followed by an LSTM network for sequence modeling.

---

## Project Overview

Deepfake videos often contain subtle spatial artifacts and temporal inconsistencies that are difficult to detect using frame-level analysis alone. This project addresses the problem by combining convolutional and recurrent neural networks to perform end-to-end video classification into REAL and FAKE categories.

---

## Model Architecture

- Backbone CNN: ResNeXt-50 (32×4d), pretrained on ImageNet
- Feature Extraction: Frame-level spatial features
- Temporal Modeling: LSTM over frame sequences
- Pooling: Adaptive Average Pooling
- Regularization: Dropout (0.4)
- Loss Function: Cross-Entropy Loss
- Optimizer: Adam

### Architecture Flow

Video → Frames → ResNeXt-50 → Feature Maps → Adaptive Avg Pool → LSTM → Fully Connected Layer → REAL / FAKE

---

## Implementation Details

- Each video is sampled into a fixed-length frame sequence
- CNN extracts discriminative spatial embeddings from each frame
- LSTM captures temporal dependencies across frames
- Final prediction is obtained by averaging LSTM outputs over time
