# Deepfake Video Detection using CNN–LSTM<img width="1112" height="588" alt="Screenshot 2026-01-08 at 12 56 38 AM" src="https://github.com/user-attachments/assets/3a29220f-8965-4123-aa04-bca17010bd9f" />

- Dataset: https://www.kaggle.com/datasets/pranabkc/deepfake-with-cropped-faces-from-video
- Frontend-Repo: https://github.com/modakverma-dev/deepfake-frontend
- Backend-Repo: https://github.com/modakverma-dev/deepfake-backend
- Huggingface-Model: https://huggingface.co/maddy08/deepfake-video-detection


This project implements a deep learning–based approach for detecting deepfake videos by modeling both spatial facial features and temporal inconsistencies across video frames. A pretrained ResNeXt-50 CNN is used for feature extraction, followed by an LSTM network for sequence modeling.

---

## Project Overview

Deepfake videos often contain subtle spatial artifacts and temporal inconsistencies that are difficult to detect using frame-level analysis alone. This project addresses the problem by combining convolutional and recurrent neural networks to perform end-to-end video classification into REAL and FAKE categories.
<img width="938" height="586" alt="Screenshot 2026-01-08 at 2 46 14 AM" src="https://github.com/user-attachments/assets/b89493f4-7694-4d39-8caa-c9ac3bcc7ce0" />

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
