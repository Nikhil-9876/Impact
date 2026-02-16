<p align="center">
  <img src="https://img.shields.io/badge/AI-Audio%20Detector-blueviolet?style=for-the-badge&logo=soundcloud&logoColor=white" alt="AI Audio Detector"/>
  <img src="https://img.shields.io/badge/WavLM-Ensemble-orange?style=for-the-badge&logo=pytorch&logoColor=white" alt="WavLM"/>
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Deployable-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

# 🛡️ Impact — AI Voice Deepfake Detector

> **Detect AI-generated speech from real human voices across multiple Indian languages using a WavLM-based ensemble deep learning model.**

Impact is an end-to-end system that classifies audio as **AI-generated** or **genuine human speech** with high confidence. Built for multilingual Indian language support, it combines Microsoft's WavLM foundation model with dual classification heads (AASIST + OC-Softmax) in a production-ready FastAPI service.

---

## 📐 Architecture Diagram

<!-- Eraser Diagram Embed — Replace the src URL below with your Eraser diagram link -->
<!-- To create your diagram: https://app.eraser.io → Create a new diagram → Export as image or use embed link -->

![Architecture Diagram](https://app.eraser.io/workspace/YOUR_WORKSPACE_ID)

<!-- If using Eraser embed (interactive): -->
<!-- <a href="https://app.eraser.io/workspace/YOUR_WORKSPACE_ID" target="_blank"><img src="https://app.eraser.io/workspace/YOUR_WORKSPACE_ID/preview" alt="Architecture Diagram" /></a> -->

<!-- Alternative: If you export the diagram as a PNG and add it to the repo: -->
<!-- ![Architecture Diagram](./assets/architecture-diagram.png) -->

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **WavLM Backbone** | Leverages Microsoft WavLM-base as a frozen feature extractor for robust audio representations |
| 🎯 **Dual-Head Ensemble** | AASIST (attention-based) + OC-Softmax heads with weighted ensemble (60/40) |
| 🌐 **Multilingual** | Supports **Tamil, English, Hindi, Malayalam, Telugu** |
| 🔊 **Sliding Window** | Processes audio up to 60s using overlapping 5-second windows (50% overlap) |
| 🛡️ **Audio Validation** | Rejects silent, clipped, or non-speech audio before inference |
| 🔐 **API Key Auth** | Secure access via `x-api-key` header |
| 🐳 **Docker Ready** | Single-command deployment with Docker |
| 📊 **Quality Checks** | RMS energy, zero-crossing rate, spectral centroid, and clipping validation |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                           │
│            (Audio File / Base64 MP3 + Language + API Key)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  /api/detect  │  │  /api/voice  │  │  /api/encode-base64   │  │
│  │  -from-file   │  │  -detection  │  │                       │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘  │
│         │                 │                                      │
│         ▼                 ▼                                      │
│  ┌─────────────────────────────────┐                             │
│  │     Audio Preprocessing         │                             │
│  │  • Load & Resample (16kHz)      │                             │
│  │  • Quality Validation           │                             │
│  │  • Bandpass Filter (80-7800 Hz) │                             │
│  │  • Peak Normalization           │                             │
│  │  • Sliding Window (5s, 50% hop) │                             │
│  └──────────────┬──────────────────┘                             │
│                 ▼                                                │
│  ┌─────────────────────────────────┐                             │
│  │   WavLM Feature Extraction      │                             │
│  │   (microsoft/wavlm-base)        │                             │
│  │   768-dim hidden states          │                             │
│  └──────────────┬──────────────────┘                             │
│                 ▼                                                │
│  ┌────────────────────┬────────────────────┐                     │
│  │   AASIST Head       │   OC-Softmax Head  │                    │
│  │   (Attention + MLP) │   (LayerNorm + MLP) │                   │
│  │   Weight: 0.6       │   Weight: 0.4       │                   │
│  └─────────┬──────────┘└─────────┬──────────┘                    │
│            └──────────┬──────────┘                               │
│                       ▼                                          │
│            ┌─────────────────────┐                               │
│            │  Ensemble Average   │                               │
│            │  across all windows │                               │
│            └─────────┬───────────┘                               │
│                      ▼                                           │
│            ┌─────────────────────┐                               │
│            │   Classification    │                               │
│            │  AI_GENERATED or    │                               │
│            │  HUMAN + Confidence │                               │
│            └─────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (for MP3 support)
- CUDA GPU (recommended, CPU supported)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Impact.git
cd Impact
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export API_KEY="your-secret-api-key"
```

### 3. Add Model Weights

Place your trained model files in the project root:
- `best_model.pt` — Trained WavLM + AASIST + OC-Softmax checkpoint
- `optimal_threshold.txt` — Detection threshold (optional, defaults to 0.5)

### 4. Run the API

```bash
python api.py
```

The API will be live at **http://localhost:8000** — interactive docs at **http://localhost:8000/docs**

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t impact-ai-detector .

# Run
docker run -p 7860:7860 -e API_KEY="your-secret-api-key" impact-ai-detector
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info & available endpoints |
| `GET` | `/health` | Health check & model status |
| `POST` | `/api/detect-from-file` | **Upload audio file directly** (recommended) |
| `POST` | `/api/voice-detection` | Detect from base64-encoded MP3 |
| `POST` | `/api/encode-to-base64` | Convert audio file to base64 |

### Example: File Upload (Recommended)

```bash
curl -X POST "http://localhost:8000/api/detect-from-file" \
  -H "x-api-key: your-secret-api-key" \
  -F "file=@sample_audio.mp3" \
  -F "language=English"
```

### Example Response

```json
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92
}
```

### Example: Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/api/detect-from-file",
    headers={"x-api-key": "your-secret-api-key"},
    files={"file": open("audio.mp3", "rb")},
    data={"language": "Tamil"}
)
print(response.json())
```

---

## 🧪 Model Training

The training pipeline is in [`Modelf.ipynb`](Modelf.ipynb) and runs on Kaggle with GPU acceleration.

### Training Data

| Class | Languages | Source |
|-------|-----------|--------|
| **Human** | English, Hindi, Tamil, Telugu, Malayalam | AI4Bharat dataset |
| **AI** | English, Hindi, Tamil, Telugu, Malayalam | AI-generated speech samples |

### Training Pipeline

1. **Data Loading** — Multi-path loader supporting flat/nested folder structures
2. **Augmentation** — Speed perturbation, gain variation, noise injection, codec simulation, random EQ, clipping
3. **Feature Extraction** — WavLM-base with top-2 layer fine-tuning
4. **Classification** — Dual-head training (AASIST + OC-Softmax) with label smoothing
5. **Optimization** — AdamW optimizer, gradient clipping, early stopping on AUC
6. **Threshold Tuning** — ROC-based optimal threshold selection

### Key Training Configs

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16,000 Hz |
| Window Duration | 5.0 seconds |
| Batch Size | 32 |
| Learning Rate | 2e-4 |
| Epochs | 10 |
| Dropout | 0.3 |
| Label Smoothing | 0.05 |
| WavLM Unfrozen Layers | Top 2 |

---

## 📁 Project Structure

```
Impact/
├── api.py                 # FastAPI production server (inference API)
├── Modelf.ipynb           # Training notebook (Kaggle)
├── keepalive_app.py       # HuggingFace Space keep-alive service
├── test_file_upload.py    # API testing script
├── Dockerfile             # Docker container config
├── requirements.txt       # Python dependencies
├── best_model.pt          # Trained model weights (not in repo)
├── optimal_threshold.txt  # Detection threshold (not in repo)
├── README.md              # This file
├── README_API.md          # Detailed API documentation
└── README_KEEPALIVE.md    # Keep-alive service docs
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | PyTorch, Transformers (HuggingFace) |
| **Audio Processing** | torchaudio, librosa, soundfile |
| **Foundation Model** | Microsoft WavLM-base |
| **API Framework** | FastAPI + Uvicorn |
| **Containerization** | Docker |
| **Deployment** | HuggingFace Spaces / Any cloud |
| **Training Platform** | Kaggle (GPU P100/T4) |

---

## 🌐 Deployment on HuggingFace Spaces

The project includes a **keep-alive service** ([`keepalive_app.py`](keepalive_app.py)) that pings your HuggingFace Space every 24 hours to prevent sleep mode.

```bash
# Set your HF Space URL
export HF_SPACE_URL="https://your-username-your-space.hf.space"

# Run the keep-alive service
python keepalive_app.py
```

---

## 🤝 Team

Built for hackathon submission by **Team Impact**.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ for safer AI — detecting deepfakes, one audio at a time.</b>
</p>