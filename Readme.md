# visionAIry

> Real-time indoor navigation assistance for visually impaired individuals using computer vision, depth estimation, and natural language audio feedback.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Android](https://img.shields.io/badge/Android-Kotlin-green?style=flat-square&logo=android)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Overview

**visionAIry** is a real-time assistive navigation system that helps visually impaired users move through indoor environments safely and independently. The system streams live video from an Android device to a Python backend, where object detection, monocular depth estimation, and LLM-driven reasoning combine to deliver continuous spatial audio feedback.

The pipeline is optimized for low-latency inference through frame skipping, image downscaling, and GPU acceleration — balancing perceptual accuracy with real-world responsiveness.

---

## Architecture

```
Android Camera
      │
      ▼ (Socket.IO / Flask)
 ┌────────────────────────────────┐
 │         Python Backend         │
 │                                │
 │  YOLO11n ──► Object Detection  │
 │  Depth Anything V2 ──► Metric  │
 │                  Depth (m)     │
 │  Groq LLM ──► Spatial Reasoning│
 │  Yapper TTS ──► Audio Output   │
 └────────────────────────────────┘
```

---

## Features

- **Object & Door Detection** — YOLO11n with custom-trained weights for indoor environments
- **Metric Depth Estimation** — Depth Anything V2 (ViT-S, Hypersim fine-tune) providing per-object distance in meters
- **LLM Spatial Reasoning** — Groq-hosted LLM generates context-aware navigation cues from structured scene data
- **Audio Feedback** — Yapper TTS delivers verbal responses in real time
- **Mobile Integration** — Android app streams frames via Socket.IO to the Flask inference server
- **Performance Optimizations** — Frame skipping, resolution downscaling, and optional GPU acceleration

---

## Getting Started

### Prerequisites

- Python 3.10+
- Anaconda / Miniconda
- Android Studio (for mobile deployment)
- CUDA-compatible GPU (recommended) or Apple Silicon (MPS fallback supported)

### 1. Environment Setup

```bash
cd depth_anything/Depth-Anything-V2
conda create -n visionairy python=3.10
conda activate visionairy
pip install -r requirements.txt
```

> **macOS (Apple Silicon):** Set the MPS fallback before running any scripts:
> ```bash
> export PYTORCH_ENABLE_MPS_FALLBACK=1
> ```

> **Windows (CUDA):** Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the appropriate `torch-cuda` build if not already present.

### 2. Download the Depth Model

Download the **Depth-Anything-V2-Small (Hypersim)** metric depth checkpoint:

- [Google Drive](https://drive.google.com/file/d/1sH69FufnBDQmAJkZIf4lwVhBFr2bOBjE/view?usp=drive_link)
- [Original source](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)

Place the downloaded file in:

```
depth_anything/Depth-Anything-V2/checkpoints/
```

### 3. Configure API Key

Add your Groq API key to both:
- `run_webcam_metric_combined_flask.py`
- `run_webcam_metric_combined.py`

---

## Running the System

### Webcam (local testing)

```bash
cd depth_anything/Depth-Anything-V2
python run_webcam_metric_combined_flask.py --inputsource webcam --encoder vits
```

### Video file

```bash
# Download sample videos from:
# https://drive.google.com/drive/u/0/folders/1Szij42KgBW6JFPqv2YAGnSqQbktg6dK2
# Place the `common/` folder at the repo root, then:

python run_webcam_metric_combined.py --encoder vits
```

### Android + Flask (full pipeline)

**Step 1 — Start the Flask server:**

```bash
cd depth_anything/Depth-Anything-V2
python run_webcam_metric_combined_flask.py --inputsource phone --encoder vits
```

The terminal will display the server's local network address, e.g.:

```
[+] Flask server is starting...
[+] Server running on:
    - Local:   http://127.0.0.1:5001
    - Network: http://192.168.x.x:5001
```

**Step 2 — Configure the Android app:**

Open `AndroidApp/` in Android Studio and update two files with the network IP from Step 1:

**`MainActivity.kt`**
```kotlin
socket = IO.socket("http://192.168.x.x:5001", opts)
```
Path: `app/src/main/java/edu/northeastern/visionairy/MainActivity.kt`

**`network_security_config.xml`**
```xml
<domain includeSubdomains="true">192.168.x.x</domain>
```
Path: `app/src/main/res/xml/network_security_config.xml`

**Step 3 — Build and deploy** the Android app from Android Studio. The app will automatically connect to the Flask server on launch.

---

## Dependencies

```
matplotlib
opencv-python
torch
torchvision
flask
flask-cors
flask-socketio
eventlet
ultralytics
numpy
Pillow
scipy
scikit-image
tqdm
groq
yapper-tts
```

---

## File Reference

| File | Description |
|---|---|
| `run_webcam_metric_combined_flask.py` | Primary entry point — Flask server with Socket.IO, YOLO detection, depth estimation, LLM reasoning, and TTS for phone input |
| `run_webcam_metric_combined.py` | Standalone pipeline for webcam or video file input |
| `yoloTransferLearning.ipynb` | Transfer learning notebook for fine-tuning YOLO on custom indoor datasets |
| `LLMassistant.py` | Groq LLM interface for scene-to-instruction generation |
| `assistant.py` | TTS and audio response pipeline |

---

## Repository Structure

```
visionAIry/
├── Androidapp/                          # Kotlin Android application
├── common/                              # Sample videos and ground-truth images
├── DEPTH/                               # Depth exploration notebooks
├── depth_anything/
│   └── Depth-Anything-V2/              # Core inference pipeline
│       ├── depth_anything_v2/
│       ├── metric_depth/
│       ├── run_webcam_metric_combined_flask.py
│       ├── run_webcam_metric_combined.py
│       ├── LLMassistant.py
│       ├── assistant.py
│       └── requirements.txt
├── YOLO/
│   ├── yolo11n.pt
│   └── yoloTransferLearning.ipynb
├── yolo11n.pt
└── yolo_custom_weights.pt
```

---

## Roadmap

- [ ] **Directional guidance** — compass-aware turn-by-turn instructions ("turn left in 2 meters")
- [ ] **Outdoor navigation** — GPS integration with LiDAR-calibrated depth for open environments
- [ ] **Edge deployment** — on-device inference using ONNX / TFLite to eliminate server dependency
- [ ] **Semantic mapping** — persistent scene memory across frames for more coherent spatial descriptions
- [ ] **Multi-language TTS** — internationalization of audio feedback
- [ ] **Obstacle urgency ranking** — priority scoring to alert users only to the most immediate hazards
- [ ] **Wearable integration** — support for smart glasses or chest-mounted camera rigs

---

## License

This project is licensed under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2024 Karthik Koduru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgements

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — monocular metric depth estimation
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — real-time object detection
- [Groq](https://groq.com) — low-latency LLM inference
- [Yapper TTS](https://pypi.org/project/yapper-tts/) — text-to-speech audio output