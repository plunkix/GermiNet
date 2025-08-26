# GermiNet

**Germination detection and classification of cotton seeds using deep learning.**

## Demo
[![GermiNet Demo](https://img.youtube.com/vi/ND4zWlc05s0/0.jpg)](https://youtu.be/ND4zWlc05s0)

## Overview
GermiNet is a pipeline for detecting and classifying germinated cotton seeds in images, built to tackle the challenges of high-instance, field-collected datasets with high accuracy.

Features:
- Preprocessing pipeline for seed images  
- Deep learning-based instance segmentation using Detectron2  
- Model evaluation and visualization of seed counts  
- Full-stack deployment with containerized backend + frontend  
- Optimized COCO polygon annotations for faster training  

⚠️ **Note:** Data and pre-trained weights are not included due to size and licensing.

---

## Tech Stack
- **Python, PyTorch** – core ML framework  
- **Detectron2** – instance segmentation backbone  
- **OpenCV** – preprocessing and visualization  
- **FastAPI** – backend for inference services  
- **JavaScript (React/Node)** – frontend + server logic  
- **Docker** – containerized deployment  
  - Base container with Detectron2 + dependencies  
  - Full-stack container for GermiNet (backend + frontend)  

---

## Contact
For questions or collaborations, reach out via GitHub.


