# GermiNet

**Germination detection and classification of cotton seeds using deep learning.**

## Demo
[![GermiNet Demo](https://img.youtube.com/vi/ND4zWlc05s0/0.jpg)](https://youtu.be/ND4zWlc05s0)

## Overview
GermiNet is a pipeline for detecting and classifying germinated cotton seeds in images, built to tackle the challenges of high-instance, field-collected datasets.

Features:
- Preprocessing pipeline for seed images  
- Deep learning-based instance segmentation using Detectron2  
- Inference on single images or batches  
- Optimized COCO polygon annotations for faster training  

⚠️ **Note:** Data and pre-trained weights are not included due to size and licensing. Train your own using COCO-formatted annotations.

---

## Tech Stack
- Python, PyTorch  
- Detectron2  
- OpenCV  

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/Germinet.git
cd Germinet

# Install dependencies
pip install -r requirements.txt

# Run training (update paths to your dataset)
python train.py --coco_json_path /path/to/annotations.json --image_folder /path/to/images --output_dir ./outputs

# Run inference on a single image
python inference.py --model_dir ./outputs/run_YYYYMMDD_HHMMSS --image /path/to/test_image.jpg

# Or run inference on a folder
python inference.py --model_dir ./outputs/run_YYYYMMDD_HHMMSS --folder /path/to/test_images --output ./outputs/inference_results
