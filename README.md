# 🔍 Forensic Object Detection using Faster R-CNN (ResNet101 + FPN)

This project implements an **end-to-end forensic object detection system** using **Faster R-CNN with a ResNet101-FPN backbone**.  
It covers **training, evaluation, visualization, explainability (Grad-CAM)**, and an **interactive Streamlit-based inspection app**.

The system is designed to detect **crime-scene evidence objects** from images with high precision and interpretability.

---

## 📌 Project Overview

- **Task**: Object Detection (Forensic / Crime Scene Evidence)
- **Model**: Faster R-CNN
- **Backbone**: ResNet101 + Feature Pyramid Network (FPN)
- **Framework**: PyTorch + Torchvision
- **Deployment**: Streamlit Web App
- **Explainability**: Grad-CAM

---

## 🧠 Detected Classes (Foreground)

The model is trained on **7 forensic object classes**:

1. Blood  
2. Finger-print  
3. Hammer  
4. Handgun  
5. Human-body  
6. Knife  
7. Shotgun  

> Background is automatically handled as **class 0** by Faster R-CNN.

---

## 🗂 Dataset

### 📥 Download Dataset (Kaggle)

The dataset is hosted on Kaggle and can be downloaded programmatically using `kagglehub`.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("satyanisthadas/resnet101")

print("Path to dataset files:", path)
```

### 📁 Dataset Path (Used in Code)

After downloading, the dataset is accessed using:

```text
/kaggle/input/resnet101/dataset/
```

### 📂 Directory Structure

```text
dataset/
├── train/
│   ├── images (.jpg / .png)
│   └── _annotations.coco.json
├── valid/
│   ├── images
│   └── _annotations.coco.json
└── test/
    ├── images
    └── _annotations.coco.json
```

---

## ⚙️ Model Configuration (From Code)

| Parameter | Value |
|--------|------|
| Architecture | Faster R-CNN |
| Backbone | ResNet101 |
| FPN | Enabled |
| Anchor Sizes | (16, 32, 64, 128, 256) |
| Aspect Ratios | (0.5, 1.0, 2.0) |
| Foreground Classes | 7 |
| Total Classes | 8 (including background) |
| Optimizer | SGD |
| Learning Rate | 0.005 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| Scheduler | ReduceLROnPlateau |
| Epochs | 35 |
| Batch Size | 4 |
| IoU Threshold | 0.4 |
| Confidence Threshold | 0.45 |

---

## 🏋️ Training Pipeline

- COCO-format annotations
- Custom `CrimeSceneDataset` class
- Data augmentation:
  - Horizontal Flip
  - Color Jitter
  - Gaussian Blur
  - Random Affine
- Loss computed internally by Faster R-CNN
- Best model checkpoint saved automatically

### 📦 Saved Model
```text
best_faster_rcnn_final4.pth
```

---

## 📊 Evaluation Metrics

The project includes:
- Precision
- Recall
- Accuracy
- F1-score
- Approximate mAP
- Per-class metrics
- Confusion Matrix (background-aware)

Implemented using:
- `torchvision.ops.box_iou`
- `sklearn.metrics.confusion_matrix`

---

## 🔍 Visualization

- Bounding box visualization (OpenCV + Matplotlib)
- Side-by-side original vs predicted images
- Random test image sampling
- Class-wise visualization

---

## 🧪 Explainability (Grad-CAM)

Grad-CAM is applied on:
```python
model.backbone.body.layer4[-1]
```

This helps visualize **which regions of the image influence the detection decision**.

---

## 🌐 Streamlit Web App

### Features:
- Upload image for detection
- Class-wise Non-Maximum Suppression
- Adjustable IoU & confidence thresholds
- Colored bounding boxes per class
- Cropped detection previews
- CPU / CUDA selection
- Checkpoint metadata display

Run locally:
```bash
streamlit run app.py
```

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

> Model architecture details are intentionally **not** included in `requirements.txt` as per best practices.

---

## 📁 Important Files

| File | Description |
|----|-----------|
| `CrimeSceneDataset` | Custom PyTorch dataset |
| `train_one_epoch()` | Training loop |
| `validate()` | Validation loss |
| `per_class_metrics()` | Class-wise evaluation |
| `compute_confusion_matrix()` | Confusion matrix |
| `Grad-CAM scripts` | Model explainability |
| `app.py` | Streamlit deployment |
| `labels.txt` | Class labels |
| `best_faster_rcnn_final4.pth` | Trained weights |

---

## ✅ Key Highlights

- Full ML lifecycle: training → evaluation → deployment
- COCO-compliant dataset handling
- Explainable AI using Grad-CAM
- Production-style Streamlit interface
- Clean, modular, reproducible code

---

## 👤 Author 

**Dr. Ashis Kumar Pati**

Dept. of Data Science - ITER - SOA University

**Satyanistha Das**  
B.Tech (Final Year) – Data Science  

**Abhishek Sahu**  
B.Tech (Final Year) – Data Science

---

⭐ If you find this project useful, consider starring the repository!
