# 🧬 Forensic Object Detection using Faster R-CNN  
### _Crime Scene Evidence Detection with Explainable AI_

> **An end-to-end deep learning system for detecting forensic evidence from crime-scene images — built with precision, interpretability, and deployment in mind.**

---

## 📁 Directory Layout

```
dataset/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

✔ Fully **COCO-compliant annotation format**

---

## ⚙️ Model Architecture & Configuration

| Component | Specification |
|---------|--------------|
| Detector | Faster R-CNN |
| Backbone | ResNet101 |
| Feature Pyramid Network | Enabled |
| Anchor Sizes | (16, 32, 64, 128, 256) |
| Aspect Ratios | (0.5, 1.0, 2.0) |
| Total Classes | 8 (7 foreground + background) |
| Optimizer | SGD |
| Learning Rate | 0.005 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| LR Scheduler | ReduceLROnPlateau |
| Epochs | 35 |
| Batch Size | 4 |
| IoU Threshold | 0.40 |
| Confidence Threshold | 0.45 |

---

## 🏋️ Training Pipeline

- Custom **CrimeSceneDataset** (PyTorch)
- Automatic loss computation handled internally by Faster R-CNN
- **Data augmentations applied**:
  - Horizontal Flip
  - Color Jitter
  - Gaussian Blur
  - Random Affine Transformations
- Best model checkpoint saved based on validation loss

---

## 💾 Trained Weights

```
best_faster_rcnn_final4.pth
```

---

## 📊 Evaluation & Metrics

- Precision  
- Recall  
- Accuracy  
- F1-Score  
- Approximate mAP  
- Per-class performance breakdown  
- Background-aware confusion matrix  

**Tools used:**
- `torchvision.ops.box_iou`
- `sklearn.metrics.confusion_matrix`

---

## 🔬 Explainable AI — Grad-CAM

```python
model.backbone.body.layer4[-1]
```

---

## 🌐 Streamlit App

```bash
streamlit run app.py
```

---

## 👤 Contributors

Dr. Ashis Kumar Pati  
Satyanistha Das  
Abhishek Sahu  

---

⭐ _Accuracy matters. Interpretability matters more._
