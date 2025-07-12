# 🔬 UNet-LaparoSeg – Endometriosis Segmentation from Laparoscopic Images

**UNet-LaparoSeg** is a deep learning model designed for localizing and segmenting endometriosis lesions in laparoscopic images. It features a custom UNet architecture enhanced with a ResNet34 encoder, ASPP, and attention gates. The system also provides Persian-language diagnostic reports and visualization overlays.

> 📌 **Live Demo (currently offline)**  
> The web demo is temporarily **disabled due to infrastructure limitations**.  
> Previous access point: [http://endovis.alihaghighat.ir/](http://endovis.alihaghighat.ir/)

---

## ⚙️ Key Features

- ✅ UNet-based segmentation with pretrained ResNet34 encoder
- 🧠 ASPP and Attention Gate modules for enhanced lesion localization
- 🧪 Color mask-to-label conversion with pixel tolerance
- 📦 Smart class-balanced sampling based on lesion size distribution
- 🧰 Albumentations-based data augmentation (elastic, dropout, flips, etc.)
- 📊 Full evaluation support: Precision, Recall, Dice, IoU, and more
- 📷 Auto-generation of overlay PNGs and transparent masks
- 📄 Persian medical-style report generation per test image
- ⚡ REST API via FastAPI for live inference and deployment

---

## 🧪 Sample Output

Input → Predicted Mask → Overlay

![Sample Output](./api/output_single/Screenshot_2025-07-12-16.58.55.jpg)

Reports are auto-generated in **Persian**, detailing lesion type and anatomical location (e.g., *Endo-Ovary*, *Endo-Uterus*, etc.).

---

## 📂 Repository Structure

📂 Repository Structure
.
├── api/                          # FastAPI-based backend
│   ├── __pycache__/              # Cached Python files
│   ├── output_single/            # Prediction outputs (masks, overlays)
│   ├── test_frames/              # Input frames for testing
│   ├── app.py                    # Main FastAPI application
│   ├── best.pth                  # Trained model weights
│   ├── model.py                  # Inference logic
│   └── *.png                     # Temporary prediction result images
│
├── Glenda_v1.5_classes/         # Dataset and training artifacts
│   ├── annots/                  # Annotation masks
│   ├── frames/                  # Raw video frames
│   ├── logs_unet/               # Training logs
│   ├── masks_numeric/           # Numerical masks for segmentation
│   ├── split_vis/               # Train/val/test split visualizations
│   └── callesification.ipynb    # Notebook for classification/analysis
    ├── coco.json                # COCO-style annotation file
    ├── config.json              # Configuration file for model or pipeline
    ├── label_colors.html        # Color legend for label visualization
    ├── labels.txt               # List of semantic class labels
    ├── loss_curve.png           # Training loss curve
    ├── model.py                 # Possibly alternate model definition
    ├── statistics_overall.csv   # Evaluation metrics and results
    ├── val_dice_curve.png       # Validation Dice score over epochs
    └── Screenshot*.png         
     # Miscellaneous screenshots


---

## 📊 Evaluation (Test Set)

| Metric     | Value   |
|------------|---------|
| Dice (macro) | 0.0666  |
| IoU (macro)  | 0.0243  |
| Recall       | 0.0385  |
| Sensitivity  | 0.3359  |
| Accuracy     | 0.7565  |

> ⚠️ These metrics reflect real-world challenges such as **class imbalance** and **irregularly shaped or small lesion areas**.

---

## 👥 Team & Credits

This project is part of the **LAP MAP** research initiative focused on automatic diagnosis of endometriosis from laparoscopic imagery.

- 🛠 Model architecture, pipeline engineering, deployment & reporting: **[Ali Haghighat](https://www.linkedin.com/in/alii-haghighat/)**
- 💡 Research scope, dataset definition, clinical insight:
  - [Niloofar Choobin](https://www.linkedin.com/in/niloofar-choobin-6129b4291/)
  - [Mobina Vatankhah](https://www.linkedin.com/in/mobina-vatankhah-554534223/)
  - [Asma Ahmadi](https://www.linkedin.com/in/asma-ahmadi-/)
  - [Reza Mirlohi](https://www.linkedin.com/in/reza-mirlohi-aa53632a8/)
---

## 📜 License & Ownership

- All Python code (modeling, inference, API, visualization): © 2025 **Ali Haghighat**
- Research design, dataset preparation, clinical scope: Intellectual Property of **LAP MAP Team**

> ❗ **Do not use** this repository for commercial or clinical deployment **without written permission** from the authors.

---

## ✨ Future Work

- ✅ Confidence-aware scoring in diagnostic reports
- 🔍 Explainability tools (e.g., Grad-CAM integration)
- ☁️ Deployment packaging (Docker, Hugging Face Spaces, Streamlit Cloud)
- 🌐 Multilingual report generation (English + Persian)

---

