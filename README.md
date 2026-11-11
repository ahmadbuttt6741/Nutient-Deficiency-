# 🌱 AI-Based Plant Nutrition Deficiency Detection System

An end-to-end Computer Vision and Deep Learning framework for identifying plant nutrient deficiencies from leaf imagery, providing Explainable AI (Grad-CAM), confidence scoring with uncertainty rejection, and actionable agronomic advisory guidance.

Developed based on the Academic Project Proposal: *AI-Based Plant Nutrition Deficiency Detection System using Artificial Intelligence, Deep Learning and Computer Vision*.

---

## 🚀 Key Features

- **Transfer Learning Backbone:** Fine-tuned `MobileNetV2` (with support for `EfficientNet-B0` and `ResNet50`) for rapid, accurate classification on CPU and GPU.
- **Multi-Nutrient Diagnostic Classes:**
  - `Healthy Leaf`
  - `Nitrogen (N)` deficiency
  - `Phosphorus (P)` deficiency
  - `Potassium (K)` deficiency
  - `Calcium (Ca)` deficiency
  - `Iron (Fe)` deficiency
  - `Magnesium (Mg)` deficiency
  - `Sulphur (S)` deficiency
  - `Zinc (Zn)` deficiency
- **Explainable AI (Grad-CAM):** Visual activation heatmaps highlighting the exact spatial regions (margins, interveinal tissue, chlorotic zones) that drove the model's diagnosis.
- **Image Quality & Uncertainty Filtering:** Automatically detects blurry/out-of-focus leaf photos and alerts when prediction confidence is below diagnostic threshold (< 55%).
- **Public Landing Page & Authentication:** Enterprise SaaS welcome portal with two-column introduction, feature pillars, research statistics, and a secure SQLite-backed authentication system (Sign In, User Registration, and Password Reset) powered by salted PBKDF2-SHA256 password hashing.
- **Interactive Protected Workstation:** Modern, responsive SaaS-style web application accessible upon authentication, featuring leaf diagnostics, real-time Grad-CAM saliency maps, and mobile-friendly responsive breakpoints.
- **Branding & Credits:** Engineered & developed by **Nanstack** AI Solutions Lab (with direct link to [nanstack.tech](https://nanstack.tech)).
- **Bilingual Support:** Full English 🇬🇧 and Urdu 🇵🇰 agronomic advisory guidance (Section 10.7).
- **Academic Evaluation Pipeline:** Produces comprehensive confusion matrix plots, Precision, Recall, and F1-Scores on held-out test splits.

---

## 📁 Project Structure

```
Nutrient Deficiency/
├── AI_Plant_Nutrition_Proposal_Academic_Cover.pdf   # Project Proposal Specification
├── requirements.txt                                 # Core dependencies
├── README.md                                        # Documentation
├── app/
│   └── app.py                                       # Interactive Streamlit Web Application
├── src/
│   ├── __init__.py
│   ├── config.py                                    # Central hyperparameter & path configs
│   ├── dataset.py                                   # PyTorch dataset, dataloaders & augmentations
│   ├── model.py                                     # Transfer learning architecture (MobileNetV2)
│   ├── train.py                                     # Training script with AdamW & Cosine Annealing
│   ├── evaluate.py                                  # Evaluation report & confusion matrix generator
│   ├── xai.py                                       # Standalone PyTorch Grad-CAM engine
│   ├── inference.py                                 # Quality checks, prediction & advisory engine
│   └── knowledge_base.py                            # Agronomic symptoms, fertilizers & remedies
├── scripts/
│   └── prepare_dataset.py                           # Curation & standardization of Kaggle datasets
├── data/                                            # Processed train/val/test splits
└── models/                                          # Saved weights, metrics, & confusion matrix
```

---

## ⚡ Quick Start

### 1. Activate Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Prepare and Split the Dataset
```powershell
python scripts/prepare_dataset.py
```

### 3. Train the Model
```powershell
python -m src.train
```

### 4. Evaluate on Held-Out Test Set
```powershell
python -m src.evaluate
```

### 5. Launch the Web Application
```powershell
streamlit run app/app.py
```

---

## 🔐 Default Authentication Credentials

| Role | Username / Email | Password |
| :--- | :--- | :--- |
| **Botany Admin / User** | `botany` (or `botany@agrivision.ai`) | `nanstack` |
| **Chief Agronomist (Admin)** | `admin` (or `admin@agrivision.ai`) | `nanstack` |
| **AI Researcher** | `researcher` (or `researcher@nanstack.tech`) | `nanstack` |

# Nutient-Deficiency-
