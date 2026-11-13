import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
SAVED_MODEL_PATH = MODELS_DIR / "plant_deficiency_mobilenetv2.pth"
HISTORY_PATH = MODELS_DIR / "training_history.json"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

# Source Raw Dataset Paths
BANANA_RAW_DIR = BASE_DIR / "archive" / "Nutrient Deficient Banana Plant Leaves" / "Nutrient Deficient RAW Images of Banana Leaves"
BANANA_AUG_DIR = BASE_DIR / "archive" / "Nutrient Deficient Banana Plant Leaves" / "Augmented Images of Nutrient Deficient Banana leaves"
RICE_RAW_DIR = BASE_DIR / "archive (1)" / "rice_plant_lacks_nutrients"

# Model Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # 0 is safe on Windows
EPOCHS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
CONFIDENCE_THRESHOLD = 0.55  # Below this, report uncertainty/ask for clearer leaf image

# Standard target classes aligned with Project Proposal
TARGET_CLASSES = [
    "Calcium (Ca)",
    "Healthy",
    "Iron (Fe)",
    "Magnesium (Mg)",
    "Nitrogen (N)",
    "Phosphorus (P)",
    "Potassium (K)",
    "Sulphur (S)",
    "Zinc (Zn)"
]
