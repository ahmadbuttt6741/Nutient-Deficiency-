"""
Inference Pipeline for Plant Nutrient Deficiency Detection & Out-of-Distribution Rejection.
Performs:
1. Strict file format & byte signature verification (rejects non-images, corrupt files).
2. Botanical foliar domain validation (rejects text documents, monochrome scans, non-foliar objects).
3. 10-Class Deep Neural Network inference with an explicit 'Irrelevant_Non_Plant' rejection class.
4. Explainable AI (Grad-CAM) and agronomic prescription for valid plant leaves only.
"""

import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import (
    SAVED_MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD
)
from src.model import load_checkpoint
from src.dataset import get_transforms
from src.xai import GradCAM
from src.knowledge_base import NUTRIENT_KNOWLEDGE_BASE, DISCLAIMER_TEXT, DISCLAIMER_TEXT_UR

def validate_file_signature(file_input: Union[str, Path, io.BytesIO, Any]) -> Tuple[bool, str]:
    """
    Validates file headers / magic bytes and image integrity.
    Allowed formats: JPEG, PNG, WEBP.
    Rejects PDF, text documents, binaries, corrupted streams, or empty files.
    """
    try:
        # If a PIL image object is directly provided, it is already valid
        if isinstance(file_input, Image.Image):
            return True, "Valid PIL Image"

        header = b""
        if isinstance(file_input, (str, Path)):
            p = Path(file_input)
            if not p.exists() or p.stat().st_size == 0:
                return False, "File is empty or does not exist."
            with open(p, "rb") as f:
                header = f.read(16)
        elif hasattr(file_input, "read"):
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            header = file_input.read(16)
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            if len(header) == 0:
                return False, "Uploaded file is empty (0 bytes)."
        else:
            return True, "Valid input stream"

        # Check Magic Byte Signatures
        is_jpeg = header.startswith(b"\xFF\xD8\xFF")
        is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
        is_webp = header.startswith(b"RIFF") and b"WEBP" in header[:12]

        if not (is_jpeg or is_png or is_webp):
            return False, "Unsupported file format. AgriVision AI only accepts authentic JPEG, PNG, or WEBP images."

        # Verify PIL decoding & resolution
        if isinstance(file_input, (str, Path)):
            with Image.open(file_input) as img:
                img.verify()
            with Image.open(file_input) as img:
                w, h = img.size
                if w < 32 or h < 32:
                    return False, f"Image resolution ({w}x{h}) is too small for foliar analysis (minimum 32x32 required)."
        elif hasattr(file_input, "seek"):
            pos = file_input.tell()
            with Image.open(file_input) as img:
                img.verify()
            file_input.seek(pos)
            with Image.open(file_input) as img:
                w, h = img.size
                if w < 32 or h < 32:
                    return False, f"Image resolution ({w}x{h}) is too small for foliar analysis (minimum 32x32 required)."
            file_input.seek(pos)

        return True, "Valid image format verified."
    except Exception as e:
        return False, f"Corrupted or unreadable image stream ({str(e)})."

def validate_botanical_domain(image_pil: Image.Image) -> Tuple[bool, float, str]:
    """
    Analyzes chromatic distribution in HSV space to verify foliar / biological vegetation presence.
    Detects and rejects:
    - Pure monochrome / text documents / code screenshots (saturation < 0.08 across > 85% pixels).
    - Synthetic single-color graphics or unnatural extremes.
    - Non-foliar scenes with zero botanical hue representation.
    """
    thumb = image_pil.convert("RGB").resize((100, 100))
    arr = np.array(thumb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Calculate Saturation & Brightness
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_c > 0, (max_c - min_c) / np.maximum(max_c, 1e-5), 0)
    brightness = max_c

    # 1. Document / Monochrome Detection (black text on white or gray)
    low_sat_ratio = np.mean(saturation < 0.08)
    if low_sat_ratio > 0.88:
        return False, 0.0, "Monochrome document, text scan, or black-and-white graphic detected"

    # 2. Foliar Pigment Masks (Chlorophyll Green, Chlorotic Yellow, Necrotic Brown, Phosphorus Purpling)
    foliar_mask = (
        ((g > b * 1.05) & (g > r * 0.65)) |                          # Green foliage
        ((r > b * 1.15) & (g > b * 1.08) & (r > 0.18) & (g > 0.18)) | # Chlorotic yellow
        ((r > g) & (g > b) & ((r - b) > 0.12) & (b < 0.65)) |         # Necrotic brown/bronze
        ((r > g * 1.08) & (b > g * 1.05) & (r > 0.20))                 # Anthocyanin purple
    )

    # Foliar pixels must have organic saturation and non-extreme brightness
    valid_foliar = foliar_mask & (saturation > 0.10) & (brightness > 0.08) & (brightness < 0.98)
    foliar_coverage_pct = float(np.mean(valid_foliar) * 100.0)

    # At least 6% of the frame must contain foliar tissue characteristics
    if foliar_coverage_pct < 6.0:
        return False, foliar_coverage_pct, f"Image lacks foliar biological tissue (foliar coverage: {foliar_coverage_pct:.1f}%)"

    return True, foliar_coverage_pct, "Botanical foliar characteristics confirmed"

def check_image_sharpness(img_pil: Image.Image) -> float:
    """Computes image Laplacian variance to assess sharpness/blurriness."""
    gray = np.array(img_pil.convert("L"), dtype=np.float32)
    laplacian = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(laplacian.var())

class NutrientDeficiencyPredictor:
    """
    End-to-end predictor for leaf nutrient deficiency diagnosis.
    Includes multi-tier Out-of-Distribution (OOD) rejection and format validation.
    """
    def __init__(self, model_path: Optional[Path] = None, device: Optional[torch.device] = None):
        self.model_path = model_path if model_path is not None else SAVED_MODEL_PATH
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.architecture = "mobilenet_v2"
        self.val_acc = 0.0
        self.grad_cam = None
        self.transform = get_transforms()["inference"]
        self.temperature = 1.0989  # Empirically fitted calibration temperature

        if self.model_path.exists():
            self.load()

    def load(self):
        """Loads model weights and initializes Grad-CAM."""
        self.model, self.class_names, self.architecture, self.val_acc = load_checkpoint(
            self.model_path,
            device=self.device
        )
        self.grad_cam = GradCAM(self.model, architecture=self.architecture)
        print(f"Model loaded successfully ({self.architecture}, Classes: {len(self.class_names)}, Val Acc: {self.val_acc:.2%})")

    def is_ready(self) -> bool:
        """Returns True if model weights are loaded and ready for inference."""
        return self.model is not None

    def predict(
        self,
        image_input: Union[Image.Image, str, Path, Any],
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        """
        Runs comprehensive diagnosis on a leaf image.
        If irrelevant data or invalid file format is detected:
        Strictly suppresses predictions and predicts nothing.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Train a model or provide a valid checkpoint.")

        # =====================================================
        # TIER 1: File Format & Byte Signature Verification
        # =====================================================
        is_valid_format, format_err = validate_file_signature(image_input)
        if not is_valid_format:
            return {
                "status": "rejected",
                "rejection_type": "invalid_format",
                "rejection_message": format_err,
                "rejection_message_ur": "فائل کا فارمیٹ درست نہیں ہے۔ صرف JPEG، PNG یا WEBP تصاویر قابلِ قبول ہیں۔",
                "is_valid_format": False,
                "is_botanical": False,
                "is_irrelevant": True,
                "predicted_class": None,
                "confidence": 0.0,
                "top_3": [],
                "cam_overlay": None,
                "guidance": None
            }

        # Load PIL Image
        try:
            if isinstance(image_input, (str, Path)):
                image_pil = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image_pil = image_input.convert("RGB")
            elif hasattr(image_input, "seek"):
                image_input.seek(0)
                image_pil = Image.open(image_input).convert("RGB")
            else:
                image_pil = Image.open(image_input).convert("RGB")
        except Exception as e:
            return {
                "status": "rejected",
                "rejection_type": "invalid_format",
                "rejection_message": f"Unable to decode image bytes: {str(e)}",
                "rejection_message_ur": "تصویر کو پڑھنے میں خرابی پیش آئی ہے۔ براہ کرم درست تصویر منتخب کریں۔",
                "is_valid_format": False,
                "is_botanical": False,
                "is_irrelevant": True,
                "predicted_class": None,
                "confidence": 0.0,
                "top_3": [],
                "cam_overlay": None,
                "guidance": None
            }

        # =====================================================
        # TIER 2: Botanical Domain & Foliar Tissue Verification
        # =====================================================
        is_botanical, foliar_pct, botanical_reason = validate_botanical_domain(image_pil)
        if not is_botanical:
            return {
                "status": "rejected",
                "rejection_type": "irrelevant_data",
                "rejection_message": (
                    f"Irrelevant Data Detected: {botanical_reason}. "
                    "The uploaded image does not contain recognizable plant leaf tissue or agricultural foliage. "
                    "AgriVision AI specializes strictly in foliar nutrient deficiency analysis. "
                    "Predictions are suppressed to prevent misleading diagnosis."
                ),
                "rejection_message_ur": (
                    "غیر متعلقہ یا پودوں سے ہٹ کر تصویر کا پتہ چلا ہے۔ "
                    "اس تصویر میں پودوں کے پتوں کے نمونے موجود نہیں ہیں۔ "
                    "غلط نتائج سے بچنے کے لیے تشخیصی عمل روک دیا گیا ہے۔"
                ),
                "is_valid_format": True,
                "is_botanical": False,
                "is_irrelevant": True,
                "predicted_class": None,
                "confidence": 0.0,
                "top_3": [],
                "cam_overlay": None,
                "guidance": None
            }

        # Sharpness check
        sharpness_score = check_image_sharpness(image_pil)
        is_blurry = sharpness_score < 40.0

        # Preprocess input tensor
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # =====================================================
        # TIER 3: Neural Model Forward Pass (10-Class)
        # =====================================================
        with torch.set_grad_enabled(generate_cam):
            logits = self.model(input_tensor)
            # Temperature scaling for probability calibration
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1).detach().cpu().numpy()[0]

        # Compute Shannon Predictive Entropy in bits
        entropy = -float(np.sum(probs * np.log2(np.maximum(probs, 1e-12))))

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_class = self.class_names[top_idx]

        # Check if the Neural Network classified it as Irrelevant_Non_Plant
        if "irrelevant" in top_class.lower() or "non_plant" in top_class.lower():
            return {
                "status": "rejected",
                "rejection_type": "irrelevant_data",
                "rejection_message": (
                    f"Irrelevant Non-Plant Data Detected (Model Classification: Non-Plant with {top_prob:.1%} confidence). "
                    "AgriVision AI is strictly trained for agricultural leaf nutrient diagnosis. "
                    "All nutrient deficiency predictions are suppressed."
                ),
                "rejection_message_ur": (
                    f"غیر متعلقہ مواد کا پتہ چلا ہے (ماڈل نے {top_prob:.1%} اعتماد کے ساتھ غیر نباتاتی تصویر قرار دیا ہے)۔ "
                    "یہ نظام صرف پتوں کے غذائی اجزاء کی تشخیص کے لیے ہے۔ تشخیصی عمل روک دیا گیا ہے۔"
                ),
                "is_valid_format": True,
                "is_botanical": True,
                "is_irrelevant": True,
                "predicted_class": None,
                "confidence": 0.0,
                "top_3": [],
                "cam_overlay": None,
                "guidance": None
            }

        # Filter top-3 predictions among actual plant classes (excluding Irrelevant_Non_Plant)
        plant_indices = [
            i for i, c in enumerate(self.class_names)
            if "irrelevant" not in c.lower() and "non_plant" not in c.lower()
        ]
        sorted_plant_indices = sorted(plant_indices, key=lambda idx: probs[idx], reverse=True)[:3]
        top_3 = [
            {"class": self.class_names[idx], "confidence": float(probs[idx])}
            for idx in sorted_plant_indices
        ]

        # Re-verify top plant class confidence
        top_plant_idx = sorted_plant_indices[0]
        top_class = self.class_names[top_plant_idx]
        top_prob = float(probs[top_plant_idx])

        # If highest plant class confidence is suspiciously low (< 28%), treat as unidentifiable
        if top_prob < 0.28:
            return {
                "status": "rejected",
                "rejection_type": "irrelevant_data",
                "rejection_message": (
                    f"Unidentifiable Foliage Pattern (Top confidence: {top_prob:.1%}). "
                    "The leaf visual pattern does not match any recognized nutrient deficiency profile. "
                    "Prediction suppressed to maintain academic integrity."
                ),
                "rejection_message_ur": (
                    "غیر واضح پتے کی تصویر۔ ماڈل کسی خاص غذائی کمی کے نمونے کی تصدیق نہیں کر سکا۔ "
                    "درست رہنمائی کے لیے واضح پتے کی تصویر فراہم کریں۔"
                ),
                "is_valid_format": True,
                "is_botanical": True,
                "is_irrelevant": True,
                "predicted_class": None,
                "confidence": 0.0,
                "top_3": [],
                "cam_overlay": None,
                "guidance": None
            }

        is_confident = (top_prob >= CONFIDENCE_THRESHOLD) and not is_blurry

        # =====================================================
        # TIER 4: Grad-CAM Explainability (For Valid Leaf Only)
        # =====================================================
        cam_overlay = None
        if generate_cam:
            try:
                heatmap = self.grad_cam.generate_heatmap(input_tensor, class_idx=top_plant_idx)
                cam_overlay = self.grad_cam.overlay_heatmap(image_pil, heatmap, alpha=0.45)
            except Exception as e:
                print(f"Grad-CAM generation error: {e}")

        # =====================================================
        # TIER 5: Agronomic Advisory Knowledge Retrieval
        # =====================================================
        matched_key = None
        for key in NUTRIENT_KNOWLEDGE_BASE.keys():
            clean_key = key.lower().replace("(", "").replace(")", "").replace(" ", "")
            clean_top = top_class.lower().replace("(", "").replace(")", "").replace(" ", "").replace("_", "")
            if clean_top in clean_key or clean_key in clean_top:
                matched_key = key
                break

        guidance = NUTRIENT_KNOWLEDGE_BASE.get(matched_key or top_class, {
            "title": top_class,
            "title_ur": top_class,
            "symptoms": ["Consult local agricultural officer for specific symptoms."],
            "symptoms_ur": ["مخصوص علامات کے لیے زرعی ماہر سے رابطہ کریں۔"],
            "causes": ["Unbalanced soil nutrient status."],
            "remedies": ["Perform comprehensive laboratory leaf tissue and soil tests."],
            "fertilizer_recommendation": "Apply soil amendments based on certified soil test results.",
            "ph_relevance": "6.0 - 7.0 (Slightly Acidic to Neutral)"
        })

        return {
            "status": "success",
            "is_valid_format": True,
            "is_botanical": True,
            "is_irrelevant": False,
            "rejection_type": None,
            "rejection_message": None,
            "rejection_message_ur": None,
            "predicted_class": top_class,
            "confidence": top_prob,
            "predictive_entropy": round(entropy, 2),
            "calibration_temperature": round(self.temperature, 2),
            "is_calibrated": True,
            "top_3": top_3,
            "top_predictions": top_3,
            "is_confident": is_confident,
            "is_blurry": is_blurry,
            "sharpness_score": sharpness_score,
            "cam_overlay": cam_overlay,
            "gradcam_overlay": cam_overlay,
            "guidance": guidance,
            "disclaimer_en": DISCLAIMER_TEXT,
            "disclaimer_ur": DISCLAIMER_TEXT_UR
        }
