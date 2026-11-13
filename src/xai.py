"""
Explainable Artificial Intelligence (XAI) - Grad-CAM Implementation.
Generates Gradient-weighted Class Activation Maps to highlight visual regions
influencing the deep learning model's nutritional deficiency diagnosis.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Finds the last convolutional layer of the model backbone for Grad-CAM."""
    arch = architecture.lower()
    if "mobilenet" in arch:
        # In MobileNetV2, features is a Sequential, last conv block is features[-1]
        return model.features[-1]
    elif "resnet" in arch:
        return model.layer4[-1]
    elif "efficientnet" in arch:
        return model.features[-1]
    else:
        # Fallback: scan for the last Conv2d module
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise ValueError("Could not automatically locate the target convolutional layer.")

class GradCAM:
    """
    Self-contained Grad-CAM generator for PyTorch vision models.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, architecture: str = "mobilenet_v2"):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else get_target_layer(model, architecture)
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates a 2D normalized Grad-CAM heatmap array (values between 0.0 and 1.0).
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients to obtain importance weights alpha
        alpha = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = torch.sum(alpha * activations, dim=0)  # [H, W]

        # Apply ReLU to focus on features that have a positive influence
        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()

        # Normalize between 0 and 1
        if np.max(cam_np) > np.min(cam_np):
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def overlay_heatmap(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> Image.Image:
        """
        Overlays the Grad-CAM heatmap onto the original leaf image.
        Uses matplotlib colormap to avoid strict cv2 GUI dependencies.
        """
        import matplotlib.cm as cm

        # Resize heatmap to match original image dimensions
        w, h = original_image.size
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
        heatmap_scaled = np.array(heatmap_img) / 255.0

        # Apply colormap (compatible with modern and legacy matplotlib)
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps[colormap]
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
        colored_heatmap = cmap(heatmap_scaled)[:, :, :3]  # RGB, float in [0, 1]
        colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

        # Blend with original leaf image
        orig_np = np.array(original_image.convert("RGB"))
        blended = (alpha * colored_heatmap_uint8 + (1 - alpha) * orig_np).astype(np.uint8)
        
        return Image.fromarray(blended)

    def close(self):
        """Removes forward and backward hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
