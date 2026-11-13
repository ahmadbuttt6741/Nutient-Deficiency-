"""
Measures inference latency distributions (p50, p95, p99), throughput,
and model size metrics on CPU.
"""

import os
import time
import json
import torch
import numpy as np
from PIL import Image

from src.inference import NutrientDeficiencyPredictor
from src.config import SAVED_MODEL_PATH

def benchmark_inference():
    print("=" * 65)
    print("PHASE 3: INFERENCE LATENCY & MODEL PROFILE BENCHMARK")
    print("=" * 65)

    predictor = NutrientDeficiencyPredictor()
    test_leaf = Image.open("app/assets/samples/healthy_sample.jpg").convert("RGB")

    # Warmup
    print("Warming up model on CPU (5 iterations)...")
    for _ in range(5):
        predictor.predict(test_leaf, generate_cam=False)

    latencies_no_cam = []
    print("Benchmarking 100 runs without CAM...")
    for _ in range(100):
        t0 = time.perf_counter()
        predictor.predict(test_leaf, generate_cam=False)
        latencies_no_cam.append((time.perf_counter() - t0) * 1000.0) # ms

    latencies_with_cam = []
    print("Benchmarking 50 runs with Grad-CAM...")
    for _ in range(50):
        t0 = time.perf_counter()
        predictor.predict(test_leaf, generate_cam=True)
        latencies_with_cam.append((time.perf_counter() - t0) * 1000.0) # ms

    # Latency stats
    lat_no_cam = np.array(latencies_no_cam)
    lat_cam = np.array(latencies_with_cam)

    # Model size stats
    model_size_mb = os.path.getsize(SAVED_MODEL_PATH) / (1024 * 1024)
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)

    results = {
        "model_file_size_mb": round(model_size_mb, 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "pure_inference_ms": {
            "mean": round(float(np.mean(lat_no_cam)), 2),
            "median_p50": round(float(np.median(lat_no_cam)), 2),
            "p95": round(float(np.percentile(lat_no_cam, 95)), 2),
            "p99": round(float(np.percentile(lat_no_cam, 99)), 2),
            "min": round(float(np.min(lat_no_cam)), 2),
            "max": round(float(np.max(lat_no_cam)), 2)
        },
        "full_pipeline_with_cam_ms": {
            "mean": round(float(np.mean(lat_cam)), 2),
            "median_p50": round(float(np.median(lat_cam)), 2),
            "p95": round(float(np.percentile(lat_cam, 95)), 2),
            "p99": round(float(np.percentile(lat_cam, 99)), 2)
        }
    }

    print("\nBENCHMARK RESULTS:")
    print(f"   Model Weights Size:       {model_size_mb:.2f} MB")
    print(f"   Total Parameters:         {total_params:,}")
    print(f"   Inference Latency (Mean): {results['pure_inference_ms']['mean']} ms ({1000/results['pure_inference_ms']['mean']:.1f} FPS)")
    print(f"   Inference Latency (p50):  {results['pure_inference_ms']['median_p50']} ms")
    print(f"   Inference Latency (p95):  {results['pure_inference_ms']['p95']} ms")
    print(f"   Inference Latency (p99):  {results['pure_inference_ms']['p99']} ms")
    print(f"   With Grad-CAM Heatmap:    {results['full_pipeline_with_cam_ms']['mean']} ms")

    with open("models/latency_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    benchmark_inference()
