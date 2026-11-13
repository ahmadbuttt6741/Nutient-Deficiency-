"""
Generates and curates the 10th dataset class: Irrelevant_Non_Plant.
Includes real-world non-plant photographs (urban, objects, animals, architecture)
and procedural non-plant data (text documents, geometric patterns, noise, solid gradients).
Uses ThreadPoolExecutor for high-speed concurrent fetching.
"""

import os
import io
import ssl
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image, ImageDraw

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

def generate_procedural_image(img_id: int) -> Image.Image:
    """Generates a non-plant procedural image (document, geometry, noise, or texture)."""
    kind = img_id % 5
    size = (224, 224)
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if kind == 0:
        # Document / Text / Receipt simulation
        bg_col = random.choice([(255, 255, 255), (245, 245, 240), (230, 235, 240)])
        img = Image.new("RGB", size, bg_col)
        draw = ImageDraw.Draw(img)
        y = random.randint(15, 30)
        while y < 200:
            line_len = random.randint(100, 190)
            height = random.randint(4, 9)
            draw.rectangle([20, y, 20 + line_len, y + height], fill=(random.randint(20, 70), random.randint(20, 70), random.randint(20, 70)))
            y += height + random.randint(8, 14)

    elif kind == 1:
        # Abstract Geometry / Modern Art
        bg = (random.randint(0, 50), random.randint(0, 50), random.randint(50, 100))
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(5, 12)):
            x1, y1 = random.randint(0, 200), random.randint(0, 200)
            x2, y2 = x1 + random.randint(20, 80), y1 + random.randint(20, 80)
            col = (random.randint(150, 255), random.randint(50, 200), random.randint(50, 255))
            shape = random.choice(["circle", "rect", "line"])
            if shape == "circle":
                draw.ellipse([x1, y1, x2, y2], fill=col)
            elif shape == "rect":
                draw.rectangle([x1, y1, x2, y2], fill=col)
            else:
                draw.line([x1, y1, x2, y2], fill=col, width=random.randint(2, 6))

    elif kind == 2:
        # High-frequency Noise / Texture
        noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(noise)

    elif kind == 3:
        # Color Gradients / Solid color block
        base_color = random.choice([
            (220, 38, 38),   # Vibrant Red
            (37, 99, 235),   # Blue
            (147, 51, 234),  # Purple
            (234, 88, 12),   # Orange
            (100, 116, 139), # Slate Grey
            (15, 23, 42),    # Dark Charcoal
            (248, 250, 252), # Off-white
        ])
        img = Image.new("RGB", size, base_color)
        draw = ImageDraw.Draw(img)
        contrasting = (255 - base_color[0], 255 - base_color[1], 255 - base_color[2])
        draw.line([0, 0, 224, 224], fill=contrasting, width=4)
        draw.line([0, 224, 224, 0], fill=contrasting, width=4)

    elif kind == 4:
        # Checkerboard / Grid pattern
        cell_size = random.choice([16, 28, 32])
        c1 = (random.randint(0, 100), random.randint(0, 100), random.randint(100, 200))
        c2 = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        for row in range(0, 224, cell_size):
            for col in range(0, 224, cell_size):
                fill = c1 if ((row // cell_size) + (col // cell_size)) % 2 == 0 else c2
                draw.rectangle([col, row, col + cell_size, row + cell_size], fill=fill)

    return img

def fetch_picsum_image(photo_id: int) -> Image.Image:
    """Fetches real-world photo from Picsum."""
    url = f"https://picsum.photos/id/{photo_id}/224/224"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=5) as resp:
        data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")

def build_irrelevant_dataset():
    train_dir = "data/processed/train/Irrelevant_Non_Plant"
    val_dir = "data/processed/val/Irrelevant_Non_Plant"
    test_dir = "data/processed/test/Irrelevant_Non_Plant"

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    total_needed = 280 + 35 + 50  # 365 total
    images = []

    print(f"Collecting {total_needed} diverse irrelevant non-plant images...", flush=True)

    # Fast concurrent fetching for real photos
    photo_ids = list(range(10, 350))
    random.shuffle(photo_ids)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_picsum_image, pid): pid for pid in photo_ids[:250]}
        for future in as_completed(futures):
            if len(images) >= 230:
                break
            try:
                img = future.result()
                images.append(img)
            except Exception:
                pass

    print(f"Real-world photos collected: {len(images)}", flush=True)

    # Fill remaining slots with synthetic non-plant images
    synthetic_id = 0
    while len(images) < total_needed:
        images.append(generate_procedural_image(synthetic_id))
        synthetic_id += 1

    print(f"Total dataset assembled: {len(images)} images.", flush=True)
    random.shuffle(images)

    # Split into train, val, test
    train_imgs = images[:280]
    val_imgs = images[280:315]
    test_imgs = images[315:365]

    for idx, img in enumerate(train_imgs):
        img.save(os.path.join(train_dir, f"Irrelevant_Non_Plant_train_{idx:04d}.jpg"), quality=90)
    for idx, img in enumerate(val_imgs):
        img.save(os.path.join(val_dir, f"Irrelevant_Non_Plant_val_{idx:04d}.jpg"), quality=90)
    for idx, img in enumerate(test_imgs):
        img.save(os.path.join(test_dir, f"Irrelevant_Non_Plant_test_{idx:04d}.jpg"), quality=90)

    print(f"Done! Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}", flush=True)

if __name__ == "__main__":
    build_irrelevant_dataset()
