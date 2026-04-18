#!/usr/bin/env python3
"""
BLIP Image Captioning Script
Reads images from /workspace/training/raw-images/
Saves captions to /workspace/training/captions/blip/
"""
import argparse
import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

RAW_DIR    = Path("/workspace/training/raw-images")
OUTPUT_DIR = Path("/workspace/training/captions/blip")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_ID   = "Salesforce/blip-image-captioning-base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="", help="Trigger word prefix")
    args = parser.parse_args()
    prefix = args.prefix.strip()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in EXTENSIONS]
    if not images:
        print(f"No se encontraron imágenes en {RAW_DIR}", flush=True)
        sys.exit(0)

    total = len(images)
    print(f"Cargando modelo BLIP: {MODEL_ID}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}", flush=True)

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model     = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()

    print(f"\nProcesando {total} imágenes...\n", flush=True)
    errors = 0

    for idx, img_path in enumerate(sorted(images), 1):
        print(f"[{idx}/{total}] {img_path.name}", flush=True)
        try:
            image  = Image.open(img_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=75)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()

            final    = f"{prefix} {caption}".strip() if prefix else caption
            out_path = OUTPUT_DIR / (img_path.stem + ".txt")
            out_path.write_text(final, encoding="utf-8")
            print(f"  → {final}", flush=True)
        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}", flush=True)

    print(f"\n✓ Completado: {total - errors}/{total} imágenes procesadas.", flush=True)
    if errors:
        print(f"  {errors} imagen(es) con error.", flush=True)


if __name__ == "__main__":
    main()
