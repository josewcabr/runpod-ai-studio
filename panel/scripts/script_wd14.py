#!/usr/bin/env python3
"""
WD14 Image Tagger Script
Reads images from /workspace/training/raw-images/
Saves tags to /workspace/training/captions/wd14/
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

RAW_DIR           = Path("/workspace/training/raw-images")
OUTPUT_DIR        = Path("/workspace/training/captions/wd14")
EXTENSIONS        = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_REPO        = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
GENERAL_THRESHOLD = 0.35
CHAR_THRESHOLD    = 0.35
IMAGE_SIZE        = 448


def load_model():
    print(f"Descargando modelo WD14: {MODEL_REPO}", flush=True)
    model_path = hf_hub_download(MODEL_REPO, filename="model.onnx")
    tags_path  = hf_hub_download(MODEL_REPO, filename="selected_tags.csv")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session   = ort.InferenceSession(model_path, providers=providers)
    tags_df   = pd.read_csv(tags_path)
    return session, tags_df


def preprocess(img_path: Path) -> np.ndarray:
    img    = Image.open(img_path).convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr[:, :, ::-1]         # RGB → BGR
    arr = np.expand_dims(arr, 0)  # batch dim
    return arr


def predict(session, arr: np.ndarray, tags_df: pd.DataFrame, prefix: str) -> str:
    input_name = session.get_inputs()[0].name
    probs      = session.run(None, {input_name: arr})[0][0]

    tags = []
    for i, prob in enumerate(probs):
        if i >= len(tags_df):
            break
        row       = tags_df.iloc[i]
        cat       = row.get("category", 0)
        threshold = CHAR_THRESHOLD if cat == 4 else GENERAL_THRESHOLD
        if prob >= threshold:
            tags.append(row["name"].replace("_", " "))

    tags_str = ", ".join(tags)
    return f"{prefix}, {tags_str}".strip(", ") if prefix else tags_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="", help="Trigger word prefix")
    args   = parser.parse_args()
    prefix = args.prefix.strip()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in EXTENSIONS]
    if not images:
        print(f"No se encontraron imágenes en {RAW_DIR}", flush=True)
        sys.exit(0)

    total             = len(images)
    session, tags_df  = load_model()
    used_providers    = session.get_providers()
    device            = "CUDA" if "CUDAExecutionProvider" in used_providers else "CPU"
    print(f"Dispositivo: {device}", flush=True)
    print(f"\nProcesando {total} imágenes...\n", flush=True)

    errors = 0
    for idx, img_path in enumerate(sorted(images), 1):
        print(f"[{idx}/{total}] {img_path.name}", flush=True)
        try:
            arr    = preprocess(img_path)
            result = predict(session, arr, tags_df, prefix)

            out_path = OUTPUT_DIR / (img_path.stem + ".txt")
            out_path.write_text(result, encoding="utf-8")
            print(f"  → {result[:80]}{'...' if len(result) > 80 else ''}", flush=True)
        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}", flush=True)

    print(f"\n✓ Completado: {total - errors}/{total} imágenes procesadas.", flush=True)
    if errors:
        print(f"  {errors} imagen(es) con error.", flush=True)


if __name__ == "__main__":
    main()
