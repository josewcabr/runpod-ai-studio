#!/usr/bin/env python3
"""
AI Studio — Krea 2 LoRA Training Automation Runner (Musubi-Tuner)

Orchestrates the 3-step pipeline:
  Step 1: Cache Latents (Qwen-Image VAE) via krea2_cache_latents.py
  Step 2: Cache Text Encoder Outputs (Qwen3-VL) via krea2_cache_text_encoder_outputs.py
  Step 3: Train LoRA Network via accelerate launch krea2_train_network.py --network_module networks.lora_krea2
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Subprocess tracker for clean signal propagation
_CURRENT_SUBPROC = None


def _sig_handler(signum, frame):
    global _CURRENT_SUBPROC
    print(f"\n[KREA2 RUNNER] Señal recibida ({signum}). Terminando subprocesos...", flush=True)
    if _CURRENT_SUBPROC and _CURRENT_SUBPROC.poll() is None:
        try:
            _CURRENT_SUBPROC.terminate()
            _CURRENT_SUBPROC.wait(timeout=5)
        except Exception:
            try:
                _CURRENT_SUBPROC.kill()
            except Exception:
                pass
    sys.exit(130)


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


def log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def generate_dataset_toml(config: dict, toml_path: Path):
    """
    Generates dataset.toml in the format expected by musubi-tuner for Krea 2.
    """
    reso = config.get("resolution", [1024, 1024])
    if isinstance(reso, int):
        reso = [reso, reso]
    caption_ext = config.get("caption_extension", ".txt")
    if not caption_ext.startswith("."):
        caption_ext = f".{caption_ext}"

    batch_size = int(config.get("batch_size", 1))
    image_dir = str(Path(config["image_directory"]).resolve())
    cache_dir = str(Path(config["cache_directory"]).resolve())
    num_repeats = int(config.get("num_repeats", 1))

    toml_content = f"""# Generado automáticamente por AI Studio para Krea 2 (musubi-tuner)
[general]
resolution = [{reso[0]}, {reso[1]}]
caption_extension = "{caption_ext}"
batch_size = {batch_size}
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "{image_dir}"
cache_directory = "{cache_dir}"
num_repeats = {num_repeats}
"""
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(toml_content, encoding="utf-8")
    log(f"Configuración de dataset escrita en: {toml_path}")


def run_command(cmd: list[str], cwd: Path, step_name: str) -> bool:
    global _CURRENT_SUBPROC
    log(f"--- Iniciando {step_name} ---")
    log(f"Ejecutando: {' '.join(cmd)}")
    log(f"Directorio de trabajo: {cwd}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _CURRENT_SUBPROC = proc

    # Stream output line by line
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    except Exception as e:
        log(f"Error en streaming de salida: {e}")

    proc.wait()
    retcode = proc.returncode
    _CURRENT_SUBPROC = None

    if retcode == 0:
        log(f"✓ {step_name} completado con éxito.")
        return True
    else:
        log(f"✗ ERROR: {step_name} falló con código de salida {retcode}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Runner de entrenamiento Krea 2 LoRA")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo JSON con la configuración del job")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log(f"ERROR: Archivo de configuración {config_path} no encontrado.")
        sys.exit(1)

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"ERROR al parsear {config_path}: {e}")
        sys.exit(1)

    workspace = Path(config.get("workspace", "/workspace"))
    musubi_dir = workspace / "musubi-tuner"
    musubi_venv = musubi_dir / "venv"

    # Determinar ejecutables dentro del venv de musubi-tuner
    python_bin = musubi_venv / "bin" / "python"
    accelerate_bin = musubi_venv / "bin" / "accelerate"

    if not python_bin.exists():
        python_bin = Path(sys.executable)
        log(f"Aviso: Python venv no encontrado en {musubi_venv}, usando {python_bin}")
    if not accelerate_bin.exists():
        accelerate_bin = "accelerate"

    # Validar modelo base DiT
    dit_model = Path(config.get("dit_model", ""))
    if not dit_model.exists():
        # Intentar fallback en /workspace/models/checkpoints/
        candidates = list((workspace / "models" / "checkpoints").glob("*krea*raw*.safetensors"))
        if candidates:
            dit_model = candidates[0]
            log(f"Auto-detectado modelo RAW DiT: {dit_model}")
        else:
            log(f"ERROR: Modelo DiT no encontrado: {dit_model}")
            sys.exit(1)

    # Validar VAE
    vae_model = Path(config.get("vae_model", ""))
    if not vae_model.exists():
        candidates = list((workspace / "models" / "vae").glob("*qwen*image*vae*.safetensors")) + \
                     list((workspace / "models" / "vae").glob("*vae*.safetensors"))
        if candidates:
            vae_model = candidates[0]
            log(f"Auto-detectado VAE: {vae_model}")
        else:
            # Fallback opcional si el modelo DiT contiene VAE o está en checkpoints
            vae_model = dit_model

    # Validar Text Encoder (Qwen3-VL)
    te_model = Path(config.get("text_encoder_model", ""))
    if not te_model.exists():
        candidates = list((workspace / "models" / "text_encoders").glob("*qwen*3*vl*.safetensors")) + \
                     list((workspace / "models" / "clip").glob("*qwen*.safetensors"))
        if candidates:
            te_model = candidates[0]
            log(f"Auto-detectado Text Encoder: {te_model}")
        else:
            te_model = dit_model

    # Dataset e imágenes
    img_dir = Path(config.get("image_directory", workspace / "training" / "images"))
    if not img_dir.exists() or not any(img_dir.iterdir()):
        log(f"ERROR: Directorio de imágenes vacío o no existe: {img_dir}")
        sys.exit(1)

    cache_dir = Path(config.get("cache_directory", workspace / "training" / "cache_krea2"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    config["cache_directory"] = str(cache_dir)
    config["image_directory"] = str(img_dir)

    # Generar dataset.toml
    toml_path = cache_dir / "dataset.toml"
    generate_dataset_toml(config, toml_path)

    skip_existing = config.get("skip_existing_cache", True)
    batch_size = str(config.get("batch_size", 1))

    # ═══════════════════════════════════════════════════════════════
    # PASO 1: Caché de Latentes (Qwen-Image VAE)
    # ═══════════════════════════════════════════════════════════════
    log("=========================================================")
    log("[PASO 1/3] Pre-cacheando latentes de imágenes con VAE...")
    log("=========================================================")
    cache_latents_script = musubi_dir / "src" / "musubi_tuner" / "krea2_cache_latents.py"
    if not cache_latents_script.exists():
        cache_latents_script = musubi_dir / "krea2_cache_latents.py"

    latents_cmd = [
        str(python_bin),
        str(cache_latents_script),
        "--dataset_config", str(toml_path),
        "--vae", str(vae_model),
        "--batch_size", batch_size,
    ]
    if skip_existing:
        latents_cmd.append("--skip_existing")

    ok = run_command(latents_cmd, musubi_dir, "Paso 1: Caché de Latentes")
    if not ok:
        sys.exit(1)

    time.sleep(2)  # Pausa para liberación total de VRAM

    # ═══════════════════════════════════════════════════════════════
    # PASO 2: Caché de Salidas de Texto (Qwen3-VL)
    # ═══════════════════════════════════════════════════════════════
    log("=========================================================")
    log("[PASO 2/3] Pre-cacheando embeddings de texto con Qwen3-VL...")
    log("=========================================================")
    cache_te_script = musubi_dir / "src" / "musubi_tuner" / "krea2_cache_text_encoder_outputs.py"
    if not cache_te_script.exists():
        cache_te_script = musubi_dir / "krea2_cache_text_encoder_outputs.py"

    te_cmd = [
        str(python_bin),
        str(cache_te_script),
        "--dataset_config", str(toml_path),
        "--text_encoder", str(te_model),
        "--batch_size", batch_size,
    ]
    if skip_existing:
        te_cmd.append("--skip_existing")

    ok = run_command(te_cmd, musubi_dir, "Paso 2: Caché de Texto")
    if not ok:
        sys.exit(1)

    time.sleep(2)  # Pausa para liberación total de VRAM

    # ═══════════════════════════════════════════════════════════════
    # PASO 3: Entrenamiento LoRA DiT
    # ═══════════════════════════════════════════════════════════════
    log("=========================================================")
    log("[PASO 3/3] Iniciando entrenamiento de red LoRA Krea 2...")
    log("=========================================================")
    train_script = musubi_dir / "src" / "musubi_tuner" / "krea2_train_network.py"
    if not train_script.exists():
        train_script = musubi_dir / "krea2_train_network.py"

    output_dir = Path(config.get("output_dir", workspace / "training" / "output" / "loras"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = config.get("output_name", "krea2_lora").strip() or "krea2_lora"

    network_dim = int(config.get("network_dim", 32))
    network_alpha = float(config.get("network_alpha", 32))
    learning_rate = str(config.get("learning_rate", "1e-4"))
    optimizer_type = config.get("optimizer_type", "adamw8bit")
    lr_scheduler = config.get("lr_scheduler", "cosine")
    lr_warmup_steps = int(config.get("lr_warmup_steps", 50))
    max_train_epochs = int(config.get("max_train_epochs", 10))
    save_every_n_epochs = int(config.get("save_every_n_epochs", 1))
    mixed_precision = config.get("mixed_precision", "bf16")
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))

    train_cmd = [
        str(accelerate_bin), "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", mixed_precision,
        str(train_script),
        "--dit", str(dit_model),
        "--dataset_config", str(toml_path),
        "--network_module", "networks.lora_krea2",
        "--network_dim", str(network_dim),
        "--network_alpha", str(network_alpha),
        "--optimizer_type", optimizer_type,
        "--learning_rate", learning_rate,
        "--lr_scheduler", lr_scheduler,
        "--max_train_epochs", str(max_train_epochs),
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--output_dir", str(output_dir),
        "--output_name", output_name,
        "--mixed_precision", mixed_precision,
        "--save_precision", mixed_precision,
    ]

    if lr_warmup_steps > 0:
        train_cmd.extend(["--lr_warmup_steps", str(lr_warmup_steps)])

    if grad_accum_steps > 1:
        train_cmd.extend(["--gradient_accumulation_steps", str(grad_accum_steps)])

    if config.get("gradient_checkpointing", True):
        train_cmd.append("--gradient_checkpointing")

    # Optimizaciones VRAM específicas de DiT en musubi-tuner
    if config.get("fp8_base", False):
        train_cmd.append("--fp8_base")
        log("Habilitado: --fp8_base (base weights cuantizados a FP8 para reducir VRAM)")

    blocks_to_swap = int(config.get("blocks_to_swap", 0))
    if blocks_to_swap > 0:
        train_cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])
        log(f"Habilitado: --blocks_to_swap {blocks_to_swap} (intercambio de capas CPU <-> GPU)")

    if config.get("convrot_int8", False):
        train_cmd.append("--convrot_int8")
        log("Habilitado: --convrot_int8 (cuantización INT8)")

    if vae_model.exists() and vae_model != dit_model:
        train_cmd.extend(["--vae", str(vae_model)])

    ok = run_command(train_cmd, musubi_dir, "Paso 3: Entrenamiento LoRA")
    if not ok:
        sys.exit(1)

    log("=========================================================")
    log("🎉 ¡ENTRENAMIENTO KREA 2 COMPLETADO CON ÉXITO!")
    log(f"El archivo LoRA ha sido guardado en: {output_dir / (output_name + '.safetensors')}")
    log("Es compatible directamente con ComfyUI usando Krea 2 Turbo GGUF.")
    log("=========================================================")


if __name__ == "__main__":
    main()
