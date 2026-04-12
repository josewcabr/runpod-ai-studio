#!/bin/bash
# =================================================================
# RunPod AI Studio — setup.sh
# Base image: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
#
# Start Command del template RunPod:
#   bash -c "
#     if [ -d /workspace/studio ]; then
#       git -C /workspace/studio pull --quiet
#     else
#       git clone https://github.com/josewcabr/runpod-ai-studio.git /workspace/studio
#     fi
#     bash /workspace/studio/setup.sh
#     sleep infinity
#   "
#
# Puertos a exponer en RunPod:
#   3000 → Control Panel   (siempre activo)
#   7860 → A1111/Forge/Kohya (slot exclusivo, lanzar desde el panel)
#   8188 → ComfyUI         (siempre activo)
#   8888 → Jupyter Lab     (siempre activo)
#   6006 → TensorBoard     (auto al lanzar Kohya)
#
# RunPod Secrets recomendados:
#   HF_TOKEN      → HuggingFace (necesario para FLUX.1-dev)
#   CIVITAI_TOKEN → CivitAI
# =================================================================
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_ROOT_USER_ACTION=ignore

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/studio"
MODELS_DIR="$WORKSPACE/models"
LOGS_DIR="$WORKSPACE/logs"

# ── Helpers ───────────────────────────────────────────────────────
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/setup.log"

log()     { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
section() { echo "" | tee -a "$LOG_FILE"; log "══ $* ══"; }
done_flag() { echo "$1/.setup_done"; }
is_done()   { [ -f "$(done_flag "$1")" ]; }
mark_done() { touch "$(done_flag "$1")"; }

log "RunPod AI Studio — setup iniciado"
log "GPU count: ${RUNPOD_GPU_COUNT:-1}"
[ -z "${HF_TOKEN:-}" ]      && log "⚠  HF_TOKEN no detectado (FLUX.1-dev no disponible)"
[ -z "${CIVITAI_TOKEN:-}" ] && log "⚠  CIVITAI_TOKEN no detectado"

# ── 1. Paquetes del sistema (siempre, rápido) ─────────────────────
section "Paquetes del sistema"
apt-get update -y 2>&1 | tail -1

# Instalar en grupos — si un paquete falla el resto sigue
apt-get install -y git wget curl tmux screen vim bc || true
apt-get install -y python3-venv python3-tk || true
apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev || true
apt-get install -y libgoogle-perftools4 libtcmalloc-minimal4 || true
apt-get install -y libnccl2 libnccl-dev || true
apt-get install -y build-essential || true

# Verificar que tmux está instalado — es crítico para lanzar servicios
if ! command -v tmux &>/dev/null; then
    log "⚠ tmux no encontrado, reintentando..."
    apt-get install -y tmux
fi

# Dependencias del panel (van al Python del sistema)
pip install --upgrade pip 2>/dev/null
pip install flask requests psutil --ignore-installed 2>&1 | tee -a "$LOG_FILE" || true
log "✅ Paquetes listos"

# ── 2. Configuración de Accelerate ───────────────────────────────
section "Accelerate config"
ACCEL_DIR="/root/.cache/huggingface/accelerate"
mkdir -p "$ACCEL_DIR"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"

if (( GPU_COUNT > 1 )); then
    log "Multi-GPU detectado ($GPU_COUNT GPUs)"
    cp "$REPO_DIR/configs/accelerate/multi_gpu.yaml" "$ACCEL_DIR/default_config.yaml"
    sed -i "s/num_processes: 4/num_processes: $GPU_COUNT/g" "$ACCEL_DIR/default_config.yaml"
else
    log "Single GPU"
    cp "$REPO_DIR/configs/accelerate/single_gpu.yaml" "$ACCEL_DIR/default_config.yaml"
fi
log "✅ Accelerate configurado"

# ── 3. Kohya_ss ──────────────────────────────────────────────────
section "Kohya_ss"
cd "$WORKSPACE"

if [ ! -d "kohya_ss" ]; then
    git clone --recursive https://github.com/bmaltais/kohya_ss.git
fi
cd kohya_ss
git checkout master 2>/dev/null || true
git pull --recurse-submodules 2>/dev/null || true

if ! is_done "$WORKSPACE/kohya_ss"; then
    log "Instalando dependencias de Kohya..."

    # Fix de rutas CUDNN/TensorRT (necesario en runpod/pytorch)
    CUDNN_PATH=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null || echo "")
    TENSORRT_PATH=$(python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null || echo "")
    [ -n "$CUDNN_PATH" ]    && export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
    [ -n "$TENSORRT_PATH" ] && export LD_LIBRARY_PATH="$TENSORRT_PATH:${LD_LIBRARY_PATH:-}"

    # Setup oficial de kohya (gestiona su propio entorno)
    chmod +x ./setup.sh

    # Asegurar python3-tk antes de que el setup de Kohya lo compruebe
    apt-get install -y python3-tk 2>/dev/null || true

    ./setup.sh -n -p -r -s -u


    # Kohya instala torch 2.5.0+cu124 y xformers 0.0.28.post2 por su cuenta.
    # Solo aseguramos torchaudio compatible y bitsandbytes actualizado.
    log "Aplicando fixes de bitsandbytes y torchaudio..."
    pip uninstall bitsandbytes -y 2>/dev/null || true
    pip install bitsandbytes --upgrade -q
    pip install -q \
        torchaudio==2.5.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

    mark_done "$WORKSPACE/kohya_ss"
    log "✅ Kohya_ss instalado"
else
    log "⏭  Ya instalado"
fi

# ── 4. ComfyUI ───────────────────────────────────────────────────
section "ComfyUI"
cd "$WORKSPACE"

if [ ! -d "ComfyUI" ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
fi

if ! is_done "$WORKSPACE/ComfyUI"; then
    cd ComfyUI
    log "Instalando dependencias de ComfyUI..."
    pip install -q -r requirements.txt

    # ComfyUI Manager
    cd custom_nodes
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git 2>/dev/null || true
    [ -f ComfyUI-Manager/requirements.txt ] && pip install -q -r ComfyUI-Manager/requirements.txt

    mark_done "$WORKSPACE/ComfyUI"
    log "✅ ComfyUI + Manager instalados"
else
    log "⏭  Ya instalado"
fi

# ── 5. Automatic1111 ─────────────────────────────────────────────
section "Automatic1111"
cd "$WORKSPACE"

if [ ! -d "stable-diffusion-webui" ]; then
    git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    log "✅ Clonado (se auto-instala en el primer arranque)"
else
    log "⏭  Ya clonado"
fi

# ── 6. Forge ─────────────────────────────────────────────────────
section "Forge"
cd "$WORKSPACE"

if [ ! -d "stable-diffusion-webui-forge" ]; then
    git clone --depth 1 https://github.com/lllyasviel/stable-diffusion-webui-forge.git
    log "✅ Clonado (se auto-instala en el primer arranque)"
else
    log "⏭  Ya clonado"
fi

# ── 7. Estructura de modelos y symlinks ───────────────────────────
section "Modelos y symlinks"
mkdir -p "$MODELS_DIR"/{checkpoints,loras,vae,clip,unet,controlnet,embeddings,upscalers,hypernetworks}
mkdir -p "$WORKSPACE/training"/{images,output,config}

# ComfyUI
COMFY_M="$WORKSPACE/ComfyUI/models"
declare -A COMFY_MAP=(
    ["checkpoints"]="checkpoints"
    ["loras"]="loras"
    ["vae"]="vae"
    ["clip"]="clip"
    ["unet"]="unet"
    ["controlnet"]="controlnet"
    ["embeddings"]="embeddings"
    ["upscale_models"]="upscalers"
)
for dst in "${!COMFY_MAP[@]}"; do
    rm -rf "${COMFY_M:?}/$dst"
    ln -sfn "$MODELS_DIR/${COMFY_MAP[$dst]}" "$COMFY_M/$dst"
done

# A1111 y Forge (misma estructura de carpetas entre sí)
declare -A WEBUI_MAP=(
    ["Stable-diffusion"]="checkpoints"
    ["Lora"]="loras"
    ["VAE"]="vae"
    ["embeddings"]="embeddings"
    ["ControlNet"]="controlnet"
    ["ESRGAN"]="upscalers"
    ["Hypernetwork"]="hypernetworks"
)
for WEBUI_M in \
    "$WORKSPACE/stable-diffusion-webui/models" \
    "$WORKSPACE/stable-diffusion-webui-forge/models"
do
    mkdir -p "$WEBUI_M"
    for dst in "${!WEBUI_MAP[@]}"; do
        rm -rf "${WEBUI_M:?}/$dst"
        ln -sfn "$MODELS_DIR/${WEBUI_MAP[$dst]}" "$WEBUI_M/$dst"
    done
done

log "✅ Symlinks creados — /workspace/models es la fuente única"

# ── 8. Arrancar servicios con tmux ───────────────────────────────
section "Arrancando servicios"

tmux kill-server 2>/dev/null || true
sleep 1
tmux new-session -d -s studio -x 220 -y 50

# ── Panel de control (:3000) — siempre activo
tmux rename-window -t studio:0 'panel'
tmux send-keys -t studio:panel \
    "cd $REPO_DIR/panel && python app.py 2>&1 | tee $LOGS_DIR/panel.log" Enter

# ── Jupyter Lab (:8888) — siempre activo
#    Sin token ni contraseña: el proxy de RunPod ya protege el acceso.
#    notebook-dir=/workspace para tener acceso completo al volumen:
#    modelos, training, repos, logs, todo visible y subible desde el navegador.
tmux new-window -t studio -n 'jupyter'
tmux send-keys -t studio:jupyter \
    "jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password='' \
        --notebook-dir=/workspace \
        2>&1 | tee $LOGS_DIR/jupyter.log" Enter

# ── ComfyUI (:8188) — siempre activo
tmux new-window -t studio -n 'comfyui'
tmux send-keys -t studio:comfyui \
    "cd $WORKSPACE/ComfyUI && python main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --enable-cors-header \
        2>&1 | tee $LOGS_DIR/comfyui.log" Enter

# A1111, Forge y Kohya se lanzan desde el panel (slot exclusivo :7860)

section "Setup completado ✅"
log "  Panel de control  → puerto 3000"
log "  Jupyter Lab       → puerto 8888"
log "  ComfyUI           → puerto 8188  (arrancando...)"
log "  A1111/Forge/Kohya → puerto 7860  (lanzar desde el panel)"
log "  TensorBoard       → puerto 6006  (auto con Kohya)"
log ""
log "  tmux attach -t studio   → ver servicios activos"
log "  Ventanas tmux: panel | jupyter | comfyui"
log "  Modelos en /workspace/models/"
