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
#   7860 → Forge/Kohya (slot exclusivo, lanzar desde el panel)
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
CUDA_ENV_FILE="/root/.studio_cuda_env"

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

# Asegurar setuptools/wheel en sistema — necesario para el bootstrap de Forge
# (pkg_resources viene de setuptools)
pip install --upgrade pip setuptools wheel -q 2>/dev/null || true

# Dependencias del panel (van al Python del sistema)
if ! pip install flask requests psutil --ignore-installed 2>&1 | tee -a "$LOG_FILE"; then
    log "⚠  ADVERTENCIA: pip install panel deps falló — el panel puede no arrancar"
fi
log "✅ Paquetes listos"

# ── 2. Configuración de Accelerate ───────────────────────────────
section "Accelerate config"
ACCEL_DIR="/root/.cache/huggingface/accelerate"
mkdir -p "$ACCEL_DIR"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"

if (( GPU_COUNT > 1 )); then
    log "Multi-GPU detectado ($GPU_COUNT GPUs)"
    cp "$REPO_DIR/configs/accelerate/multi_gpu.yaml" "$ACCEL_DIR/default_config.yaml"
    # Regex robusta: reemplaza cualquier número en num_processes
    sed -i "s/num_processes: [0-9]*/num_processes: $GPU_COUNT/g" "$ACCEL_DIR/default_config.yaml"
else
    log "Single GPU"
    cp "$REPO_DIR/configs/accelerate/single_gpu.yaml" "$ACCEL_DIR/default_config.yaml"
fi
log "✅ Accelerate configurado"

# ── 3. Kohya_ss ──────────────────────────────────────────────────
section "Kohya_ss"
cd "$WORKSPACE" || { log "FATAL: No se puede acceder a $WORKSPACE"; exit 1; }

if [ ! -d "kohya_ss" ]; then
    git clone --recursive https://github.com/bmaltais/kohya_ss.git
fi
cd kohya_ss || { log "FATAL: No se puede acceder a kohya_ss"; exit 1; }
git checkout master 2>/dev/null || true
git pull --recurse-submodules 2>&1 | tee -a "$LOG_FILE" || \
    log "⚠  git pull kohya_ss falló (continuando con versión actual)"

if ! is_done "$WORKSPACE/kohya_ss" || [ ! -d "$WORKSPACE/kohya_ss/venv" ]; then
    log "Instalando dependencias de Kohya..."

    # Fix de rutas CUDNN/TensorRT (necesario en runpod/pytorch)
    CUDNN_PATH=$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null || echo "")
    TENSORRT_PATH=$(python3 -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null || echo "")
    [ -n "$CUDNN_PATH" ]    && export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
    [ -n "$TENSORRT_PATH" ] && export LD_LIBRARY_PATH="$TENSORRT_PATH:${LD_LIBRARY_PATH:-}"

    # Persistir LD_LIBRARY_PATH para que las sesiones tmux lo hereden
    {
        echo "# Generado por setup.sh — rutas CUDA para Kohya/ComfyUI"
        [ -n "$CUDNN_PATH" ]    && echo "export LD_LIBRARY_PATH=\"$CUDNN_PATH/lib:\${LD_LIBRARY_PATH:-}\""
        [ -n "$TENSORRT_PATH" ] && echo "export LD_LIBRARY_PATH=\"$TENSORRT_PATH:\${LD_LIBRARY_PATH:-}\""
    } > "$CUDA_ENV_FILE"

    apt-get install -y python3-tk 2>/dev/null || true

    # ── Crear venv propio ANTES de cualquier pip install ─────────────
    # No se usa ./setup.sh de kohya: en RunPod falla al crear el venv y
    # cae al Python del sistema, contaminándolo e instalando fuera del venv.
    KOHYA_VENV="$WORKSPACE/kohya_ss/venv"
    log "Creando venv de Kohya..."
    python3 -m venv "$KOHYA_VENV"
    # shellcheck disable=SC1091
    source "$KOHYA_VENV/bin/activate"
    pip install --upgrade pip wheel setuptools -q
    # Verificar que pkg_resources es importable — setuptools puede fallar silenciosamente con -q
    python -c "import pkg_resources" 2>/dev/null || pip install --force-reinstall setuptools -q
    log "Venv activado: $KOHYA_VENV"

    # ── Constraints PRIMERO: protege numpy y torch desde el inicio ────
    # El constraints file debe existir ANTES de instalar torch para que
    # torchvision no arrastre numpy 2.x como dependencia transitiva.
    cat > /tmp/kohya_constraints.txt <<'EOF'
torch==2.1.2+cu121
torchvision==0.16.2+cu121
torchaudio==2.1.2+cu121
xformers==0.0.23.post1
numpy==1.26.4
EOF

    # ── Stack PyTorch controlado para RTX 4090 (CUDA 12.1) ───────────
    log "Instalando PyTorch 2.1.2+cu121..."
    pip install -q \
        torch==2.1.2+cu121 \
        torchvision==0.16.2+cu121 \
        torchaudio==2.1.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
        -c /tmp/kohya_constraints.txt

    log "Verificando PyTorch..."
    python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" \
        2>&1 | tee -a "$LOG_FILE" || true

    # xformers exacto para torch 2.1.2 — --no-deps para no arrastrar otro torch
    log "Instalando xformers..."
    pip install --no-deps -q xformers==0.0.23.post1

    # ── requirements.txt de kohya (excluye torch/xformers ya instalados) ─
    # Se filtran también entradas -e ./ para manejar sd-scripts explícitamente
    log "Instalando requirements.txt de Kohya..."
    grep -vE "^\s*(torch|torchvision|torchaudio|xformers)([ =><!]|$)" \
        "$WORKSPACE/kohya_ss/requirements.txt" \
    | grep -vE "^\s*-e\s+\." \
        > /tmp/kohya_requirements_filtered.txt
    pip install -q \
        -r /tmp/kohya_requirements_filtered.txt \
        -c /tmp/kohya_constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121

    # ── sd-scripts: backend real de entrenamiento (LoRA, DreamBooth, FLUX) ─
    # Instalación explícita con ruta absoluta para evitar ambigüedad de CWD
    log "Instalando sd-scripts..."
    pip install -q -e "$WORKSPACE/kohya_ss/sd-scripts" \
        -c /tmp/kohya_constraints.txt

    # ── Extras críticos para LoRA y FLUX (no siempre en requirements.txt) ─
    log "Instalando extras de entrenamiento (LoRA/FLUX)..."
    pip install -q \
        lycoris-lora \
        prodigyopt \
        schedulefree \
        lion-pytorch \
        open-clip-torch \
        -c /tmp/kohya_constraints.txt

    # ── bitsandbytes: necesario para AdamW8bit y optimizadores 8-bit ────
    # Se instala explícitamente porque puede fallar silenciosamente desde
    # requirements.txt (versión incompatible o error de compilación CUDA).
    # 0.43.3 es la última versión estable con CUDA 12.1 + torch 2.1.2.
    log "Instalando bitsandbytes para optimizadores 8-bit..."
    pip install -q bitsandbytes==0.43.3

    deactivate 2>/dev/null || true

    mark_done "$WORKSPACE/kohya_ss"
    log "✅ Kohya_ss instalado"
else
    log "⏭  Ya instalado"
fi

# ── 4. ComfyUI ───────────────────────────────────────────────────
section "ComfyUI"
cd "$WORKSPACE" || { log "FATAL: No se puede acceder a $WORKSPACE"; exit 1; }

if [ ! -d "ComfyUI" ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
fi

if ! is_done "$WORKSPACE/ComfyUI" || [ ! -d "$WORKSPACE/ComfyUI/venv" ]; then
    cd ComfyUI || { log "FATAL: No se puede acceder a ComfyUI"; exit 1; }

    log "Creando venv de ComfyUI..."
    python3 -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    pip install --upgrade pip wheel -q

    # ── Constraints PRIMERO: protege torch de ser sobreescrito ────────────
    # Sin esto, pip install -r requirements.txt reemplaza torch+cu121 con un
    # torch CPU-only desde PyPI → CUDA not available → crash en model_management.
    cat > /tmp/comfy_constraints.txt <<'EOF'
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
EOF

    # ── Stack PyTorch cu121 para ComfyUI ──────────────────────────────────
    # cu121: coincide con el toolkit CUDA 12.1.1 de la imagen base y con
    # cuDNN 8.x del sistema. cu124+ requiere cuDNN 9.x → CUDNN_STATUS_NOT_INITIALIZED.
    # torch 2.5.1: tiene torch.library.custom_op que comfy_kitchen requiere.
    log "Instalando PyTorch cu121 para ComfyUI..."
    pip install -q \
        torch==2.5.1+cu121 \
        torchvision==0.20.1+cu121 \
        torchaudio==2.5.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

    log "Verificando PyTorch..."
    python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" \
        2>&1 | tee -a "$LOG_FILE" || true

    # ── requirements.txt de ComfyUI (filtrando torch para no sobreescribirlo) ─
    log "Instalando dependencias de ComfyUI..."
    grep -vE "^\s*(torch|torchvision|torchaudio)([ =><!]|$)" \
        requirements.txt > /tmp/comfy_requirements_filtered.txt
    pip install -q \
        -r /tmp/comfy_requirements_filtered.txt \
        -c /tmp/comfy_constraints.txt

    # ComfyUI Manager
    cd custom_nodes || { log "FATAL: No se puede acceder a ComfyUI/custom_nodes"; exit 1; }
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git 2>/dev/null || \
        log "⚠  ComfyUI-Manager no se pudo clonar, continuando sin él"
    [ -f ComfyUI-Manager/requirements.txt ] && \
        pip install -q -r ComfyUI-Manager/requirements.txt -c /tmp/comfy_constraints.txt
    deactivate

    mark_done "$WORKSPACE/ComfyUI"
    log "✅ ComfyUI + Manager instalados"
else
    log "⏭  Ya instalado"
fi


# ── 5. Estructura de modelos y symlinks ───────────────────────────
section "Modelos y symlinks"
mkdir -p "$MODELS_DIR"/{checkpoints,loras,vae,clip,unet,controlnet,embeddings,upscalers,hypernetworks}
mkdir -p "$WORKSPACE/training"/{images,raw-images,captions,output,config}
mkdir -p "$WORKSPACE/training/captions"/{blip,wd14}
mkdir -p "$WORKSPACE/training/output/loras"

# ComfyUI
COMFY_M="$WORKSPACE/ComfyUI/models"
mkdir -p "$COMFY_M"
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

# extra_model_paths.yaml: ComfyUI busca loras también en training/output/loras.
# Los .safetensors de Kohya aparecen en ComfyUI automáticamente al terminar el training.
cat > "$WORKSPACE/ComfyUI/extra_model_paths.yaml" <<'EOF'
comfyui:
    loras: /workspace/training/output/loras/
EOF

# Forge (solo si ya está instalado — se instala desde el panel)
if [ -d "$WORKSPACE/stable-diffusion-webui-forge" ]; then
    declare -A WEBUI_MAP=(
        ["Stable-diffusion"]="checkpoints"
        ["Lora"]="loras"
        ["VAE"]="vae"
        ["embeddings"]="embeddings"
        ["ControlNet"]="controlnet"
        ["ESRGAN"]="upscalers"
        ["Hypernetwork"]="hypernetworks"
    )
    for WEBUI_M in "$WORKSPACE/stable-diffusion-webui-forge/models"; do
        mkdir -p "$WEBUI_M"
        for dst in "${!WEBUI_MAP[@]}"; do
            rm -rf "${WEBUI_M:?}/$dst"
            ln -sfn "$MODELS_DIR/${WEBUI_MAP[$dst]}" "$WEBUI_M/$dst"
        done
    done
    log "✅ Symlinks Forge creados"
fi

log "✅ Symlinks creados — /workspace/models es la fuente única"

# ── 6. Arrancar servicios con tmux ───────────────────────────────
section "Arrancando servicios"

tmux kill-server 2>/dev/null || true
sleep 1
tmux new-session -d -s studio -x 220 -y 50

# Prefijo común: cargar vars CUDA si existen
CUDA_SOURCE="[ -f $CUDA_ENV_FILE ] && source $CUDA_ENV_FILE; "

# ── Panel de control (:3000) — siempre activo
tmux rename-window -t studio:0 'panel'
tmux send-keys -t studio:panel \
    "${CUDA_SOURCE}cd $REPO_DIR/panel && python3 app.py 2>&1 | tee $LOGS_DIR/panel.log" Enter

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
# Lanza en background dentro de la ventana tmux y escribe el PID para que
# el panel pueda detectar y gestionar este proceso (stop/restart).
mkdir -p "$LOGS_DIR/pids"
tmux new-window -t studio -n 'comfyui'
tmux send-keys -t studio:comfyui \
    "${CUDA_SOURCE}[ -f $WORKSPACE/ComfyUI/venv/bin/activate ] && source $WORKSPACE/ComfyUI/venv/bin/activate || true; \
cd $WORKSPACE/ComfyUI && \
python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
>> $LOGS_DIR/comfyui.log 2>&1 & echo \$! > $LOGS_DIR/pids/comfyui.pid; wait" Enter

# Forge y Kohya se lanzan desde el panel (slot exclusivo :7860)

# ── Reinicio Jupyter (arranca mal en el primer boot, se reinicia al final del setup)
pkill -f jupyter || true
sleep 3
tmux send-keys -t studio:jupyter "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/ 2>&1 | tee $LOGS_DIR/jupyter.log" Enter

section "Setup completado ✅"
log "  Panel de control  → puerto 3000"
log "  Jupyter Lab       → puerto 8888"
log "  ComfyUI           → puerto 8188  (arrancando...)"
log "  Forge/Kohya → puerto 7860  (lanzar desde el panel)"
log "  TensorBoard       → puerto 6006  (auto con Kohya)"
log ""
log "  tmux attach -t studio   → ver servicios activos"
log "  Ventanas tmux: panel | jupyter | comfyui"
log "  Modelos en /workspace/models/"
