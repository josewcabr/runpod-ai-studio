# RunPod AI Studio

Panel de control unificado para entornos de generación y entrenamiento de imágenes IA en RunPod. Gestiona múltiples herramientas (ComfyUI, Forge, Kohya_ss) desde una interfaz web centralizada, con descarga de modelos integrada desde CivitAI y HuggingFace.

---

## Requisitos

- **Imagen base**: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **GPU recomendada**: RTX 4090 (24 GB VRAM) o superior
- **Container Disk**: mínimo 50 GB
- **Volume Mount Path**: `/workspace`

### RunPod Secrets (opcionales pero recomendados)

| Variable | Uso |
|---|---|
| `HF_TOKEN` | Necesario para modelos privados de HuggingFace como FLUX.1-dev |
| `CIVITAI_TOKEN` | API Key de CivitAI para modelos restringidos y mayor velocidad |

### Puertos a exponer

| Puerto | Servicio |
|---|---|
| 3000 | Control Panel (siempre activo) |
| 7860 | Slot exclusivo: Forge / Kohya_ss |
| 8188 | ComfyUI (siempre activo) |
| 8888 | Jupyter Lab (siempre activo) |
| 6006 | TensorBoard (auto al lanzar Kohya) |
| 22 | SSH (TCP) |

---

## Start Command del template

```bash
bash -c "
  if [ -d /workspace/studio ]; then
    git -C /workspace/studio pull --quiet
  else
    git clone https://github.com/josewcabr/runpod-ai-studio.git /workspace/studio
  fi
  bash /workspace/studio/setup.sh
  sleep infinity
"
```

El `git pull` garantiza que siempre se usa la última versión del repositorio. El `sleep infinity` mantiene el contenedor vivo tras el setup.

---

## Estructura del repositorio

```
runpod-ai-studio/
├── setup.sh                          # Script principal de instalación
├── README.md
├── panel/
│   ├── app.py                        # Backend Flask (puerto 3000)
│   ├── scripts/
│   │   ├── train_krea2.py            # Runner orquestador Krea 2 (Musubi-Tuner)
│   │   ├── script_blip.py            # Captioning BLIP
│   │   └── script_wd14.py            # Captioning WD14
│   └── static/
│       └── index.html                # Frontend SPA unificado
└── configs/
    └── accelerate/
        ├── single_gpu.yaml           # Config accelerate para 1 GPU
        └── multi_gpu.yaml            # Config accelerate para múltiples GPUs
```

---

## setup.sh — Flujo de instalación

El script es **idempotente**: usa flags `.setup_done` en cada herramienta para no reinstalar si el pod se reinicia. El volumen `/workspace` es persistente entre reinicios.

### Sección 1 — Paquetes del sistema
Instala paquetes apt en grupos separados con `|| true` para que un fallo puntual no detenga el proceso. Verifica explícitamente que `tmux` esté instalado ya que es crítico para el arranque de servicios. Instala `flask`, `requests` y `psutil` en el Python del sistema para el panel.

### Sección 2 — Accelerate
Detecta `$RUNPOD_GPU_COUNT` y copia el yaml correspondiente a `/root/.cache/huggingface/accelerate/default_config.yaml`. Para multi-GPU adapta `num_processes` con `sed`.

### Sección 3 — Kohya_ss (SDXL & SD1.5)
- Clona `https://github.com/bmaltais/kohya_ss.git` con submódulos
- Crea su propio venv `/workspace/kohya_ss/venv` fijado a `torch 2.1.2+cu121`, `xformers 0.0.23.post1` y `sd-scripts`.
- Permanece aislado e intacto para evitar conflictos con modelos modernos DiT.

### Sección 4 — ComfyUI
- Clona `https://github.com/comfyanonymous/ComfyUI.git`
- Entorno virtual con `torch 2.5.1+cu121`.
- Instala ComfyUI Manager desde `https://github.com/ltdrdata/ComfyUI-Manager.git`.
- Configura `extra_model_paths.yaml` para detectar automáticamente los LoRAs generados en `/workspace/training/output/loras/`.

### Sección 5 — Musubi-Tuner (Krea 2 LoRA)
- Clona `https://github.com/kohya-ss/musubi-tuner.git`.
- Crea su entorno virtual dedicado `/workspace/musubi-tuner/venv` con **`torch>=2.6.0`**, `accelerate`, `bitsandbytes`, `transformers` modernos y soporte para **Qwen3-VL**.
- No comparte dependencias con Kohya_ss/sd-scripts para garantizar estabilidad total.

### Sección 6 — Forge
- Clona `https://github.com/lllyasviel/stable-diffusion-webui-forge.git`
- Se auto-instala en el primer arranque desde el panel.

### Sección 7 — Estructura de modelos y symlinks
Crea `/workspace/models/` como **fuente única de modelos**:

```
/workspace/models/
├── checkpoints/
├── loras/
├── vae/
├── clip/
├── unet/
├── controlnet/
├── embeddings/
├── upscalers/
├── text_encoders/
└── hypernetworks/
```

Crea symlinks desde las carpetas que espera cada herramienta:

| Herramienta | Carpeta propia | Apunta a |
|---|---|---|
| ComfyUI | `ComfyUI/models/checkpoints` | `/workspace/models/checkpoints` |
| ComfyUI | `ComfyUI/models/loras` | `/workspace/models/loras` |
| ComfyUI | `ComfyUI/models/upscale_models` | `/workspace/models/upscalers` |
| Forge | `models/Stable-diffusion` | `/workspace/models/checkpoints` |
| Forge | `models/Lora` | `/workspace/models/loras` |
| Forge | `models/VAE` | `/workspace/models/vae` |
| Forge | `models/ControlNet` | `/workspace/models/controlnet` |
| Forge | `models/ESRGAN` | `/workspace/models/upscalers` |

Un modelo descargado una sola vez es visible automáticamente en todas las herramientas.

### Sección 8 — Arranque de servicios con tmux
Crea sesión tmux `studio` con 3 ventanas permanentes:

| Ventana tmux | Proceso | Puerto |
|---|---|---|
| `panel` | `python app.py` | 3000 |
| `jupyter` | `jupyter lab` | 8888 |
| `comfyui` | `python main.py --listen 0.0.0.0 --enable-cors-header` | 8188 |

Forge y Kohya se lanzan desde el panel bajo demanda (slot exclusivo puerto 7860).

---

## panel/app.py — Backend Flask

API REST que gestiona procesos, descargas y modelos. Corre en el Python del sistema (no en venv).

### Definición de servicios

```python
SERVICES = {

    'forge':   { 'group': 'exclusive', 'port': 7860, ... },
    'kohya':   { 'group': 'exclusive', 'port': 7860, 'tensorboard': True, ... },
    'comfyui': { 'group': 'independent', 'port': 8188, ... },
}
```

- **`exclusive`**: solo uno activo a la vez en :7860. Al lanzar uno, para el que esté corriendo.
- **`independent`**: convive con cualquier otro servicio.

### Flags de lanzamiento

| Servicio | Flags relevantes |
|---|---|
| Forge | `--listen --port 7860 --api --cors-allow-origins=*` |
| Kohya | `--server_port 7860 --listen=0.0.0.0 --headless` |
| ComfyUI | `--listen 0.0.0.0 --port 8188 --enable-cors-header` |

### TensorBoard
Se lanza automáticamente junto con Kohya (`tensorboard --logdir=/workspace/logs --port=6006 --host=0.0.0.0`) y se para cuando Kohya se para.

### Gestión de procesos
Usa `os.setsid()` para crear grupos de proceso. Al parar un servicio envía `SIGTERM` al grupo completo y espera 6 segundos antes de `SIGKILL`. Los PIDs se persisten en `/workspace/logs/pids/` para sobrevivir a reinicios del panel.

### Endpoints principales

| Método | Ruta | Función |
|---|---|---|
| GET | `/api/status` | Estado global: GPU, disco, servicios, tokens, descargas activas |
| POST | `/api/services/<sid>/start` | Lanzar servicio |
| POST | `/api/services/<sid>/stop` | Parar servicio |
| POST | `/api/services/<sid>/restart` | Reiniciar servicio |
| GET | `/api/services/<sid>/log` | Últimas N líneas del log |
| GET | `/api/civitai/model?id=<id>` | Info de modelo por ID o URL de CivitAI (civitai.com o civitai.red) |
| POST | `/api/civitai/download` | Iniciar descarga desde CivitAI |
| POST | `/api/direct/download` | Descarga directa por URL arbitraria (con token automático) |
| GET | `/api/hf/model?repo=<repo>` | Listar ficheros descargables de un repo HF |
| POST | `/api/hf/download` | Iniciar descarga desde HuggingFace |
| GET | `/api/downloads` | Estado de todas las descargas |
| DELETE | `/api/downloads/<job_id>` | Eliminar job de la cola |
| GET | `/api/models` | Modelos instalados en `/workspace/models/` |
| POST | `/api/models/delete` | Borrar un modelo (con protección de path traversal) |
| POST | `/api/tokens` | Sobreescribir tokens en memoria |
| GET | `/api/train/krea2/venv-status` | Estado del entorno virtual de Musubi-Tuner |
| POST | `/api/train/krea2/install` | Instalación en segundo plano de Musubi-Tuner |
| GET | `/api/train/krea2/install-log` | Log en streaming de la instalación de Musubi |
| GET | `/api/train/krea2/models` | Listado de checkpoints, VAE y Text Encoders para Krea 2 |
| POST | `/api/train/krea2/start` | Iniciar pipeline de 3 pasos (Caché VAE -> Caché Texto -> Train) |
| GET | `/api/train/krea2/status` | Estado en tiempo real y fase actual del entrenamiento |
| GET | `/api/train/krea2/log` | Streaming de líneas del log de entrenamiento |
| POST | `/api/train/krea2/stop` | Detener entrenamiento y matar subprocesos GPU |
| POST | `/api/train/krea2/clear-cache` | Limpiar caché de latentes y texto en `/workspace/training/cache_krea2` |

### Descargas
Las descargas corren en threads en background usando `requests` con streaming por chunks de 2 MB. El progreso se calcula a partir del header `Content-Length`. Los tokens (`HF_TOKEN`, `CIVITAI_TOKEN`, o variables inyectadas por RunPod como `RUNPOD_SECRET_CIVITAI_TOKEN` y `RUNPOD_SECRET_HF_TOKEN`) se leen y aplican automáticamente.

### CivitAI & civitai.red
Acepta ID de modelo, ID de versión, o URLs completas (incluyendo `civitai.red` y enlaces directos `/api/download/models/<versionId>?fileId=...`). Consulta la API pública correspondiente (`/models/{id}` o `/model-versions/{id}`) y pre-selecciona automáticamente el archivo objetivo si la URL especifica `fileId`. En la descarga, inyecta el token en la query para prevenir conflictos con cabeceras en redirecciones a Cloudflare R2 / AWS S3.

### Descarga Directa por URL
Permite pegar cualquier enlace directo a un binario (`.safetensors`, `.gguf`, etc.), seleccionando la carpeta de destino. El nombre de archivo se extrae automáticamente del header `Content-Disposition` del servidor o de la URL.

### HuggingFace
Acepta `org/repo` o URL completa. La API `https://huggingface.co/api/models/<repo>` devuelve la lista de ficheros del repositorio. El backend filtra solo extensiones descargables (`.safetensors`, `.ckpt`, `.pt`, `.bin`, `.gguf`, `.pth`).

---

## Entrenamiento LoRA Krea 2 (Musubi-Tuner)

Krea 2 es un modelo texto-a-imagen basado en una arquitectura DiT (Diffusion Transformer) de flujo único que emplea el codificador multimodal **Qwen3-VL-4B-Instruct** y el **Qwen-Image VAE**. Para entrenar LoRAs en este modelo sin saturar la VRAM ni mezclar dependencias con SDXL, el panel integra un runner especializado (`panel/scripts/train_krea2.py`) montado sobre **`musubi-tuner`**.

### Flujo de 3 Etapas Automatizado

1. **Paso 1 — Pre-caché de Latentes (`krea2_cache_latents.py`)**:
   - Genera dinámicamente un archivo de configuración `dataset.toml` con buckets, resoluciones y rutas de caché únicas.
   - Codifica las imágenes del dataset a latentes en disco usando el VAE de Krea/Qwen (`qwen_image_vae.safetensors`).
   - Libera completamente la VRAM al terminar el proceso.

2. **Paso 2 — Pre-caché de Embeddings de Texto (`krea2_cache_text_encoder_outputs.py`)**:
   - Procesa los captions asociados a las imágenes extrayendo las representaciones ocultas multi-capa de **Qwen3-VL**.
   - Almacena los tensores pre-calculados en `/workspace/training/cache_krea2`.
   - Libera la memoria GPU antes de pasar al DiT.

3. **Paso 3 — Entrenamiento de Red DiT LoRA (`krea2_train_network.py`)**:
   - Ejecuta `accelerate launch` llamando a la red `--network_module networks.lora_krea2` sobre el modelo base RAW en bf16 (`krea2_raw_bf16.safetensors`).
   - El LoRA entrenado se almacena directamente en `/workspace/training/output/loras/`.
   - **Compatibilidad con ComfyUI**: Aunque se entrena sobre la versión RAW bf16, los pesos del LoRA `.safetensors` resultantes son compatibles al 100% para inferencia en ComfyUI usando checkpoints **Krea 2 Turbo GGUF** (Q4_0, Q8_0, etc.).

### Gestión y Optimización de VRAM (RunPod 24GB GPUs)

Para permitir el entrenamiento en GPUs comerciales como RTX 3090 / RTX 4090 (24 GB VRAM) sin errores de Out-Of-Memory (OOM):
- **`--gradient_checkpointing`**: Reduce drásticamente el consumo de memoria en backpropagation.
- **`--fp8_base`**: Cuantiza las capas del modelo DiT base a precisión FP8 durante el entrenamiento.
- **`--blocks_to_swap <N>`**: Intercambio de bloques entre CPU y GPU (offload dinámico de capas). Un valor entre 8 y 15 permite entrenar cómodamente sin agotar los 24GB.
- **Optimizadores 8-bit**: Soporte nativo para `adamw8bit` y `paged_adamw8bit` mediante `bitsandbytes`.

---

## panel/static/index.html — Frontend SPA

Aplicación de página única en HTML/CSS/JS puro, sin dependencias externas. Se sirve directamente desde Flask.

### Secciones de la UI

**Dashboard**
Vista rápida con tarjetas de estado de todos los servicios activos y descargas en curso. Se actualiza automáticamente cada 3 segundos mediante polling.

**Lanzador**
- Slot exclusivo (:7860): cards de Forge y Kohya con botones start/stop/restart. Muestra aviso de qué servicio se parará al lanzar otro.
- Servicios independientes: ComfyUI con botones open/restart/stop.
- TensorBoard: estado y enlace al puerto 6006, visible solo cuando Kohya está en el slot.

**Krea 2 LoRA**
Panel interactivo dedicado para configurar y lanzar entrenamientos con Musubi-Tuner:
- Comprobación automática del entorno virtual con botón de instalación en 1 clic si no está listo.
- Selectores automáticos de modelo DiT RAW, VAE y Text Encoder detectados en `/workspace/models/`.
- Configuración de resolución, repeticiones, batch size, rank/dim, alpha, learning rate, scheduler y épocas.
- Toggles de optimización de VRAM: Gradient Checkpointing, FP8 Base y control deslizante de Blocks to Swap.
- Terminal en vivo con streaming en tiempo real de logs y badges de estado por etapa (Caché VAE → Caché Texto → Entrenamiento).
- Botón para limpiar caché DiT y parada segura de subprocesos.

**CivitAI**
Input de texto que acepta ID o URL completa. Muestra nombre, tipo, categoría detectada y lista de versiones con tamaño. Un clic inicia la descarga al directorio correcto.

**HuggingFace**
Input de repo (`org/modelo` o URL). Lista ficheros descargables con sus tamaños. Selector de carpeta de destino con sugerencia automática basada en el nombre del repo. Botón de descarga.

**Descargas**
Cola en tiempo real con barra de progreso, MB descargados/total, estado (queued/downloading/done/error) y botón de eliminar. Botón para limpiar completadas.

**Instalados**
Lista de todos los ficheros en `/workspace/models/` agrupados por categoría, con nombre, tamaño y botón de borrar (con confirmación).

**Tokens**
Muestra estado de `HF_TOKEN` y `CIVITAI_TOKEN` — verde si detectados como RunPod Secret, rojo si no. Formulario para sobreescribir en memoria (se pierden al reiniciar el pod).

**Logs**
Visor de logs con selector de servicio (ComfyUI, Forge, Kohya, TensorBoard, Panel). Muestra las últimas 200 líneas con scroll automático al final.

### Polling automático
Llama a `/api/status` cada 3 segundos para mantener actualizado el estado de servicios, GPU, VRAM, disco y descargas activas. La sección de descargas también se refresca cada 3 segundos si está visible.

### Navegación RunPod
La función `getServiceUrl(port)` detecta si se está ejecutando bajo el proxy de RunPod (dominio `*.proxy.runpod.net`) y construye la URL correcta para abrir cada servicio en el navegador.

---

## Jupyter Lab

Arranca en el puerto 8888 sin token ni contraseña (el proxy de RunPod protege el acceso). El directorio raíz es `/workspace`, lo que permite navegar, subir y descargar cualquier fichero del volumen desde el navegador — incluyendo imágenes de entrenamiento, modelos y outputs de Kohya.

---

## Comportamiento en reinicios

| Elemento | Persiste |
|---|---|
| `/workspace/models/` | ✅ Volumen persistente |
| `/workspace/kohya_ss/` | ✅ Volumen persistente |
| `/workspace/musubi-tuner/` | ✅ Volumen persistente |
| `/workspace/ComfyUI/` | ✅ Volumen persistente |
| `/workspace/studio/` | ✅ Volumen persistente |
| Paquetes apt del sistema | ❌ Se reinstalan (rápido) |
| Tokens sobreescritos manualmente | ❌ Solo en memoria |
| Tokens de RunPod Secrets | ✅ Se inyectan automáticamente |

El script detecta los flags `.setup_done` y salta la reinstalación de Kohya y ComfyUI en reinicios posteriores, reduciendo el tiempo de arranque de ~25 minutos a ~2 minutos.
