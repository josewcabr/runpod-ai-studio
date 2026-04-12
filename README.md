# RunPod AI Studio

Panel de control unificado para entornos de generación y entrenamiento de imágenes IA en RunPod. Gestiona múltiples herramientas (ComfyUI, Automatic1111, Forge, Kohya_ss) desde una interfaz web centralizada, con descarga de modelos integrada desde CivitAI y HuggingFace.

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
| 7860 | Slot exclusivo: A1111 / Forge / Kohya_ss |
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
│   └── static/
│       └── index.html                # Frontend SPA
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

### Sección 3 — Kohya_ss
- Clona `https://github.com/bmaltais/kohya_ss.git` con submódulos
- Instala `python3-tk` antes del setup de Kohya (requerido por su script)
- Ejecuta `./setup.sh -n -p -r -s -u` — el setup oficial de Kohya que instala `torch 2.5.0+cu124`, `xformers 0.0.28.post2` y todas las dependencias en un entorno virtual propio
- Post-fix: actualiza `bitsandbytes` e instala `torchaudio==2.5.0+cu124`

### Sección 4 — ComfyUI
- Clona `https://github.com/comfyanonymous/ComfyUI.git`
- Instala dependencias con `pip install -r requirements.txt`
- Instala ComfyUI Manager desde `https://github.com/ltdrdata/ComfyUI-Manager.git`

### Sección 5 — Automatic1111
- Clona `https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`
- No instala dependencias — A1111 las gestiona en su primer arranque automáticamente

### Sección 6 — Forge
- Clona `https://github.com/lllyasviel/stable-diffusion-webui-forge.git`
- Igual que A1111, se auto-instala en el primer arranque

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
└── hypernetworks/
```

Crea symlinks desde las carpetas que espera cada herramienta:

| Herramienta | Carpeta propia | Apunta a |
|---|---|---|
| ComfyUI | `ComfyUI/models/checkpoints` | `/workspace/models/checkpoints` |
| ComfyUI | `ComfyUI/models/loras` | `/workspace/models/loras` |
| ComfyUI | `ComfyUI/models/upscale_models` | `/workspace/models/upscalers` |
| A1111/Forge | `models/Stable-diffusion` | `/workspace/models/checkpoints` |
| A1111/Forge | `models/Lora` | `/workspace/models/loras` |
| A1111/Forge | `models/VAE` | `/workspace/models/vae` |
| A1111/Forge | `models/ControlNet` | `/workspace/models/controlnet` |
| A1111/Forge | `models/ESRGAN` | `/workspace/models/upscalers` |

Un modelo descargado una sola vez es visible automáticamente en todas las herramientas.

### Sección 8 — Arranque de servicios con tmux
Crea sesión tmux `studio` con 3 ventanas permanentes:

| Ventana tmux | Proceso | Puerto |
|---|---|---|
| `panel` | `python app.py` | 3000 |
| `jupyter` | `jupyter lab` | 8888 |
| `comfyui` | `python main.py --listen 0.0.0.0 --enable-cors-header` | 8188 |

A1111, Forge y Kohya se lanzan desde el panel bajo demanda (slot exclusivo puerto 7860).

---

## panel/app.py — Backend Flask

API REST que gestiona procesos, descargas y modelos. Corre en el Python del sistema (no en venv).

### Definición de servicios

```python
SERVICES = {
    'a1111':   { 'group': 'exclusive', 'port': 7860, ... },
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
| A1111 | `--listen --port 7860 --api --cors-allow-origins=* --no-half-vae` |
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
| GET | `/api/civitai/model?id=<id>` | Info de modelo por ID o URL de CivitAI |
| POST | `/api/civitai/download` | Iniciar descarga desde CivitAI |
| GET | `/api/hf/model?repo=<repo>` | Listar ficheros descargables de un repo HF |
| POST | `/api/hf/download` | Iniciar descarga desde HuggingFace |
| GET | `/api/downloads` | Estado de todas las descargas |
| DELETE | `/api/downloads/<job_id>` | Eliminar job de la cola |
| GET | `/api/models` | Modelos instalados en `/workspace/models/` |
| POST | `/api/models/delete` | Borrar un modelo (con protección de path traversal) |
| POST | `/api/tokens` | Sobreescribir tokens en memoria |

### Descargas
Las descargas corren en threads en background usando `requests` con streaming por chunks de 2 MB. El progreso se calcula a partir del header `Content-Length`. Los tokens (`HF_TOKEN`, `CIVITAI_TOKEN`) se leen de variables de entorno (RunPod Secrets) y se pasan como headers de autorización.

### CivitAI
Acepta ID numérico o URL completa. La API `https://civitai.com/api/v1/models/<id>` devuelve el tipo del modelo (`Checkpoint`, `LORA`, `VAE`, etc.) que se mapea automáticamente a la carpeta correcta en `/workspace/models/`. Si el modelo tiene múltiples versiones, el frontend las lista para que el usuario seleccione.

### HuggingFace
Acepta `org/repo` o URL completa. La API `https://huggingface.co/api/models/<repo>` devuelve la lista de ficheros del repositorio. El backend filtra solo extensiones descargables (`.safetensors`, `.ckpt`, `.pt`, `.bin`, `.gguf`, `.pth`).

---

## panel/static/index.html — Frontend SPA

Aplicación de página única en HTML/CSS/JS puro, sin dependencias externas. Se sirve directamente desde Flask.

### Secciones de la UI

**Dashboard**
Vista rápida con tarjetas de estado de todos los servicios activos y descargas en curso. Se actualiza automáticamente cada 3 segundos mediante polling.

**Lanzador**
- Slot exclusivo (:7860): cards de A1111, Forge y Kohya con botones start/stop/restart. Muestra aviso de qué servicio se parará al lanzar otro.
- Servicios independientes: ComfyUI con botones open/restart/stop.
- TensorBoard: estado y enlace al puerto 6006, visible solo cuando Kohya está en el slot.

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
Visor de logs con selector de servicio (ComfyUI, A1111, Forge, Kohya, TensorBoard, Panel). Muestra las últimas 200 líneas con scroll automático al final.

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
| `/workspace/ComfyUI/` | ✅ Volumen persistente |
| `/workspace/studio/` | ✅ Volumen persistente |
| Paquetes apt del sistema | ❌ Se reinstalan (rápido) |
| Tokens sobreescritos manualmente | ❌ Solo en memoria |
| Tokens de RunPod Secrets | ✅ Se inyectan automáticamente |

El script detecta los flags `.setup_done` y salta la reinstalación de Kohya y ComfyUI en reinicios posteriores, reduciendo el tiempo de arranque de ~25 minutos a ~2 minutos.
