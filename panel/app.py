#!/usr/bin/env python3
"""
AI Studio — Control Panel Backend
Flask app que gestiona servicios, descargas y modelos.
"""
import io
import json
import os
import re
import signal
import threading
import subprocess
import uuid
import time
import zipfile
from collections import deque
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 * 1024  # 20 GB — safetensors pueden pesar mucho

# ── Paths ─────────────────────────────────────────────────────────
WORKSPACE  = Path(os.environ.get('WORKSPACE', '/workspace'))
_WS        = str(WORKSPACE)   # versión string para interpolación en comandos shell
MODELS_DIR = WORKSPACE / 'models'
LOGS_DIR   = WORKSPACE / 'logs'
PIDS_DIR   = LOGS_DIR / 'pids'
PIDS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR   = WORKSPACE / 'training' / 'config'
COMFY_OUTPUT = WORKSPACE / 'ComfyUI' / 'output'

CAPTION_VENV    = WORKSPACE / 'training' / 'venv-caption'
CAPTION_SCRIPTS = Path(__file__).parent / "scripts"
CAPTION_LOG     = LOGS_DIR / 'caption.log'
CAPTION_INST_LOG = LOGS_DIR / 'caption-install.log'
TOKENS_FILE     = WORKSPACE / '.studio_tokens'

# ── Musubi-Tuner (Krea 2) Paths ───────────────────────────────────
MUSUBI_DIR        = WORKSPACE / 'musubi-tuner'
MUSUBI_VENV       = MUSUBI_DIR / 'venv'
KREA2_CACHE_DIR   = WORKSPACE / 'training' / 'cache_krea2'
KREA2_CONFIG_DIR  = WORKSPACE / 'training' / 'config'
KREA2_TRAIN_LOG   = LOGS_DIR / 'krea2_training.log'
KREA2_INST_LOG    = LOGS_DIR / 'musubi-install.log'

# ── Gestión de tokens persistentes ───────────────────────────────
# Fuentes de tokens en orden de prioridad:
#   1. Variable de entorno del launcher (RunPod Secrets / vast.ai Env Vars / inline export)
#   2. Archivo $WORKSPACE/.studio_tokens (guardado desde el panel o por setup.sh)
# El archivo persiste entre reinicios del panel y del contenedor.

_TOKEN_KEYS   = ('HF_TOKEN', 'CIVITAI_TOKEN')
_token_sources: dict[str, str] = {}   # key → 'env' | 'file' | ''

def _load_tokens_file():
    """Carga tokens del archivo al entorno si no están ya como variables de entorno."""
    if not TOKENS_FILE.exists():
        return
    try:
        for line in TOKENS_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip()
            if key in _TOKEN_KEYS and val and not os.environ.get(key):
                os.environ[key] = val
                _token_sources[key] = 'file'
    except Exception as e:
        print(f'[tokens] Error al cargar {TOKENS_FILE}: {e}')

def _save_tokens_file():
    """Escribe los tokens actuales del entorno al archivo."""
    try:
        lines = [f'{k}={os.environ[k]}' for k in _TOKEN_KEYS if os.environ.get(k)]
        TOKENS_FILE.write_text('\n'.join(lines) + ('\n' if lines else ''))
        TOKENS_FILE.chmod(0o600)
    except Exception as e:
        print(f'[tokens] Error al guardar {TOKENS_FILE}: {e}')

# Inicialización al arrancar el panel:
# 1. Marcar como 'env' los tokens ya presentes en el entorno o inyectados por RunPod Secrets
for _k in _TOKEN_KEYS:
    if not os.environ.get(_k) and os.environ.get(f'RUNPOD_SECRET_{_k}'):
        os.environ[_k] = os.environ[f'RUNPOD_SECRET_{_k}']
    if os.environ.get(_k):
        _token_sources[_k] = 'env'
# 2. Completar con el archivo para los que falten
_load_tokens_file()

def _get_civitai_token() -> str:
    return os.environ.get('CIVITAI_TOKEN') or os.environ.get('RUNPOD_SECRET_CIVITAI_TOKEN') or ''

def _get_hf_token() -> str:
    return os.environ.get('HF_TOKEN') or os.environ.get('RUNPOD_SECRET_HF_TOKEN') or ''

# ── Definición de servicios ───────────────────────────────────────
# group 'exclusive' → solo uno puede estar activo en :7860
# group 'independent' → convive con los demás

SERVICES = {

    'forge': {
        'name': 'Forge',
        'group': 'exclusive',
        'port': 7860,
        'cmd': (
            f'source /root/.studio_cuda_env 2>/dev/null || true; '
            f'cd {_WS}/stable-diffusion-webui-forge && '
            f'python3 launch.py '
            f'--listen --port 7860 '
            f'--api --enable-insecure-extension-access '
            f'--cors-allow-origins=* '
            f'--skip-torch-cuda-test'
        ),
        'log': 'forge.log',
    },
    'kohya': {
        'name': 'Kohya_ss',
        'group': 'exclusive',
        'port': 7860,
        'cmd': (
            f'source /root/.studio_cuda_env 2>/dev/null || true; '
            f'[ -f {_WS}/kohya_ss/venv/bin/activate ] && source {_WS}/kohya_ss/venv/bin/activate || true; '
            f'cd {_WS}/kohya_ss && '
            f'./gui.sh --server_port 7860 --listen=0.0.0.0 --headless --noverify'
        ),
        'log': 'kohya.log',
        'tensorboard': True,
    },
    'comfyui': {
        'name': 'ComfyUI',
        'group': 'independent',
        'port': 8188,
        'cmd': (
            f'source /root/.studio_cuda_env 2>/dev/null || true; '
            f'[ -f {_WS}/ComfyUI/venv/bin/activate ] && source {_WS}/ComfyUI/venv/bin/activate || true; '
            f'cd {_WS}/ComfyUI && '
            f'python main.py '
            f'--listen 0.0.0.0 --port 8188 --enable-cors-header'
        ),
        'log': 'comfyui.log',
    },
}

TENSORBOARD_CMD = f'tensorboard --logdir={_WS}/logs --port=6006 --host=0.0.0.0'

FORGE_DIR = WORKSPACE / 'stable-diffusion-webui-forge'

# Runtime: handles de procesos en memoria
_procs       = {}    # service_id → Popen
_tb_proc     = None
_log_handles = {}    # sid → file handle abierto

_state_lock      = threading.Lock()   # protege _procs, _tb_proc y _log_handles
_dl_lock         = threading.Lock()   # protege DOWNLOADS
_forge_inst_lock = threading.Lock()   # protege _forge_installing
_forge_installing = False

_caption_proc       = None   # running caption script process
_caption_log_fh     = None   # log file handle for caption output
_caption_lock       = threading.Lock()
_caption_inst_proc  = None   # venv install process
_caption_inst_log   = None   # log file handle for install output

# Krea 2 Training runtime
_krea2_proc         = None   # running krea2 training runner process
_krea2_log_fh       = None   # log file handle for krea2 output
_krea2_lock         = threading.Lock()
_krea2_inst_proc    = None   # musubi-tuner install process
_krea2_inst_log     = None   # log file handle for musubi install output

def _forge_installed() -> bool:
    return FORGE_DIR.exists()

def _install_forge():
    """Clona Forge y crea symlinks de modelos. Corre en hilo de fondo."""
    global _forge_installing
    WEBUI_MAP = {
        'Stable-diffusion': 'checkpoints',
        'Lora':             'loras',
        'VAE':              'vae',
        'embeddings':       'embeddings',
        'ControlNet':       'controlnet',
        'ESRGAN':           'upscalers',
        'Hypernetwork':     'hypernetworks',
    }
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1',
             'https://github.com/lllyasviel/stable-diffusion-webui-forge.git'],
            cwd=str(WORKSPACE),
            check=True,
        )
        models_dir = FORGE_DIR / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        for dst, src in WEBUI_MAP.items():
            link = models_dir / dst
            if link.is_symlink() or link.is_file():
                link.unlink()
            elif link.exists():
                import shutil; shutil.rmtree(str(link))
            link.symlink_to(MODELS_DIR / src)
        print('[install_forge] Forge instalado y symlinks creados')
    except Exception as e:
        print(f'[install_forge] error: {e}')
    finally:
        with _forge_inst_lock:
            _forge_installing = False

# ── PID helpers ───────────────────────────────────────────────────
def _pid_file(sid: str) -> Path:
    return PIDS_DIR / f'{sid}.pid'

def _write_pid(sid: str, pid: int):
    _pid_file(sid).write_text(str(pid))

def _read_pid(sid: str) -> int | None:
    f = _pid_file(sid)
    try:
        return int(f.read_text().strip()) if f.exists() else None
    except Exception:
        return None

def _clear_pid(sid: str):
    f = _pid_file(sid)
    if f.exists():
        f.unlink(missing_ok=True)

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def _is_zombie(pid: int) -> bool:
    """Devuelve True si el proceso está en estado zombie (Z)."""
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        for line in status.splitlines():
            if line.startswith('State:'):
                return line.split()[1] == 'Z'
        return False
    except Exception:
        return False

def _kill(pid: int, timeout: float = 6.0):
    """Termina el process group del PID dado."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.2)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # El proceso ya terminó — esperado
    except Exception as e:
        print(f'[_kill] advertencia pid={pid}: {e}')

# ── Estado de servicio ────────────────────────────────────────────
def _status(sid: str) -> str:
    """'running' | 'stopped'"""
    with _state_lock:
        proc = _procs.get(sid)
    if proc:
        if proc.poll() is None:
            if _is_zombie(proc.pid):
                proc.wait()
                return 'stopped'
            return 'running'
    pid = _read_pid(sid)
    if pid and _pid_alive(pid) and not _is_zombie(pid):
        return 'running'
    return 'stopped'

def _tb_status() -> str:
    with _state_lock:
        tb = _tb_proc
    if tb and tb.poll() is None:
        if _is_zombie(tb.pid):
            tb.wait()
            return 'stopped'
        return 'running'
    pid = _read_pid('tensorboard')
    if pid and _pid_alive(pid) and not _is_zombie(pid):
        return 'running'
    return 'stopped'

# ── Lógica de arranque / parada ───────────────────────────────────
def _start(sid: str):
    global _tb_proc
    svc = SERVICES[sid]

    # Slot exclusivo: para cualquier otro servicio en :7860
    if svc['group'] == 'exclusive':
        for other_sid, other_svc in SERVICES.items():
            if other_svc['group'] == 'exclusive' and _status(other_sid) == 'running':
                _stop(other_sid)
        time.sleep(1.5)  # dar tiempo a que libere el puerto

    log_path = LOGS_DIR / svc['log']
    log_fh   = open(log_path, 'a', buffering=1)

    proc = subprocess.Popen(
        ['bash', '-c', svc['cmd']],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    with _state_lock:
        _procs[sid]       = proc
        _log_handles[sid] = log_fh
    _write_pid(sid, proc.pid)

    # TensorBoard se arranca junto con Kohya
    if svc.get('tensorboard'):
        time.sleep(2)
        tb_log  = open(LOGS_DIR / 'tensorboard.log', 'a', buffering=1)
        tb_proc = subprocess.Popen(
            ['bash', '-c', TENSORBOARD_CMD],
            stdout=tb_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )
        with _state_lock:
            _tb_proc                    = tb_proc
            _log_handles['tensorboard'] = tb_log
        _write_pid('tensorboard', tb_proc.pid)

def _stop(sid: str):
    global _tb_proc

    with _state_lock:
        proc = _procs.pop(sid, None)
        fh   = _log_handles.pop(sid, None)

    if proc and proc.poll() is None:
        _kill(proc.pid)
    if fh:
        try:
            fh.close()
        except Exception:
            pass

    # Matar via PID file (sobrevive a reinicios del panel)
    pid = _read_pid(sid)
    if pid:
        _kill(pid)
    _clear_pid(sid)

    # TensorBoard sigue al estado de Kohya
    if SERVICES[sid].get('tensorboard'):
        with _state_lock:
            tb       = _tb_proc
            _tb_proc = None
            tb_fh    = _log_handles.pop('tensorboard', None)
        if tb and tb.poll() is None:
            _kill(tb.pid)
        if tb_fh:
            try:
                tb_fh.close()
            except Exception:
                pass
        tb_pid = _read_pid('tensorboard')
        if tb_pid:
            _kill(tb_pid)
        _clear_pid('tensorboard')

# ── Downloads ─────────────────────────────────────────────────────
DOWNLOADS: dict = {}  # job_id → estado

CATEGORY_MAP = {
    'Checkpoint':       'checkpoints',
    'LORA':             'loras',
    'LyCORIS':          'loras',
    'LoCon':            'loras',
    'DoRA':             'loras',
    'TextualInversion': 'embeddings',
    'Controlnet':       'controlnet',
    'Upscaler':         'upscalers',
    'VAE':              'vae',
    'Hypernetwork':     'hypernetworks',
}

VALID_CATEGORIES = set(CATEGORY_MAP.values())
VALID_CATEGORIES.update({
    'audio_encoders', 'clip_vision', 'diffusion_models', 'gligen',
    'latent_upscale_models', 'photomaker', 'style_models', 'text_encoders',
})

def _human(n: int) -> str:
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} TB'

def _download_worker(job_id: str, url: str, dest: Path, headers: dict):
    import requests
    try:
        DOWNLOADS[job_id]['status'] = 'downloading'
        r = requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True)
        r.raise_for_status()

        # Resolver nombre de archivo automáticamente si es genérico
        if dest.name in ('auto', 'model.safetensors', ''):
            cd = r.headers.get('content-disposition', '')
            fn_match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\r\n]+)', cd)
            if fn_match:
                extracted = Path(fn_match.group(1).strip()).name
                if extracted:
                    dest = dest.parent / extracted
                    DOWNLOADS[job_id]['filename'] = extracted
            elif dest.name in ('auto', ''):
                url_clean = r.url.split('?')[0]
                url_name = Path(url_clean).name
                if url_name and '.' in url_name:
                    dest = dest.parent / url_name
                    DOWNLOADS[job_id]['filename'] = url_name

        total = int(r.headers.get('content-length', 0))
        DOWNLOADS[job_id]['total']     = total
        DOWNLOADS[job_id]['total_str'] = _human(total) if total else '?'

        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):  # 2 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    DOWNLOADS[job_id]['downloaded']     = downloaded
                    DOWNLOADS[job_id]['downloaded_str'] = _human(downloaded)
                    DOWNLOADS[job_id]['progress'] = (
                        round(downloaded / total * 100, 1) if total else 0
                    )

        DOWNLOADS[job_id]['status']   = 'done'
        DOWNLOADS[job_id]['progress'] = 100

    except Exception as e:
        DOWNLOADS[job_id]['status'] = 'error'
        DOWNLOADS[job_id]['error']  = str(e)
        dest.unlink(missing_ok=True)

def _new_job(source: str, filename: str, category: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    with _dl_lock:
        DOWNLOADS[job_id] = {
            'source':         source,
            'filename':       filename,
            'category':       category,
            'status':         'queued',
            'progress':       0,
            'downloaded':     0,
            'downloaded_str': '0 B',
            'total':          0,
            'total_str':      '?',
            'error':          None,
        }
    return job_id

# ── Sistema ───────────────────────────────────────────────────────
def _system_info() -> dict:
    gpu = None
    try:
        raw = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            text=True, timeout=5
        ).strip()
        gpu = []
        for line in raw.splitlines():
            p = [x.strip() for x in line.split(',')]
            total, used = int(p[2]), int(p[3])
            gpu.append({
                'index':      int(p[0]),
                'name':       p[1],
                'vram_total': total,
                'vram_used':  used,
                'vram_free':  total - used,
                'temp':       int(p[4]),
                'util':       int(p[5]),
            })
    except Exception:
        pass

    try:
        st = os.statvfs(_WS)
        disk = {
            'total_gb': round(st.f_blocks * st.f_frsize / 1e9, 1),
            'free_gb':  round(st.f_bavail * st.f_frsize / 1e9, 1),
        }
    except OSError:
        disk = {'total_gb': 0, 'free_gb': 0}

    return {'gpu': gpu, 'disk': disk}

# ═══════════════════════════════════════════════════════════════════
# RUTAS FLASK
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# ── Estado global ─────────────────────────────────────────────────
@app.route('/api/status')
def api_status():
    sys = _system_info()
    services = {}
    for sid, svc in SERVICES.items():
        entry = {
            'name':            svc['name'],
            'status':          _status(sid),
            'port':            svc['port'],
            'group':           svc['group'],
            'has_tensorboard': svc.get('tensorboard', False),
        }
        if sid == 'forge':
            with _forge_inst_lock:
                installing = _forge_installing
            entry['installed']  = _forge_installed()
            entry['installing'] = installing
        services[sid] = entry
    with _dl_lock:
        active_dl = sum(1 for d in DOWNLOADS.values() if d['status'] == 'downloading')
    return jsonify({
        **sys,
        'services':        services,
        'tensorboard':     _tb_status(),
        'tokens': {k: _token_sources.get(k, '') for k in _TOKEN_KEYS},
        'active_downloads': active_dl,
    })

# ── Servicios ─────────────────────────────────────────────────────
@app.route('/api/services/<sid>/start', methods=['POST'])
def api_start(sid):
    if sid not in SERVICES:
        return jsonify({'error': 'Servicio desconocido'}), 404
    if sid == 'forge' and not _forge_installed():
        return jsonify({'error': 'Forge no está instalado'}), 400
    if _status(sid) == 'running':
        return jsonify({'error': f'{SERVICES[sid]["name"]} ya está corriendo'}), 400
    threading.Thread(target=_start, args=(sid,), daemon=True).start()
    return jsonify({'ok': True})

@app.route('/api/services/forge/install', methods=['POST'])
def api_forge_install():
    global _forge_installing
    if _forge_installed():
        return jsonify({'error': 'Forge ya está instalado'}), 400
    with _forge_inst_lock:
        if _forge_installing:
            return jsonify({'error': 'Instalación ya en curso'}), 400
        _forge_installing = True
    threading.Thread(target=_install_forge, daemon=True).start()
    return jsonify({'ok': True})

@app.route('/api/services/<sid>/stop', methods=['POST'])
def api_stop(sid):
    if sid not in SERVICES:
        return jsonify({'error': 'Servicio desconocido'}), 404
    threading.Thread(target=_stop, args=(sid,), daemon=True).start()
    return jsonify({'ok': True})

@app.route('/api/services/<sid>/restart', methods=['POST'])
def api_restart(sid):
    if sid not in SERVICES:
        return jsonify({'error': 'Servicio desconocido'}), 404
    def _do():
        _stop(sid)
        time.sleep(2)
        _start(sid)
    threading.Thread(target=_do, daemon=True).start()
    return jsonify({'ok': True})

@app.route('/api/services/<sid>/log')
def api_log(sid):
    known = list(SERVICES.keys()) + ['tensorboard', 'panel', 'caption', 'caption-install', 'krea2_training', 'musubi-install']
    if sid not in known:
        return jsonify({'error': 'Servicio desconocido'}), 404
    log_name = SERVICES[sid]['log'] if sid in SERVICES else f'{sid}.log'
    log_path = LOGS_DIR / log_name
    n = min(int(request.args.get('lines', 150)), 500)
    try:
        with open(log_path, errors='replace') as f:
            lines = [line.rstrip('\n') for line in deque(f, maxlen=n)]
        return jsonify({'lines': lines, 'total': len(lines)})
    except FileNotFoundError:
        return jsonify({'lines': [], 'total': 0})

# ── CivitAI ───────────────────────────────────────────────────────
@app.route('/api/civitai/model')
def api_civitai_model():
    import requests as req, re

    raw = request.args.get('id', '').strip()
    if not raw:
        return jsonify({'error': 'Parámetro id requerido'}), 400

    token = _get_civitai_token()
    headers = {'User-Agent': 'Mozilla/5.0'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    # 1. Detectar si es una URL con modelVersionId, /download/models/, o /model-versions/
    ver_match = re.search(r'(?:modelVersionId=|/download/models/|/model-versions/)(\d+)', raw)
    file_id_match = re.search(r'fileId=(\d+)', raw)
    target_file_id = int(file_id_match.group(1)) if file_id_match else None

    if ver_match:
        ver_id = ver_match.group(1)
        try:
            r = req.get(f'https://civitai.com/api/v1/model-versions/{ver_id}', headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()

            model_info = data.get('model', {})
            model_type = model_info.get('type', 'Checkpoint')
            category = CATEGORY_MAP.get(model_type, 'checkpoints')

            files = data.get('files', [])
            if target_file_id:
                files.sort(key=lambda x: 0 if x.get('id') == target_file_id else 1)

            versions = []
            for f in files:
                fname = f.get('name', 'model.safetensors')
                size_kb = f.get('sizeKB', 0)
                dl_url = data.get('downloadUrl', f'https://civitai.com/api/download/models/{ver_id}')
                if f.get('id'):
                    dl_url = f'https://civitai.com/api/download/models/{ver_id}?fileId={f["id"]}'
                versions.append({
                    'id': data['id'],
                    'name': f"{data.get('name', '')} — {fname}",
                    'baseModel': data.get('baseModel', ''),
                    'downloadUrl': dl_url,
                    'filename': fname,
                    'size_str': _human(int(size_kb * 1024)),
                })

            if not versions:
                dl_url = data.get('downloadUrl', f'https://civitai.com/api/download/models/{ver_id}')
                versions.append({
                    'id': data['id'],
                    'name': data.get('name', 'v1.0'),
                    'baseModel': data.get('baseModel', ''),
                    'downloadUrl': dl_url,
                    'filename': 'model.safetensors',
                    'size_str': '?',
                })

            return jsonify({
                'id': data.get('modelId', data['id']),
                'name': f"{model_info.get('name', 'Modelo')} [{data.get('name', '')}]",
                'type': model_type,
                'category': category,
                'nsfw': model_info.get('nsfw', False),
                'versions': versions,
            })
        except req.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return jsonify({'error': 'Token inválido o modelo restringido (necesitas CIVITAI_TOKEN)'}), 401
            pass
        except Exception:
            pass

    # 2. Aceptar URL de modelo clásica o ID numérico
    m = re.search(r'/models/(\d+)', raw)
    model_id = m.group(1) if m else re.sub(r'\D', '', raw)
    if not model_id:
        return jsonify({'error': 'ID o URL de modelo inválido'}), 400

    try:
        r = req.get(
            f'https://civitai.com/api/v1/models/{model_id}',
            headers=headers, timeout=15
        )
        r.raise_for_status()
        data = r.json()

        model_type = data.get('type', 'Checkpoint')
        versions = []
        for v in data.get('modelVersions', []):
            files = v.get('files', [])
            primary = next((f for f in files if f.get('primary')), files[0] if files else None)
            if not primary:
                continue
            size_kb = primary.get('sizeKB', 0)
            versions.append({
                'id':          v['id'],
                'name':        v['name'],
                'baseModel':   v.get('baseModel', ''),
                'downloadUrl': v.get('downloadUrl', ''),
                'filename':    primary.get('name', 'model.safetensors'),
                'size_str':    _human(int(size_kb * 1024)),
            })

        return jsonify({
            'id':       data['id'],
            'name':     data['name'],
            'type':     model_type,
            'category': CATEGORY_MAP.get(model_type, 'checkpoints'),
            'nsfw':     data.get('nsfw', False),
            'versions': versions,
        })

    except req.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 401:
            return jsonify({'error': 'Token inválido o modelo restringido (necesitas CIVITAI_TOKEN)'}), 401
        if code == 404:
            # Fallback: intentar consultar como model-versions antes de fallar
            try:
                r2 = req.get(f'https://civitai.com/api/v1/model-versions/{model_id}', headers=headers, timeout=15)
                r2.raise_for_status()
                data2 = r2.json()
                model_info2 = data2.get('model', {})
                model_type2 = model_info2.get('type', 'Checkpoint')
                category2 = CATEGORY_MAP.get(model_type2, 'checkpoints')
                versions2 = []
                for f in data2.get('files', []):
                    fname = f.get('name', 'model.safetensors')
                    size_kb = f.get('sizeKB', 0)
                    dl_url = data2.get('downloadUrl', f'https://civitai.com/api/download/models/{data2["id"]}')
                    if f.get('id'):
                        dl_url = f'https://civitai.com/api/download/models/{data2["id"]}?fileId={f["id"]}'
                    versions2.append({
                        'id': data2['id'],
                        'name': f"{data2.get('name', '')} — {fname}",
                        'baseModel': data2.get('baseModel', ''),
                        'downloadUrl': dl_url,
                        'filename': fname,
                        'size_str': _human(int(size_kb * 1024)),
                    })
                return jsonify({
                    'id': data2.get('modelId', data2['id']),
                    'name': f"{model_info2.get('name', 'Modelo')} [{data2.get('name', '')}]",
                    'type': model_type2,
                    'category': category2,
                    'nsfw': model_info2.get('nsfw', False),
                    'versions': versions2,
                })
            except Exception:
                pass
            return jsonify({'error': f'Modelo o versión {model_id} no encontrado en CivitAI'}), 404
        return jsonify({'error': f'Error HTTP {code}'}), code
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/civitai/download', methods=['POST'])
def api_civitai_download():
    data = request.json or {}
    url      = data.get('downloadUrl', '').strip()
    filename = data.get('filename', 'model.safetensors').strip() or 'model.safetensors'
    category = data.get('category', 'checkpoints')

    if not url:
        return jsonify({'error': 'downloadUrl requerido'}), 400

    token = _get_civitai_token()
    headers = {'User-Agent': 'Mozilla/5.0'}

    # Si es CivitAI (civitai.com o civitai.red) y tenemos token, adjuntarlo a la URL si no está presente
    # para que en el redirect a Cloudflare R2 / S3 no haya conflicto con el header Authorization
    if token and ('civitai.com' in url or 'civitai.red' in url):
        if 'token=' not in url:
            sep = '&' if '?' in url else '?'
            url = f"{url}{sep}token={token}"
    elif token:
        headers['Authorization'] = f'Bearer {token}'

    job_id = _new_job('civitai', filename, category)
    dest   = MODELS_DIR / category / filename

    threading.Thread(
        target=_download_worker,
        args=(job_id, url, dest, headers),
        daemon=True,
    ).start()
    return jsonify({'job_id': job_id})


# ── Descarga Directa por URL (Binarios arbitrarios) ───────────────
@app.route('/api/direct/download', methods=['POST'])
def api_direct_download():
    data = request.json or {}
    url = data.get('url', '').strip()
    category = data.get('category', 'checkpoints').strip()
    filename = data.get('filename', '').strip() or 'auto'

    if not url:
        return jsonify({'error': 'URL requerida'}), 400

    if category not in VALID_CATEGORIES:
        category = 'checkpoints'

    headers = {'User-Agent': 'Mozilla/5.0'}

    civ_token = _get_civitai_token()
    hf_token = _get_hf_token()

    if ('civitai.com' in url or 'civitai.red' in url):
        if civ_token and 'token=' not in url:
            sep = '&' if '?' in url else '?'
            url = f"{url}{sep}token={civ_token}"
    elif 'huggingface.co' in url and hf_token:
        headers['Authorization'] = f'Bearer {hf_token}'

    initial_name = filename if filename != 'auto' else 'model.safetensors'
    dest = MODELS_DIR / category / initial_name
    job_id = _new_job('direct_url', initial_name, category)

    threading.Thread(
        target=_download_worker,
        args=(job_id, url, dest, headers),
        daemon=True,
    ).start()
    return jsonify({'job_id': job_id})

    threading.Thread(
        target=_download_worker,
        args=(job_id, url, dest, headers),
        daemon=True,
    ).start()
    return jsonify({'job_id': job_id})

# ── HuggingFace ───────────────────────────────────────────────────
DOWNLOADABLE_EXTS = {'.safetensors', '.ckpt', '.pt', '.bin', '.gguf', '.pth'}

@app.route('/api/hf/model')
def api_hf_model():
    import requests as req, re

    raw = request.args.get('repo', '').strip()
    if not raw:
        return jsonify({'error': 'Parámetro repo requerido'}), 400

    # Aceptar URL completa
    m = re.search(r'huggingface\.co/([^/\s?]+/[^/\s?]+)', raw)
    repo = m.group(1) if m else raw

    headers = {}
    token = os.environ.get('HF_TOKEN', '')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        r = req.get(
            f'https://huggingface.co/api/models/{repo}',
            headers=headers, timeout=15
        )
        r.raise_for_status()
        data = r.json()

        files = []
        for f in data.get('siblings', []):
            fname = f.get('rfilename', '')
            if Path(fname).suffix.lower() in DOWNLOADABLE_EXTS:
                size = f.get('size', 0)
                files.append({
                    'filename': fname,
                    'size_str': _human(size) if size else '?',
                    'size':     size,
                })
        files.sort(key=lambda x: x['filename'])

        return jsonify({
            'repo':    repo,
            'name':    data.get('modelId', repo),
            'private': data.get('private', False),
            'files':   files,
        })

    except req.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 401:
            return jsonify({'error': 'Repositorio privado — necesitas HF_TOKEN'}), 401
        if code == 404:
            return jsonify({'error': f'Repositorio "{repo}" no encontrado'}), 404
        return jsonify({'error': f'Error HTTP {code}'}), code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hf/download', methods=['POST'])
def api_hf_download():
    data     = request.json or {}
    raw      = data.get('repo', '').strip()
    filename = data.get('filename', '').strip()
    category = data.get('category', 'checkpoints')

    if not raw or not filename:
        return jsonify({'error': 'repo y filename requeridos'}), 400

    m    = re.search(r'huggingface\.co/([^/\s?]+/[^/\s?]+)', raw)
    repo = m.group(1) if m else raw

    url = re.sub(r'(?<!:)//', '/', f'https://huggingface.co/{repo}/resolve/main/{filename}')
    headers = {}
    token = os.environ.get('HF_TOKEN', '')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    basename = Path(filename).name
    dest     = MODELS_DIR / category / basename
    job_id   = _new_job('huggingface', basename, category)

    threading.Thread(
        target=_download_worker,
        args=(job_id, url, dest, headers),
        daemon=True,
    ).start()
    return jsonify({'job_id': job_id})

# ── Descargas ─────────────────────────────────────────────────────
@app.route('/api/downloads')
def api_downloads():
    with _dl_lock:
        snapshot = dict(DOWNLOADS)
    return jsonify(snapshot)

@app.route('/api/downloads/<job_id>', methods=['DELETE'])
def api_downloads_delete(job_id):
    with _dl_lock:
        DOWNLOADS.pop(job_id, None)
    return jsonify({'ok': True})

# ── Modelos instalados ────────────────────────────────────────────
@app.route('/api/models')
def api_models():
    if not MODELS_DIR.exists():
        return jsonify({})
    result = {}
    for cat_dir in sorted(MODELS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        files = []
        for f in sorted(cat_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in DOWNLOADABLE_EXTS:
                sz = f.stat().st_size
                files.append({'name': f.name, 'size': sz, 'size_str': _human(sz)})
        result[cat_dir.name] = files
    return jsonify(result)

@app.route('/api/models/delete', methods=['POST'])
def api_models_delete():
    data     = request.json or {}
    category = data.get('category', '')
    filename = data.get('filename', '')
    if not category or not filename:
        return jsonify({'error': 'category y filename requeridos'}), 400

    if category not in VALID_CATEGORIES:
        return jsonify({'error': 'Categoría no válida'}), 400

    path = (MODELS_DIR / category / filename).resolve()
    # Seguridad: impedir path traversal
    try:
        path.relative_to(MODELS_DIR.resolve())
    except ValueError:
        return jsonify({'error': 'Path no permitido'}), 403

    if not path.exists():
        return jsonify({'error': 'Archivo no encontrado'}), 404

    path.unlink()
    return jsonify({'ok': True})

# ── Tokens ────────────────────────────────────────────────────────
@app.route('/api/tokens', methods=['POST'])
def api_tokens():
    data = request.json or {}
    for key in _TOKEN_KEYS:
        val = data.get(key, '').strip()
        if val:
            os.environ[key] = val
            _token_sources[key] = 'file'
        elif data.get(key) == '':
            # String vacío explícito → borrar token
            os.environ.pop(key, None)
            _token_sources.pop(key, None)
    _save_tokens_file()
    return jsonify({'ok': True})

# ── Upload de archivos ────────────────────────────────────────────
UPLOAD_ACCEPT = {
    'loras':       ('.safetensors',),
    'checkpoints': ('.safetensors',),
    'config':      ('.json', '.txt'),
}

@app.route('/api/upload', methods=['POST'])
def api_upload():
    category = request.form.get('category', '')
    if category not in UPLOAD_ACCEPT:
        return jsonify({'error': 'Categoría no válida'}), 400
    if 'file' not in request.files:
        return jsonify({'error': 'Sin archivo'}), 400

    f        = request.files['file']
    filename = secure_filename(f.filename or '')
    if not filename:
        return jsonify({'error': 'Nombre de archivo inválido'}), 400

    exts = UPLOAD_ACCEPT[category]
    if not any(filename.lower().endswith(e) for e in exts):
        return jsonify({'error': f'Solo se permiten archivos {", ".join(exts)}'}), 400

    dest_dir = CONFIG_DIR if category == 'config' else MODELS_DIR / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    f.save(str(dest_dir / filename))
    return jsonify({'ok': True, 'filename': filename})

# ── Configuraciones ────────────────────────────────────────────────
@app.route('/api/configs')
def api_configs_list():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(CONFIG_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ('.json', '.txt'):
            sz = p.stat().st_size
            files.append({'name': p.name, 'size': sz, 'size_str': _human(sz)})
    return jsonify(files)

@app.route('/api/configs/<path:filename>')
def api_config_read(filename):
    filename = secure_filename(filename)
    path     = (CONFIG_DIR / filename).resolve()
    try:
        path.relative_to(CONFIG_DIR.resolve())
    except ValueError:
        return jsonify({'error': 'Path no permitido'}), 403
    if not path.exists():
        return jsonify({'error': 'Archivo no encontrado'}), 404
    try:
        return jsonify({'content': path.read_text(encoding='utf-8', errors='replace'), 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/configs/<path:filename>', methods=['PUT'])
def api_config_write(filename):
    filename = secure_filename(filename)
    path     = (CONFIG_DIR / filename).resolve()
    try:
        path.relative_to(CONFIG_DIR.resolve())
    except ValueError:
        return jsonify({'error': 'Path no permitido'}), 403
    content = (request.json or {}).get('content', '')
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding='utf-8')
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/configs/<path:filename>', methods=['DELETE'])
def api_config_delete(filename):
    filename = secure_filename(filename)
    path     = (CONFIG_DIR / filename).resolve()
    try:
        path.relative_to(CONFIG_DIR.resolve())
    except ValueError:
        return jsonify({'error': 'Path no permitido'}), 403
    if not path.exists():
        return jsonify({'error': 'Archivo no encontrado'}), 404
    path.unlink()
    return jsonify({'ok': True})

# ── Output de ComfyUI ──────────────────────────────────────────────
@app.route('/api/comfy/output')
def api_comfy_output():
    if not COMFY_OUTPUT.exists():
        return jsonify([])
    files = []
    for p in sorted(COMFY_OUTPUT.iterdir(), reverse=True):
        if p.is_file():
            sz = p.stat().st_size
            files.append({'name': p.name, 'size': sz, 'size_str': _human(sz)})
    return jsonify(files)

@app.route('/api/comfy/output/zip', methods=['POST'])
def api_comfy_output_zip():
    data  = request.json or {}
    names = data.get('files', [])
    if not names:
        return jsonify({'error': 'Sin archivos seleccionados'}), 400

    root  = COMFY_OUTPUT.resolve()
    paths = []
    for name in names:
        # Reject path traversal without altering the filename
        if not name or '/' in name or '\\' in name or '..' in name:
            return jsonify({'error': f'Path no permitido: {name}'}), 403
        p = (COMFY_OUTPUT / name).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            return jsonify({'error': f'Path no permitido: {name}'}), 403
        if not p.is_file():
            return jsonify({'error': f'No encontrado: {name}'}), 404
        paths.append(p)

    if not paths:
        return jsonify({'error': 'Sin archivos válidos'}), 400

    zip_name = secure_filename(data.get('zip_filename', 'comfyui_output.zip'))
    if not zip_name.endswith('.zip'):
        zip_name += '.zip'

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, p.name)
    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True, download_name=zip_name)

# ── Caption endpoints ─────────────────────────────────────────────

@app.route('/api/caption/venv-status')
def caption_venv_status():
    installed = (CAPTION_VENV / 'bin' / 'python').exists()
    return jsonify({'installed': installed})


@app.route('/api/caption/install', methods=['POST'])
def caption_install():
    global _caption_inst_proc, _caption_inst_log
    with _caption_lock:
        if _caption_inst_proc and _caption_inst_proc.poll() is None:
            return jsonify({'error': 'Instalación ya en curso'}), 409
        CAPTION_INST_LOG.parent.mkdir(parents=True, exist_ok=True)
        _caption_inst_log = open(str(CAPTION_INST_LOG), 'w', buffering=1)
        cmd = (
            f"python3 -m venv {CAPTION_VENV} && "
            f"{CAPTION_VENV}/bin/pip install --upgrade pip && "
            f"{CAPTION_VENV}/bin/pip install "
            f"\"torch>=2.6.0\" torchvision --index-url https://download.pytorch.org/whl/cu124 && "
            f"{CAPTION_VENV}/bin/pip install "
            f"transformers onnxruntime-gpu huggingface_hub pandas numpy Pillow && "
            f"echo 'INSTALACIÓN COMPLETADA'"
        )
        _caption_inst_proc = subprocess.Popen(
            ['bash', '-c', cmd],
            stdout=_caption_inst_log, stderr=subprocess.STDOUT
        )
    return jsonify({'ok': True})


@app.route('/api/caption/install-log')
def caption_install_log():
    global _caption_inst_proc
    lines_n = min(int(request.args.get('lines', 100)), 500)
    running = bool(_caption_inst_proc and _caption_inst_proc.poll() is None)
    done    = False
    error   = False
    lines   = []
    if CAPTION_INST_LOG.exists():
        all_lines = CAPTION_INST_LOG.read_text(encoding='utf-8', errors='replace').splitlines()
        lines = all_lines[-lines_n:]
        if all_lines:
            done  = 'INSTALACIÓN COMPLETADA' in all_lines[-1]
            error = (_caption_inst_proc is not None and
                     _caption_inst_proc.poll() not in (None, 0))
    return jsonify({'lines': lines, 'running': running, 'done': done, 'error': error})


@app.route('/api/caption/run', methods=['POST'])
def caption_run():
    global _caption_proc, _caption_log_fh
    data   = request.get_json(force=True) or {}
    tool   = data.get('tool', '')
    prefix = data.get('prefix', '').strip()
    if tool not in ('blip', 'wd14'):
        return jsonify({'error': 'tool debe ser blip o wd14'}), 400
    venv_python = CAPTION_VENV / 'bin' / 'python'
    if not venv_python.exists():
        return jsonify({'error': 'Venv no instalado'}), 409
    script = CAPTION_SCRIPTS / f'script_{tool}.py'
    if not script.exists():
        return jsonify({'error': f'Script no encontrado: {script}'}), 500
    with _caption_lock:
        if _caption_proc and _caption_proc.poll() is None:
            return jsonify({'error': 'Ya hay un proceso de captioning en curso'}), 409
        CAPTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        if _caption_log_fh:
            try: _caption_log_fh.close()
            except: pass
        _caption_log_fh = open(str(CAPTION_LOG), 'w', buffering=1)
        cmd = [str(venv_python), str(script), '--prefix', prefix]
        _caption_proc = subprocess.Popen(
            cmd, stdout=_caption_log_fh, stderr=subprocess.STDOUT
        )
    return jsonify({'ok': True})


@app.route('/api/caption/log')
def caption_log():
    global _caption_proc
    lines_n = min(int(request.args.get('lines', 100)), 500)
    running = bool(_caption_proc and _caption_proc.poll() is None)
    lines   = []
    if CAPTION_LOG.exists():
        all_lines = CAPTION_LOG.read_text(encoding='utf-8', errors='replace').splitlines()
        lines = all_lines[-lines_n:]
    return jsonify({'lines': lines, 'running': running})


@app.route('/api/caption/stop', methods=['POST'])
def caption_stop():
    global _caption_proc
    with _caption_lock:
        if _caption_proc and _caption_proc.poll() is None:
            _caption_proc.terminate()
            return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': 'No hay proceso activo'})


# ── Krea 2 / Musubi-Tuner Training Endpoints ──────────────────────

@app.route('/api/train/krea2/venv-status')
def krea2_venv_status():
    installed = (MUSUBI_VENV / 'bin' / 'python').exists() and (MUSUBI_DIR / 'src' / 'musubi_tuner').exists()
    return jsonify({
        'installed': installed,
        'musubi_dir': str(MUSUBI_DIR),
        'exists': MUSUBI_DIR.exists()
    })


@app.route('/api/train/krea2/install', methods=['POST'])
def krea2_install():
    global _krea2_inst_proc, _krea2_inst_log
    with _krea2_lock:
        if _krea2_inst_proc and _krea2_inst_proc.poll() is None:
            return jsonify({'error': 'Instalación de musubi-tuner ya en curso'}), 409
        KREA2_INST_LOG.parent.mkdir(parents=True, exist_ok=True)
        _krea2_inst_log = open(str(KREA2_INST_LOG), 'w', buffering=1)
        cmd = (
            f"cd {WORKSPACE} && "
            f"([ -d musubi-tuner ] || git clone https://github.com/kohya-ss/musubi-tuner.git) && "
            f"cd musubi-tuner && git pull || true && "
            f"python3 -m venv {MUSUBI_VENV} && "
            f"{MUSUBI_VENV}/bin/pip install --upgrade pip wheel setuptools && "
            f"{MUSUBI_VENV}/bin/pip install \"torch>=2.6.0\" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && "
            f"{MUSUBI_VENV}/bin/pip install -e . && "
            f"{MUSUBI_VENV}/bin/pip install bitsandbytes accelerate tomli toml huggingface_hub && "
            f"echo 'INSTALACIÓN MUSUBI COMPLETADA'"
        )
        _krea2_inst_proc = subprocess.Popen(
            ['bash', '-c', cmd],
            stdout=_krea2_inst_log, stderr=subprocess.STDOUT
        )
    return jsonify({'ok': True})


@app.route('/api/train/krea2/install-log')
def krea2_install_log():
    global _krea2_inst_proc
    lines_n = min(int(request.args.get('lines', 100)), 500)
    running = bool(_krea2_inst_proc and _krea2_inst_proc.poll() is None)
    done    = False
    error   = False
    lines   = []
    if KREA2_INST_LOG.exists():
        all_lines = KREA2_INST_LOG.read_text(encoding='utf-8', errors='replace').splitlines()
        lines = all_lines[-lines_n:]
        if all_lines:
            done  = any('INSTALACIÓN MUSUBI COMPLETADA' in l for l in all_lines[-10:])
            error = (_krea2_inst_proc is not None and
                     _krea2_inst_proc.poll() not in (None, 0))
    return jsonify({'lines': lines, 'running': running, 'done': done, 'error': error})


@app.route('/api/train/krea2/models')
def krea2_models():
    """Devuelve listas de checkpoints, vae y text encoders disponibles en el sistema."""
    checkpoints = []
    ckpt_dir = MODELS_DIR / 'checkpoints'
    if ckpt_dir.exists():
        checkpoints = [f.name for f in sorted(ckpt_dir.iterdir()) if f.is_file() and f.suffix.lower() in DOWNLOADABLE_EXTS]

    vaes = []
    vae_dir = MODELS_DIR / 'vae'
    if vae_dir.exists():
        vaes = [f.name for f in sorted(vae_dir.iterdir()) if f.is_file() and f.suffix.lower() in DOWNLOADABLE_EXTS]

    text_encoders = []
    for d in [MODELS_DIR / 'text_encoders', MODELS_DIR / 'clip']:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in DOWNLOADABLE_EXTS:
                    text_encoders.append(f.name)

    img_dir = WORKSPACE / 'training' / 'images'
    raw_dir = WORKSPACE / 'training' / 'raw-images'
    has_images = img_dir.exists() and any(img_dir.iterdir())
    has_raw = raw_dir.exists() and any(raw_dir.iterdir())

    active_dir = img_dir if has_images or not has_raw else raw_dir
    img_count = 0
    if active_dir.exists():
        img_count = len([f for f in active_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])

    return jsonify({
        'checkpoints': checkpoints,
        'vaes': vaes,
        'text_encoders': text_encoders,
        'default_image_dir': str(active_dir),
        'image_count': img_count
    })


@app.route('/api/train/krea2/start', methods=['POST'])
def krea2_train_start():
    global _krea2_proc, _krea2_log_fh
    data = request.get_json(force=True) or {}

    venv_python = MUSUBI_VENV / 'bin' / 'python'
    if not venv_python.exists():
        return jsonify({'error': 'Musubi-tuner venv no está instalado'}), 409

    script = Path(__file__).parent / 'scripts' / 'train_krea2.py'
    if not script.exists():
        return jsonify({'error': f'Script runner no encontrado: {script}'}), 500

    with _krea2_lock:
        if _krea2_proc and _krea2_proc.poll() is None:
            return jsonify({'error': 'Ya hay un entrenamiento de Krea 2 en curso'}), 409

        KREA2_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        job_config_file = KREA2_CONFIG_DIR / 'krea2_job.json'
        data['workspace'] = str(WORKSPACE)
        job_config_file.write_text(json.dumps(data, indent=2), encoding='utf-8')

        KREA2_TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        if _krea2_log_fh:
            try: _krea2_log_fh.close()
            except: pass
        _krea2_log_fh = open(str(KREA2_TRAIN_LOG), 'w', buffering=1)

        cmd = [str(venv_python), str(script), '--config', str(job_config_file)]
        _krea2_proc = subprocess.Popen(
            cmd,
            stdout=_krea2_log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )
        _write_pid('krea2_training', _krea2_proc.pid)

    return jsonify({'ok': True, 'pid': _krea2_proc.pid})


@app.route('/api/train/krea2/status')
def krea2_train_status():
    global _krea2_proc
    running = False
    with _krea2_lock:
        if _krea2_proc and _krea2_proc.poll() is None:
            running = True
    if not running:
        pid = _read_pid('krea2_training')
        if pid and _pid_alive(pid) and not _is_zombie(pid):
            running = True

    stage = 'idle'
    if running and KREA2_TRAIN_LOG.exists():
        try:
            lines = KREA2_TRAIN_LOG.read_text(encoding='utf-8', errors='replace').splitlines()[-25:]
            for l in reversed(lines):
                if 'PASO 3/3' in l or 'Entrenando red' in l or 'epoch' in l.lower():
                    stage = 'training'
                    break
                elif 'PASO 2/3' in l or 'embeddings de texto' in l:
                    stage = 'caching_text'
                    break
                elif 'PASO 1/3' in l or 'latentes' in l:
                    stage = 'caching_latents'
                    break
        except Exception:
            pass

    return jsonify({'running': running, 'stage': stage})


@app.route('/api/train/krea2/log')
def krea2_train_log():
    lines_n = min(int(request.args.get('lines', 150)), 500)
    lines = []
    if KREA2_TRAIN_LOG.exists():
        all_lines = KREA2_TRAIN_LOG.read_text(encoding='utf-8', errors='replace').splitlines()
        lines = all_lines[-lines_n:]
    running = False
    with _krea2_lock:
        if _krea2_proc and _krea2_proc.poll() is None:
            running = True
    return jsonify({'lines': lines, 'running': running})


@app.route('/api/train/krea2/stop', methods=['POST'])
def krea2_train_stop():
    global _krea2_proc
    with _krea2_lock:
        if _krea2_proc and _krea2_proc.poll() is None:
            _kill(_krea2_proc.pid)
            _krea2_proc = None
    pid = _read_pid('krea2_training')
    if pid:
        _kill(pid)
        _clear_pid('krea2_training')
    return jsonify({'ok': True})


@app.route('/api/train/krea2/clear-cache', methods=['POST'])
def krea2_clear_cache():
    count = 0
    if KREA2_CACHE_DIR.exists():
        import shutil
        for item in KREA2_CACHE_DIR.iterdir():
            if item.is_file():
                item.unlink(missing_ok=True)
                count += 1
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                count += 1
    return jsonify({'ok': True, 'cleared_items': count})


# ── Detección de plataforma ───────────────────────────────────────
def _detect_platform() -> dict:
    """
    Detecta la plataforma de nube y devuelve la info necesaria para
    que el frontend construya URLs de servicios correctamente.

    Plataformas soportadas:
    - runpod   → proxy *.proxy.runpod.net  (RUNPOD_POD_ID presente)
    - vastai   → proxy *.jupyter.vast.ai   (VAST_CONTAINERLABEL o VASTAI_* presente)
    - generic  → acceso directo por IP/hostname
    """
    # RunPod: inyecta RUNPOD_POD_ID en todos los pods
    if os.environ.get('RUNPOD_POD_ID'):
        return {
            'name': 'runpod',
            'pod_id': os.environ['RUNPOD_POD_ID'],
            # El proxy de RunPod sigue el patrón: <pod_id>-<port>.proxy.runpod.net
            'proxy_template': 'https://{pod_id}-{port}.proxy.runpod.net',
        }

    # vast.ai: inyecta VAST_CONTAINERLABEL y opcionalmente VAST_TCP_PORT_<N>
    if os.environ.get('VAST_CONTAINERLABEL'):
        return {
            'name': 'vastai',
            'container_label': os.environ['VAST_CONTAINERLABEL'],
            # vast.ai no tiene proxy centralizado; los puertos se mapean
            # al host directamente. El frontend usará window.location.hostname.
            'proxy_template': None,
        }

    return {'name': 'generic', 'proxy_template': None}


@app.route('/api/platform')
def api_platform():
    return jsonify(_detect_platform())


# ── Entrypoint ────────────────────────────────────────────────────
if __name__ == '__main__':
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    platform = _detect_platform()['name']
    print(f'🎛️  AI Studio — Control Panel en http://0.0.0.0:3000  (plataforma: {platform})')
    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)
