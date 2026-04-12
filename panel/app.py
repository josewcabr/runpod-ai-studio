#!/usr/bin/env python3
"""
RunPod AI Studio — Control Panel Backend
Flask app que gestiona servicios, descargas y modelos.
"""
import os
import signal
import threading
import subprocess
import uuid
import time
from collections import deque
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='static')

# ── Paths ─────────────────────────────────────────────────────────
WORKSPACE  = Path('/workspace')
MODELS_DIR = WORKSPACE / 'models'
LOGS_DIR   = WORKSPACE / 'logs'
PIDS_DIR   = LOGS_DIR / 'pids'
PIDS_DIR.mkdir(parents=True, exist_ok=True)

# ── Definición de servicios ───────────────────────────────────────
# group 'exclusive' → solo uno puede estar activo en :7860
# group 'independent' → convive con los demás

SERVICES = {
    'a1111': {
        'name': 'Automatic1111',
        'group': 'exclusive',
        'port': 7860,
        'cmd': (
            'source /root/.studio_cuda_env 2>/dev/null || true; '
            'cd /workspace/stable-diffusion-webui && '
            'python3 launch.py '
            '--listen --port 7860 '
            '--api --enable-insecure-extension-access '
            '--cors-allow-origins=* --no-half-vae '
            '--skip-torch-cuda-test'
        ),
        'log': 'a1111.log',
    },
    'forge': {
        'name': 'Forge',
        'group': 'exclusive',
        'port': 7860,
        'cmd': (
            'source /root/.studio_cuda_env 2>/dev/null || true; '
            'cd /workspace/stable-diffusion-webui-forge && '
            'python3 launch.py '
            '--listen --port 7860 '
            '--api --enable-insecure-extension-access '
            '--cors-allow-origins=* '
            '--skip-torch-cuda-test'
        ),
        'log': 'forge.log',
    },
    'kohya': {
        'name': 'Kohya_ss',
        'group': 'exclusive',
        'port': 7860,
        'cmd': (
            'source /root/.studio_cuda_env 2>/dev/null || true; '
            'cd /workspace/kohya_ss && '
            './gui.sh --server_port 7860 --listen=0.0.0.0 --headless'
        ),
        'log': 'kohya.log',
        'tensorboard': True,
    },
    'comfyui': {
        'name': 'ComfyUI',
        'group': 'independent',
        'port': 8188,
        'cmd': (
            'source /root/.studio_cuda_env 2>/dev/null || true; '
            'source /workspace/ComfyUI/venv/bin/activate && '
            'cd /workspace/ComfyUI && '
            'python main.py '
            '--listen 0.0.0.0 --port 8188 --enable-cors-header'
        ),
        'log': 'comfyui.log',
    },
}

TENSORBOARD_CMD = 'tensorboard --logdir=/workspace/logs --port=6006 --host=0.0.0.0'

# Runtime: handles de procesos en memoria
_procs       = {}    # service_id → Popen
_tb_proc     = None
_log_handles = {}    # sid → file handle abierto

_state_lock = threading.Lock()   # protege _procs, _tb_proc y _log_handles
_dl_lock    = threading.Lock()   # protege DOWNLOADS

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
        st = os.statvfs('/workspace')
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
    services = {
        sid: {
            'name':           svc['name'],
            'status':         _status(sid),
            'port':           svc['port'],
            'group':          svc['group'],
            'has_tensorboard': svc.get('tensorboard', False),
        }
        for sid, svc in SERVICES.items()
    }
    with _dl_lock:
        active_dl = sum(1 for d in DOWNLOADS.values() if d['status'] == 'downloading')
    return jsonify({
        **sys,
        'services':        services,
        'tensorboard':     _tb_status(),
        'tokens': {
            'HF_TOKEN':      bool(os.environ.get('HF_TOKEN')),
            'CIVITAI_TOKEN': bool(os.environ.get('CIVITAI_TOKEN')),
        },
        'active_downloads': active_dl,
    })

# ── Servicios ─────────────────────────────────────────────────────
@app.route('/api/services/<sid>/start', methods=['POST'])
def api_start(sid):
    if sid not in SERVICES:
        return jsonify({'error': 'Servicio desconocido'}), 404
    if _status(sid) == 'running':
        return jsonify({'error': f'{SERVICES[sid]["name"]} ya está corriendo'}), 400
    threading.Thread(target=_start, args=(sid,), daemon=True).start()
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
    known = list(SERVICES.keys()) + ['tensorboard', 'panel']
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

    # Aceptar URL completa o solo el ID numérico
    m = re.search(r'/models/(\d+)', raw)
    model_id = m.group(1) if m else re.sub(r'\D', '', raw)
    if not model_id:
        return jsonify({'error': 'ID de modelo inválido'}), 400

    headers = {}
    token = os.environ.get('CIVITAI_TOKEN', '')
    if token:
        headers['Authorization'] = f'Bearer {token}'

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
            return jsonify({'error': f'Modelo {model_id} no encontrado en CivitAI'}), 404
        return jsonify({'error': f'Error HTTP {code}'}), code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/civitai/download', methods=['POST'])
def api_civitai_download():
    data = request.json or {}
    url      = data.get('downloadUrl', '')
    filename = data.get('filename', 'model.safetensors')
    category = data.get('category', 'checkpoints')

    if not url:
        return jsonify({'error': 'downloadUrl requerido'}), 400

    headers = {}
    token = os.environ.get('CIVITAI_TOKEN', '')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    job_id = _new_job('civitai', filename, category)
    dest   = MODELS_DIR / category / filename

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
    repo     = data.get('repo', '').strip()
    filename = data.get('filename', '').strip()
    category = data.get('category', 'checkpoints')

    if not repo or not filename:
        return jsonify({'error': 'repo y filename requeridos'}), 400

    url = f'https://huggingface.co/{repo}/resolve/main/{filename}'
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
    for key in ('HF_TOKEN', 'CIVITAI_TOKEN'):
        val = data.get(key, '').strip()
        if val:
            os.environ[key] = val
    return jsonify({'ok': True})

# ── Entrypoint ────────────────────────────────────────────────────
if __name__ == '__main__':
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print('🎛️  RunPod AI Studio — Control Panel en http://0.0.0.0:3000')
    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)
