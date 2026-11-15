"""NoteForge draft API: an editor uploads audio and the charter's input, and gets back notes it can edit.

The frozen model and every heavy step run in child processes (ops.py), one generation at a time."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .schemas import ContinueRequest, DraftRequest, Mode, RevisionRequest

API_VERSION = '1.0'
OPS = Path(__file__).resolve().parent/'ops.py'
ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r'^[0-9a-f]{8,64}$')
AUDIO_TYPES = {'.mp3', '.ogg', '.wav', '.flac', '.m4a'}
CHUNK = 1 << 20


class ApiError(Exception):
    def __init__(self, status, code, message, field=None):
        super().__init__(message)
        self.status, self.code, self.message, self.field = status, code, message, field


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


class Settings:
    def __init__(self, **over):
        env = os.environ.get
        self.data = Path(over.get('data') or env('NOTEFORGE_DATA', ROOT/'data'))
        self.token = over.get('token', env('NOTEFORGE_API_TOKEN'))
        self.engine_python = str(over.get('engine_python') or env('NOTEFORGE_ENGINE_PYTHON', sys.executable))
        self.engine_root = str(over.get('engine_root') or env('NOTEFORGE_ENGINE_ROOT', ROOT))
        self.model = str(over.get('model') or env('NOTEFORGE_MODEL', ROOT/'models/noteforge-r5.pt'))
        self.plan = str(over.get('plan') or env('NOTEFORGE_PLAN', ROOT/'models/config/r5-inference.json'))
        self.planner_dir = str(over.get('planner_dir') or env('NOTEFORGE_PLANNER_DIR', ROOT/'models/planner_f'))
        self.mert_source = over.get('mert_source', env('NOTEFORGE_MERT_SOURCE'))
        self.mert_python = over.get('mert_python', env('NOTEFORGE_MERT_PYTHON'))  # the feature step can need its own environment
        self.device = over.get('device') or env('NOTEFORGE_DEVICE', 'cuda')
        self.batch_size = int(over.get('batch_size') or env('NOTEFORGE_BATCH_SIZE', 64))
        self.max_wall_s = float(over.get('max_wall_s') or env('NOTEFORGE_MAX_WALL_S', 1800))
        self.max_upload = int(over.get('max_upload') or env('NOTEFORGE_MAX_UPLOAD_MB', 100))*CHUNK
        self.cors = [x for x in str(over.get('cors') or env('NOTEFORGE_CORS_ORIGINS', '')).split(',') if x]
        self.fake = over.get('fake', env('NOTEFORGE_FAKE_ENGINE') == '1')
        self.backend = over.get('backend') or env('NOTEFORGE_BACKEND', 'cuda')  # cuda or m3
        self.fast_weights = over.get('fast_weights') or env('NOTEFORGE_FAST_WEIGHTS', '')
        self.fast_mert = over.get('fast_mert') or env('NOTEFORGE_FAST_MERT', '')
        self.mel_stats = str(over.get('mel_stats') or env('NOTEFORGE_MEL_STATS', ROOT/'models/config/mel-statistics.json'))
        self.warm_idle_s = float(over.get('warm_idle_s') or env('NOTEFORGE_WARM_IDLE_S', 300))  # unload the waiting model after this
        if not self.token:
            raise RuntimeError('NOTEFORGE_API_TOKEN is required')


class Store:
    def __init__(self, settings):
        self.s = settings
        self.audio, self.charts, self.drafts, self.features = (settings.data/x for x in ('audio', 'charts', 'drafts', 'features'))
        for p in (self.audio, self.charts, self.drafts, self.features):
            p.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def audio_path(self, audio_id):
        check_id(audio_id, 'audio_id')
        for p in self.audio.glob(audio_id+'.*'):
            return p
        raise ApiError(404, 'audio_not_found', f'no audio {audio_id}; upload it first', 'audio_id')

    def chart(self, chart_id, field='chart_id'):
        check_id(chart_id, field)
        folder = self.charts/chart_id
        if not (folder/'info.json').is_file():
            raise ApiError(404, 'chart_not_found', f'no chart {chart_id}; upload it first', field)
        return folder, json.loads((folder/'info.json').read_text())

    def draft_dir(self, draft_id):
        check_id(draft_id, 'draft_id')
        folder = self.drafts/draft_id
        if not (folder/'status.json').is_file():
            raise ApiError(404, 'draft_not_found', f'no draft {draft_id}')
        return folder

    def status(self, folder):
        return json.loads((folder/'status.json').read_text())

    def set_status(self, folder, **fields):
        with self.lock:
            status = self.status(folder)
            status.update(fields, updated_at=time.time())
            write_json(folder/'status.json', status)
            return status


def check_id(value, field):
    if not ID.match(value or ''):
        raise ApiError(422, 'bad_id', f'{field} is not a valid id', field)


def op_env(settings):
    env = dict(os.environ, NOTEFORGE_ENGINE_ROOT=settings.engine_root)
    if settings.fake:
        env['NOTEFORGE_FAKE_ENGINE'] = '1'
    return env


def op_error(done, fallback):
    for line in reversed((done.stdout or '').strip().splitlines()):
        try:
            error = json.loads(line).get('error')
        except (ValueError, AttributeError):
            continue
        if error:
            return error
    return dict(code=fallback, message=((done.stderr or '').strip().splitlines() or ['the engine failed without a message'])[-1][:300])


def engine_socket(settings):
    tag = hashlib.sha256(str(settings.data.resolve()).encode()).hexdigest()[:16]
    return f'{tempfile.gettempdir()}/noteforge-{os.getuid()}-{tag}.sock'


def start_engine(settings):
    """Start the engine that stays loaded, unless it is already there. Does not wait for it."""
    path = engine_socket(settings)
    if not _alive(Path(path+'.pid')):  # a socket file alone proves nothing: a killed engine leaves one behind
        with open(settings.data/'engine-server.log', 'a') as log:
            process = subprocess.Popen([settings.engine_python, str(ROOT/'noteforge/m3/engine_server.py'), path, settings.fast_weights, str(settings.warm_idle_s)],
                                       stdout=log, stderr=log, env=op_env(settings), cwd=settings.engine_root, start_new_session=True)
        Path(path+'.pid').write_text(str(process.pid))


def _alive(pid_file):
    try:
        os.kill(int(pid_file.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def stop_engine(settings):
    pid_file = Path(engine_socket(settings)+'.pid')
    try:
        os.kill(int(pid_file.read_text()), signal.SIGTERM)
    except (OSError, ValueError):
        pass


def warm_call(settings, args, timeout):
    """Ask the engine that stays loaded; start it if needed. None means use a one-off process instead."""
    path = engine_socket(settings)
    request = (json.dumps(dict(argv=[str(a) for a in args]))+'\n').encode()
    for attempt in range(80):
        try:
            with socket.socket(socket.AF_UNIX) as conn:
                conn.settimeout(timeout)
                conn.connect(path)
                conn.sendall(request)
                data = b''
                while not data.endswith(b'\n'):
                    chunk = conn.recv(1 << 16)
                    if not chunk:
                        return None
                    data += chunk
            answer = json.loads(data)
            return subprocess.CompletedProcess([], answer['returncode'], answer['stdout'], answer['stderr'])
        except socket.timeout as exc:
            raise ApiError(504, 'engine_timeout', f'{args[0]} took longer than {timeout} s') from exc
        except OSError:
            if attempt == 0:
                start_engine(settings)
            time.sleep(.25)
    return None


def run_op(settings, *args, timeout=180):
    if settings.backend == 'm3' and not settings.fake and args[0] in ('continue', 'assemble', 'export'):
        done = warm_call(settings, args, timeout)
        if done is not None:
            return done
    try:
        done = subprocess.run([settings.engine_python, str(OPS), *map(str, args)], capture_output=True, text=True, timeout=timeout,
                              env=op_env(settings), cwd=settings.engine_root)
    except subprocess.TimeoutExpired as exc:
        raise ApiError(504, 'engine_timeout', f'{args[0]} took longer than {timeout} s') from exc
    return done


class Worker(threading.Thread):
    """Runs one generation at a time. A draft left running by a restart is reported as failed, never silently redone."""

    def __init__(self, store):
        super().__init__(daemon=True, name='noteforge-worker')
        self.store, self.jobs = store, queue.Queue()
        self.current, self.process, self.cancelled = None, None, set()
        self.lock = threading.Lock()
        self.waiting = []
        self.warm_generating = False
        for folder in sorted(store.drafts.iterdir(), key=lambda p: p.stat().st_mtime):
            if not (folder/'status.json').is_file():
                continue
            state = store.status(folder)['state']
            if state == 'running':
                store.set_status(folder, state='failed', error=dict(code='interrupted', message='the service restarted during generation; submit again'))
            elif state == 'queued':
                self.submit(folder.name)

    def submit(self, draft_id):
        with self.lock:
            self.waiting.append(draft_id)
        self.jobs.put(draft_id)

    def position(self, draft_id):
        with self.lock:
            return self.waiting.index(draft_id) if draft_id in self.waiting else None

    def cancel(self, draft_id):
        with self.lock:
            self.cancelled.add(draft_id)
            if self.current == draft_id and self.warm_generating:
                stop_engine(self.store.s)  # the only way to stop a draft inside it; the next call starts it again
            elif self.current == draft_id and self.process is not None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

    def run(self):
        while True:
            draft_id = self.jobs.get()
            with self.lock:
                self.waiting.remove(draft_id)
                if draft_id in self.cancelled:
                    self.cancelled.discard(draft_id)
                    continue
                self.current = draft_id
            try:
                self.one(draft_id)
            except Exception as exc:  # noqa: BLE001 - the worker must survive any one draft
                self.store.set_status(self.store.drafts/draft_id, state='failed', error=dict(code='internal', message=str(exc)[:300]))
            finally:
                with self.lock:
                    self.current, self.process = None, None

    def child(self, draft_id, args, limit):
        """Run one op in its own process group, cancellable. Returns a CompletedProcess or None if it timed out."""
        s = self.store.s
        with self.lock:
            self.process = subprocess.Popen([s.engine_python, str(OPS), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, env=op_env(s), cwd=s.engine_root, start_new_session=True)
            process = self.process
        try:
            out, err = process.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return None
        return subprocess.CompletedProcess([], process.returncode, out, err)

    def one(self, draft_id):
        store, s = self.store, self.store.s
        folder = store.drafts/draft_id
        store.set_status(folder, state='running', started_at=time.time())
        (folder/'progress.json').unlink(missing_ok=True)
        limit = s.max_wall_s+1200  # features and the first pass come on top of the generation limit
        done = None
        if s.backend == 'm3' and not s.fake:
            # Features run beside the engine as it loads; everything after them runs inside it.
            start_engine(s)
            done = self.child(draft_id, ['features', str(folder)], limit)
            if done is not None and done.returncode == 0 and draft_id not in self.cancelled:
                self.warm_generating = True
                try:
                    done = warm_call(s, ['generate', folder, 'features-ready'], limit)
                except ApiError:
                    done = None
                finally:
                    self.warm_generating = False
                if done is None and draft_id in self.cancelled:
                    done = subprocess.CompletedProcess([], 1, '', '')
                elif done is None:
                    done = self.child(draft_id, ['generate', str(folder), 'features-ready'], limit)
        else:
            done = self.child(draft_id, ['generate', str(folder)], limit)
        if done is None:
            store.set_status(folder, state='failed', error=dict(code='engine_timeout', message=f'generation took longer than {int(limit)} s'))
            return
        with self.lock:
            cancelled = draft_id in self.cancelled
            self.cancelled.discard(draft_id)
        if cancelled:
            store.set_status(folder, state='cancelled')
        elif done.returncode == 0:
            store.set_status(folder, state='done', finished_at=time.time(), latest_revision=0)
        else:
            store.set_status(folder, state='failed', error=op_error(done, 'engine_failed'), finished_at=time.time())


def ms_note(e):
    note = dict(id=e['id'], time_ms=e['onset_us']/1000, end_ms=e['end_us']/1000, type=e['type'])
    if 'x' in e:
        note['x'] = e['x']
    if e.get('lane', -1) >= 0:
        note['lane'] = e['lane']
    return note


def agreement(mine, others, tolerance=20_000):
    """Mean F1 of this candidate's onsets against the other candidates in the same section."""
    if not others:
        return 1.
    total = 0.
    for other in others:
        if not mine and not other:
            total += 1.
            continue
        i = j = hit = 0
        while i < len(mine) and j < len(other):
            gap = mine[i]-other[j]
            if abs(gap) <= tolerance:
                hit, i, j = hit+1, i+1, j+1
            elif gap < 0:
                i += 1
            else:
                j += 1
        total += 2*hit/max(len(mine)+len(other), 1)
    return total/len(others)


def create_app(**over):
    settings = Settings(**over)
    store = Store(settings)
    worker = Worker(store)
    worker.start()
    app = FastAPI(title='NoteForge draft API', version=API_VERSION,
                  description='Draft rhythm game charts for an editor. Times are ms on the uploaded audio\'s clock.')
    app.state.store, app.state.worker = store, worker
    if settings.cors:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors, allow_methods=['*'], allow_headers=['*'])

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=dict(error=dict(code=exc.code, message=exc.message, field=exc.field)))

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        field = '.'.join(str(x) for x in first['loc'] if x != 'body')
        return JSONResponse(status_code=422, content=dict(error=dict(code='invalid_request', message=first['msg'], field=field or None)))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content=dict(error=dict(code=f'http_{exc.status_code}', message=str(exc.detail), field=None)))

    def auth(authorization: str = Header(default='')):
        if not hmac.compare_digest(authorization.encode(), f'Bearer {settings.token}'.encode()):
            raise ApiError(401, 'unauthorized', 'send Authorization: Bearer <token>')

    def missing_engine():
        if settings.fake:
            return []
        need = ('fast_weights', 'fast_mert', 'mel_stats', 'plan', 'planner_dir') if settings.backend == 'm3' else ('model', 'plan', 'planner_dir')
        return [k for k in need if not getattr(settings, k) or not Path(getattr(settings, k)).exists()]

    @app.get('/v1/health', tags=['service'])
    def health():
        """No token needed. `ready` means a draft can be generated."""
        missing = missing_engine()
        return dict(status='ok', api_version=API_VERSION, ready=not missing, missing=missing, running=worker.current,
                    queued=len(worker.waiting), engine='fake' if settings.fake else 'r5-l322m', backend=settings.backend)

    @app.get('/v1/capabilities', dependencies=[Depends(auth)], tags=['service'])
    def capabilities():
        """What a client can ask for, and the defaults."""
        return dict(modes=[m.value for m in Mode], aggregations=['vote', 'single', 'most_agreeing'], default_aggregation='vote', default_samples=8,
                    profiles=['relaxed', 'standard', 'expert', 'none'], default_profile='standard',
                    chart_formats=dict(phira=['phira_zip', 'rpe_json'], mania_4k=['osz', 'osu']), default_phira_speed=10.,
                    audio_types=sorted(AUDIO_TYPES), max_upload_mb=settings.max_upload//CHUNK,
                    level_required=dict(phira=True, mania_4k=False), section_lengths_s=[8, 16, 32], backend=settings.backend,
                    plans=['balanced', 'fast', 'sequential'] if settings.backend == 'm3' else [], recommended_samples=1 if settings.backend == 'm3' else 8, continue_supported=settings.backend == 'm3' or settings.fake)

    async def save_upload(file: UploadFile, folder: Path):
        handle, name = tempfile.mkstemp(dir=folder, prefix='.upload-')
        digest, size = hashlib.sha256(), 0
        try:
            with os.fdopen(handle, 'wb') as out:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > settings.max_upload:
                        raise ApiError(413, 'too_large', f'the limit is {settings.max_upload//CHUNK} MB')
                    digest.update(chunk)
                    out.write(chunk)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
        if not size:
            Path(name).unlink(missing_ok=True)
            raise ApiError(422, 'empty_file', 'the file is empty')
        return Path(name), digest.hexdigest(), size

    @app.post('/v1/audio', dependencies=[Depends(auth)], status_code=201, tags=['inputs'])
    async def upload_audio(file: UploadFile = File(...)):
        """Upload a song. The same file always gets the same `audio_id`, so its features are reused."""
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in AUDIO_TYPES:
            raise ApiError(415, 'unsupported_audio', f'audio must be one of {sorted(AUDIO_TYPES)}', 'file')
        path, sha, size = await save_upload(file, store.audio)
        target = store.audio/(sha+suffix)
        if not settings.fake and shutil.which('ffprobe'):
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)], capture_output=True, text=True)
            if probe.returncode or not probe.stdout.strip():
                path.unlink(missing_ok=True)
                raise ApiError(415, 'unsupported_audio', 'the file could not be read as audio', 'file')
        path.replace(target)
        return dict(audio_id=sha, bytes=size, format=suffix[1:])

    @app.post('/v1/charts', dependencies=[Depends(auth)], status_code=201, tags=['inputs'])
    async def upload_chart(file: UploadFile = File(...), mode: Mode = Form(...)):
        """Upload a charter's chart (.osu for mania, RPE or official JSON or a package zip for Phira). Returns the timing and opening style found in it."""
        path, sha, _ = await save_upload(file, store.charts)
        chart_id = hashlib.sha256(f'{mode.value}:{sha}'.encode()).hexdigest()
        folder = store.charts/chart_id
        suffix = Path(file.filename or '').suffix.lower() or '.bin'
        try:
            if (folder/'info.json').is_file():
                return json.loads((folder/'info.json').read_text())
            work = Path(tempfile.mkdtemp(dir=store.charts, prefix='.chart-'))
            source = work/('chart'+suffix)
            path.replace(source)
            done = run_op(settings, 'inspect-chart', source, mode.value)
            if done.returncode:
                error = op_error(done, 'chart_unreadable')
                shutil.rmtree(work, ignore_errors=True)
                raise ApiError(422, error['code'], error['message'], 'file')
            info = json.loads(done.stdout.strip().splitlines()[-1])
            info.update(chart_id=chart_id, file=source.name)
            write_json(work/'info.json', info)
            work.replace(folder)
            return info
        finally:
            path.unlink(missing_ok=True)

    def resolve(request: DraftRequest):
        audio = store.audio_path(request.audio_id)
        mode = request.mode.value
        resolved = dict(audio_path=str(audio), features_dir=str(store.features), model=settings.model, plan=settings.plan, planner_dir=settings.planner_dir,
                        backend=settings.backend, fast_weights=settings.fast_weights, fast_mert=settings.fast_mert, mel_stats=settings.mel_stats,
                        mert_source=settings.mert_source, mert_python=settings.mert_python, device=settings.device, batch_size=settings.batch_size, max_wall_s=settings.max_wall_s)
        if request.opening_chart_id:
            _, info = store.chart(request.opening_chart_id, 'opening_chart_id')
            if info['mode'] != mode:
                raise ApiError(422, 'mode_mismatch', f"opening chart is {info['mode']}, the draft is {mode}", 'opening_chart_id')
            if not info.get('opening_style'):
                raise ApiError(422, 'chart_empty', 'the opening chart has no notes', 'opening_chart_id')
            resolved['opening_style'] = info['opening_style']
        timing = request.timing
        if timing is not None:
            if timing.chart_id:
                folder, info = store.chart(timing.chart_id, 'timing.chart_id')
                if info['mode'] != mode:
                    raise ApiError(422, 'mode_mismatch', f"timing chart is {info['mode']}, the draft is {mode}", 'timing.chart_id')
                if not info.get('segments'):
                    raise ApiError(422, 'no_timing', 'this chart has no usable timing', 'timing.chart_id')
                resolved['timing'] = dict(segments=info['segments'])
                if mode == 'mania_4k':
                    resolved['timing']['osu_path'] = str(folder/info['file'])
            elif timing.segments:
                ordered = sorted(timing.segments, key=lambda s: s.start_ms)
                resolved['timing'] = dict(segments=[[s.start_ms*1000, 60e6/s.bpm, s.meter] for s in ordered])
            else:
                resolved['timing'] = dict(bpm=timing.bpm)
        return resolved

    def view(folder):
        status = store.status(folder)
        out = dict(status, draft_id=folder.name)
        if status['state'] == 'queued':
            out['queue_position'] = worker.position(folder.name)
        if status['state'] == 'running' and (folder/'progress.json').is_file():
            try:
                out['progress'] = json.loads((folder/'progress.json').read_text())
            except ValueError:
                pass
        if status['state'] == 'done' and (folder/'meta.json').is_file():
            meta = json.loads((folder/'meta.json').read_text())
            out['song'] = dict(duration_ms=meta['duration_us']/1000, codec=meta['codec'], clock_shift_ms=meta['shift_us']/1000)
            out['used'] = dict(level=meta.get('level'), style=meta.get('style'), timing_points=meta.get('timing_points'))
            out['revisions'] = sorted(int(p.name) for p in (folder/'revisions').iterdir() if (p/'events.json').is_file())
        return out

    @app.post('/v1/drafts', dependencies=[Depends(auth)], status_code=202, tags=['drafts'])
    def create_draft(request: DraftRequest):
        """Start a draft. Takes minutes; poll `GET /v1/drafts/{id}`. The same request returns the existing draft unless it failed or was cancelled."""
        if missing_engine():
            raise ApiError(503, 'engine_not_configured', 'missing: '+', '.join(missing_engine()))
        resolved = resolve(request)
        body = request.model_dump(mode='json')
        draft_id = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        folder = store.drafts/draft_id
        with store.lock:
            if (folder/'status.json').is_file():
                state = store.status(folder)['state']
                if state not in ('failed', 'cancelled'):
                    return view(folder)
                shutil.rmtree(folder)
            folder.mkdir()
            write_json(folder/'request.json', body)
            write_json(folder/'resolved.json', resolved)
            write_json(folder/'status.json', dict(state='queued', created_at=time.time(), updated_at=time.time()))
        worker.submit(draft_id)
        return view(folder)

    @app.get('/v1/drafts/{draft_id}', dependencies=[Depends(auth)], tags=['drafts'])
    def get_draft(draft_id: str):
        return view(store.draft_dir(draft_id))

    @app.delete('/v1/drafts/{draft_id}', dependencies=[Depends(auth)], tags=['drafts'])
    def cancel_draft(draft_id: str):
        """Cancel a queued or running draft. A finished draft is not changed."""
        folder = store.draft_dir(draft_id)
        state = store.status(folder)['state']
        if state in ('queued', 'running'):
            worker.cancel(draft_id)
            if state == 'queued':
                store.set_status(folder, state='cancelled')
        return view(folder)

    def finished(draft_id):
        folder = store.draft_dir(draft_id)
        state = store.status(folder)['state']
        if state != 'done':
            raise ApiError(409, 'draft_not_ready', f'the draft is {state}')
        return folder

    def revision_file(folder, revision):
        latest = store.status(folder)['latest_revision']
        number = latest if revision is None else revision
        path = folder/'revisions'/str(number)/'events.json'
        if not path.is_file():
            raise ApiError(404, 'revision_not_found', f'no revision {number}', 'revision')
        return number, path

    @app.get('/v1/drafts/{draft_id}/events', dependencies=[Depends(auth)], tags=['drafts'])
    def get_events(draft_id: str, revision: int | None = Query(default=None, ge=0)):
        """The notes of a revision (latest by default), ready to load into an editor."""
        folder = finished(draft_id)
        number, path = revision_file(folder, revision)
        data = json.loads(path.read_text())
        return dict(draft_id=draft_id, revision=number, mode=json.loads((folder/'request.json').read_text())['mode'], base=data['base'], choices=data['choices'],
                    warnings=data['warnings'], notes=[ms_note(e) for e in data['events']])

    @app.get('/v1/drafts/{draft_id}/sections', dependencies=[Depends(auth)], tags=['candidates'])
    def get_sections(draft_id: str, length_s: float = Query(default=16, ge=2, le=120), start_s: float = Query(default=0, ge=0), end_s: float | None = Query(default=None, gt=0),
                     max_candidates: int = Query(default=16, ge=1, le=16), include_notes: bool = True):
        """Alternatives per section, most agreeing first. Send a chosen `sample` back through `POST .../revisions`.

        These are raw candidates: rhythm only, before clean-up and placement."""
        folder = finished(draft_id)
        samples = json.loads((folder/'samples.json').read_text())['samples']
        duration = json.loads((folder/'meta.json').read_text())['duration_us']
        step, start, stop = int(length_s*1e6), int(start_s*1e6), min(duration, int(end_s*1e6) if end_s else duration)
        sections = []
        while start < stop:
            end = min(start+step, stop)
            notes = [[n for n in s if start <= n[0] < end] for s in samples]
            onsets = [sorted({n[0] for n in s}) for s in notes]
            rows = [dict(sample=i, note_count=len(s), agreement=round(agreement(onsets[i], onsets[:i]+onsets[i+1:]), 3),
                         **(dict(notes=[dict(time_ms=n[0]/1000, end_ms=n[1]/1000, type=n[2], **(dict(lane=n[3]) if n[3] >= 0 else {})) for n in s]) if include_notes else {}))
                    for i, s in enumerate(notes)]
            rows.sort(key=lambda r: (-r['agreement'], r['sample']))
            sections.append(dict(start_ms=start/1000, end_ms=end/1000, candidates=rows[:max_candidates]))
            start = end
        return dict(draft_id=draft_id, length_s=length_s, sections=sections)

    @app.post('/v1/drafts/{draft_id}/revisions', dependencies=[Depends(auth)], status_code=201, tags=['candidates'])
    def create_revision(draft_id: str, request: RevisionRequest):
        """Make a new revision from vote or a sample: send every section choice together. Earlier revisions stay."""
        folder = finished(draft_id)
        mode = json.loads((folder/'request.json').read_text())['mode']
        choices = []
        for c in request.choices:
            row = dict(start_us=int(c.start_ms*1000), end_us=int(c.end_ms*1000))
            if c.notes is not None:
                for n in c.notes:
                    if mode == 'mania_4k' and (n.lane is None or n.type not in ('tap', 'hold')):
                        raise ApiError(422, 'invalid_note', 'mania notes need a lane and type tap or hold', 'choices.notes')
                    if mode == 'phira' and n.lane is not None:
                        raise ApiError(422, 'invalid_note', 'Phira notes use x, not lane', 'choices.notes')
                row['notes'] = [dict(onset_us=int(n.time_ms*1000), end_us=int(n.end_ms*1000), type=n.type, lane=n.lane if n.lane is not None else -1, x=n.x) for n in c.notes]
            else:
                row['sample'] = c.sample
            choices.append(row)
        with revision_lock(folder):
            numbers = [int(p.name) for p in (folder/'revisions').iterdir() if (p/'events.json').is_file()]
            number = max(numbers)+1
            spec = folder/f'.revision-{number}.json'
            write_json(spec, dict(revision=number, base=request.base.model_dump(exclude_none=True), choices=choices))
            done = run_op(settings, 'assemble', folder, spec)
            spec.unlink(missing_ok=True)
            if done.returncode:
                error = op_error(done, 'assemble_failed')
                raise ApiError(422 if error['code'] in ('bad_sample', 'choices_overlap') else 500, error['code'], error['message'], 'choices')
            store.set_status(folder, latest_revision=number)
        return get_events(draft_id, number)

    @app.post('/v1/drafts/{draft_id}/continue', dependencies=[Depends(auth)], status_code=201, tags=['candidates'])
    def continue_draft(draft_id: str, request: ContinueRequest):
        """Keep the notes before `from_ms` exactly and write the rest as a new revision. Much faster than a new draft: the audio is not read again and only the remaining sections run."""
        folder = finished(draft_id)
        meta = json.loads((folder/'meta.json').read_text())
        if request.from_ms*1000 >= meta['duration_us']:
            raise ApiError(422, 'bad_cut', 'from_ms is past the end of the song', 'from_ms')
        source, _ = revision_file(folder, request.revision)
        if worker.current is not None:
            raise ApiError(409, 'busy', 'a draft is generating; try again when it finishes')
        with revision_lock(folder):
            numbers = [int(p.name) for p in (folder/'revisions').iterdir() if (p/'events.json').is_file()]
            number = max(numbers)+1
            spec = folder/f'.continue-{number}.json'
            write_json(spec, dict(revision=number, from_revision=source, cut_us=int(request.from_ms*1000), seed=request.seed))
            done = run_op(settings, 'continue', folder, spec, timeout=int(settings.max_wall_s))
            spec.unlink(missing_ok=True)
            if done.returncode:
                error = op_error(done, 'continue_failed')
                raise ApiError(501 if error['code'] == 'unsupported' else 500, error['code'], error['message'])
            store.set_status(folder, latest_revision=number)
        return get_events(draft_id, number)

    revision_locks, revision_guard = {}, threading.Lock()

    def revision_lock(folder):
        with revision_guard:
            return revision_locks.setdefault(folder.name, threading.Lock())

    @app.get('/v1/drafts/{draft_id}/chart', dependencies=[Depends(auth)], tags=['drafts'])
    def get_chart(draft_id: str, revision: int | None = Query(default=None, ge=0), format: str | None = None, speed: float = Query(default=10., gt=0, le=100)):
        """Download a playable chart file. `speed` is the Phira note speed."""
        folder = finished(draft_id)
        number, _ = revision_file(folder, revision)
        mode = json.loads((folder/'request.json').read_text())['mode']
        allowed = dict(phira=('phira_zip', 'rpe_json'), mania_4k=('osz', 'osu'))[mode]
        form = format or allowed[0]
        if form not in allowed:
            raise ApiError(422, 'bad_format', f'{mode} supports {allowed}', 'format')
        ext = dict(phira_zip='zip', rpe_json='json', osz='osz', osu='osu')[form]
        exports = folder/'exports'
        exports.mkdir(exist_ok=True)
        name = f'{number}-{speed:g}.{ext}' if mode == 'phira' else f'{number}.{ext}'
        out = exports/name
        if not out.is_file():
            done = run_op(settings, 'export', folder, number, form, speed, out)
            if done.returncode:
                error = op_error(done, 'export_failed')
                raise ApiError(500, error['code'], error['message'])
        return FileResponse(out, filename=f'noteforge-{draft_id[:8]}-r{number}.{ext}')

    return app
