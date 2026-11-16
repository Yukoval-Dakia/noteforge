"""Draft an editor-openable chart for a new audio file: osu!mania 4K .osz or Phira preview zip.

  1. Decode audio to 24 kHz mono with ffmpeg.
  2. Extract the pinned MERT-v2 layers 12/23 and complete-tail 10 ms mel.
  3. Read those receipts through the published MertFeatureStore and mel normalizer.
  4. Generate the whole song in 8 s cores with optional caller-supplied density,
     style, timing and visible chart references. No hidden target is read.
  5. Aggregate samples by onset votes and export an editable chart on the
     playback clock. The CLI and HTTP API use the same inference runtime.

The training recipe, corpus and evaluation panels are not distributed.

  python -m noteforge.scripts.draft_song --model models/noteforge-r5.pt \
      --audio song.mp3 --mode phira --out-dir drafts/song --consensus 8

Feature extraction needs ffmpeg/ffprobe, transformers==4.53.2 and a pinned
MERT-v2 source; pass --mert-source DIR or 'hf' to download the pinned revision.
"""
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noteforge.scripts import runtime as wb  # noqa: E402
from noteforge.backends import configure_torch, mlx_identity, resolve_torch_device, torch_identity  # noqa: E402

SCHEMA = 'noteforge.draft-song.v1'
MODES = ('mania_4k', 'phira')
PLATFORM_SOURCE = {'mania_4k': 'osu', 'phira': 'phira'}
PRODUCT_CONSENSUS = 8
PRODUCT_TEMPERATURE = 1.
PRODUCT_FIELD_TEMPERATURE = {'kind': 1., 'head_block': .7, 'head_cell': .7, 'head_ms': .7}
PRODUCT_VOTES = 3
PRODUCT_SOURCE = 'Frozen r5 decoding defaults: 8 samples, T=1, anchor pattern and 3-of-8 votes.'

# Per-(source, codec) playback-clock constants for the frozen model.
# (source, codec): (c_ms, support, basis)
CODEC_CONSTANTS = {
    ('osu', 'mp3'): (14, 11773, 'source_codec'),
    ('osu', 'vorbis'): (14, 3439, 'source_codec'),
    ('phira', 'flac'): (0, 0, 'source_pooled'),
    ('phira', 'mp3'): (-2, 2308, 'source_codec'),
    ('phira', 'pcm_f32le'): (0, 0, 'source_pooled'),
    ('phira', 'pcm_s16le'): (29, 58, 'source_codec'),
    ('phira', 'pcm_s24le'): (0, 0, 'source_pooled'),
    ('phira', 'pcm_u8'): (0, 0, 'source_pooled'),
    ('phira', 'vorbis'): (10, 396, 'source_codec'),
}
# Source-pooled fallback constants for codecs without a dedicated constant.
SOURCE_POOLED_CONSTANTS = {'osu': (14, 0), 'phira': (0, 0)}
CODEC_TABLE_SOURCE = dict(schema='noteforge.playback-clock-constants.v1',
                          rule='Frozen per-(source, codec) model-to-playback clock constants.')

# Tier -> frozen constant heads/s requests published for the r5 inference presets.
DIFFICULTY_PRESETS = {
    ('mania_4k', 'low'): 4.267, ('mania_4k', 'medium'): 8.494, ('mania_4k', 'high'): 13.178,
    ('phira', 'low'): 4.011, ('phira', 'medium'): 5.669, ('phira', 'high'): 7.452,
}
PRESET_SOURCE = dict(schema='noteforge.difficulty-presets.v1',
                     rule='Frozen low/medium/high heads-per-second request presets for r5.')

# Frozen mel normalization coefficients (sha256 -> repository path).
KNOWN_MEL_STATS = {
    '75a057dde89e8bd16102dddc00f8d2a7cf4ad065693557847d0d14db89255b4c':
        'models/config/mel-statistics.json',
}
# Optional repo-local pinned MERT-v2 source layout.
LOCAL_MERT_SPLIT = {'config.json': 'checkpoint/config.json', 'model.safetensors': 'checkpoint/model.safetensors',
                    'configuration_mert2.py': 'mert2_pinned/configuration_mert2.py',
                    'modeling_mert2.py': 'mert2_pinned/modeling_mert2.py'}
FEATURE_CONTRACT = dict(
    encoder='noteforge.audio.mert2.FrozenMert2 bf16, pinned Hugging Face revision',
    paths='FrozenMert2.encode per song + complete_song_mel; longer songs split into windows',
    decoder='ffmpeg',
    backend='cuda_bf16')
CODE_FILES = ('noteforge/scripts/draft_song.py', 'noteforge/scripts/runtime.py',
              'noteforge/scripts/build_chart_timing.py', 'noteforge/model/local_song_rollout.py',
              'noteforge/model/full_event_generate.py', 'noteforge/model/full_event_model.py',
              'noteforge/inference/features.py', 'noteforge/inference/normalization.py',
              'noteforge/audio/features.py', 'noteforge/audio/mert2.py',
              'noteforge/audio/local_mel_cache.py', 'noteforge/ingest/audio.py',
              'noteforge/events/chart_export.py')
TIMING_ISSUES = {'bad_timing_point', 'uninherited_nonpositive_beatlength', 'bad_uninherited_flag',
                 'no_uninherited_timing_point'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str)+'\n')
    tmp.replace(path)


def auto_device():
    return resolve_torch_device('auto')


def tool_version(name):
    try:
        out = subprocess.run([name, '-version'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.splitlines()[0].strip() if out.returncode == 0 and out.stdout else None


# ------------------------------------------------------------------ clock
def codec_constant(mode, codec, override_ms=None):
    """Model clock -> playback clock for an export to ``mode``'s platform with audio ``codec``."""
    source = PLATFORM_SOURCE[mode]
    if override_ms is not None:
        return dict(source=source, codec=codec, c_ms=int(override_ms), n_train=None, basis='user override',
                    table=CODEC_TABLE_SOURCE)
    if (source, codec) in CODEC_CONSTANTS:
        c_ms, n_train, basis = CODEC_CONSTANTS[(source, codec)]
    else:
        c_ms, n_train = SOURCE_POOLED_CONSTANTS[source]
        basis = 'source_pooled (codec has no database constant)'
    return dict(source=source, codec=codec, c_ms=c_ms, n_train=n_train, basis=basis, table=CODEC_TABLE_SOURCE)


# ------------------------------------------------------------------ model
def build_model_on_meta(config, vocabulary, state):
    """Construct the model on the meta device and copy the published weights in.

    Variables are not learned at load time: every parameter is overwritten from the
    checkpoint. Building on meta skips the random initialisation of ~223 Linear
    layers (about 0.75 s of the 0.77 s construction on the Apple host measured in
    docs/BACKENDS.md) and ``assign=True`` keeps the tensors instead of copying into
    freshly allocated storage. Falls back to the ordinary path if this PyTorch
    build rejects meta construction or the state is not a plain mapping.
    """
    import torch
    from noteforge.model.full_event_model import FullEventModel
    if not isinstance(state, dict):
        model = FullEventModel(config, vocabulary)
        model.load_state_dict(state, strict=True)
        return model
    try:
        with torch.device('meta'):
            model = FullEventModel(config, vocabulary)
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to('cpu')
    except Exception:  # noqa: BLE001 - any meta/assign incompatibility falls back to a plain load
        model = FullEventModel(config, vocabulary)
        model.load_state_dict(state, strict=True)
    return model


def load_model(path, device, weights='auto'):
    """Load a published r5 inference model -> (eval model, identity, declared mel normalization)."""
    import torch
    from noteforge.model.full_event_codec import EventVocabulary
    from noteforge.model.full_event_model import FullEventConfig, FullEventModel
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(saved, dict) or 'config' not in saved or 'model' not in saved:
        raise SystemExit(f'{path}: not a published r5 model (needs config + model)')
    if saved.get('schema') != 'noteforge.finalmodelv1':
        raise SystemExit(f'{path}: not a published r5 model (expected noteforge.finalmodelv1)')
    chosen = saved.get('weights')
    if chosen != 'ema' or weights not in ('auto', 'ema'):
        raise SystemExit(f'{path} is a published model holding {chosen!r} weights')
    state = saved['model']
    config = FullEventConfig(**saved['config'])
    vocabulary = dict(saved.get('vocabulary') or {})
    model = build_model_on_meta(config, EventVocabulary(**vocabulary), state)
    model = model.to(device).eval()
    inference = saved.get('inference') or {}
    declared = inference.get('mel_normalization')
    identity = dict(path=str(path), sha256=wb.sha_file(path), bytes=path.stat().st_size,
                    kind='published_model', schema=saved.get('schema'), weights=chosen,
                    parameters=sum(p.numel() for p in model.parameters()), inference=inference,
                    config=asdict(config), vocabulary=vocabulary)
    del saved, state
    return model, identity, declared


def resolve_mel_normalization(declared, *, mel_stats=None, root=ROOT):
    """(statistics or None, identity) for the model's declared mel normalization.

    ``declared``: 'none' | {path, sha256} (inference contract) | None (undeclared: --mel-stats).
    The statistics file is located at --mel-stats, the declared path (absolute or repo-relative) or a known local
    copy, and must match the declared sha256."""
    if declared == 'none':
        return None, dict(applied=False, declared='none')
    expected = None if declared is None else declared.get('sha256')
    if declared is None and mel_stats is None:
        raise SystemExit('The model does not declare its mel normalization: pass --mel-stats')
    candidates = []
    if mel_stats is not None:
        candidates.append(Path(mel_stats))
    if declared is not None and declared.get('path'):
        declared_path = Path(declared['path'])
        candidates += [declared_path] if declared_path.is_absolute() else [root/declared_path]
    if expected in KNOWN_MEL_STATS:
        candidates.append(root/KNOWN_MEL_STATS[expected])
    tried = []
    for candidate in candidates:
        if not candidate.is_file():
            tried.append(f'{candidate} (missing)')
            continue
        digest = wb.sha_file(candidate)
        if expected is not None and digest != expected:
            tried.append(f'{candidate} (sha256 {digest[:12]} != {expected[:12]})')
            continue
        stats = json.loads(candidate.read_text())
        coefficient_sha = stats.get('coefficient_sha256')
        if (declared is not None and declared.get('coefficient_sha256') is not None
                and coefficient_sha != declared['coefficient_sha256']):
            tried.append(f'{candidate} (coefficient sha256 {coefficient_sha} != declared)')
            continue
        return stats, dict(applied=True, declared=declared, path=str(candidate), sha256=digest,
                           schema=stats.get('schema'), coefficient_sha256=coefficient_sha,
                           applied_by='noteforge.inference.normalization.MelNormalizedFeatures')
    raise SystemExit('Mel normalization statistics not found / identity mismatch; tried: ' + '; '.join(tried))


# ------------------------------------------------------------------ features
def resolve_mert_source(arg, root=ROOT):
    """(directory for FrozenMert2(local_source=...) or None for the pinned Hugging Face revision, identity, temp)."""
    from noteforge.audio.mert2 import LOCAL_SOURCE_SHA256, REPO, REVISION
    if arg == 'hf':
        return None, dict(kind='huggingface', repo=REPO, revision=REVISION), None
    candidates = [Path(arg)] if arg else [ROOT/'models/mert2-source']
    for candidate in candidates:
        if all((candidate/name).is_file() for name in LOCAL_SOURCE_SHA256):
            return candidate, dict(kind='directory', path=str(candidate), expected_sha256=LOCAL_SOURCE_SHA256), None
        split = {name: candidate/rel for name, rel in LOCAL_MERT_SPLIT.items()}
        if all(p.is_file() for p in split.values()):
            # One directory as from_pretrained/validate_local_source expect: symlinks to the pinned files.
            assembled = Path(tempfile.mkdtemp(prefix='noteforge-mert-source-'))
            for name, p in split.items():
                (assembled/name).symlink_to(p.resolve())
            return assembled, dict(kind='assembled_symlinks', path=str(candidate),
                                   files={k: str(v) for k, v in split.items()},
                                   expected_sha256=LOCAL_SOURCE_SHA256), assembled
    raise SystemExit('Pinned MERT-v2 source not found (tried ' + ', '.join(map(str, candidates)) + '); pass '
                     '--mert-source DIR (config.json, model.safetensors, configuration_mert2.py, modeling_mert2.py) '
                     "or --mert-source hf to download the pinned revision")


def make_extractor(device, source):
    """The database MERT encoder: FrozenMert2 on cuda (the database path) / cpu; on mps the local execution policy
    of the local MERT execution policy (CPU FP32 frontend, query-chunked SDPA, bf16 encoder on MPS)."""
    from noteforge.audio.mert2 import FrozenMert2
    if device != 'mps':
        return FrozenMert2(device=device, local_source=source), f'{device}_bf16'
    from noteforge.audio.local_mert import CpuMelTransfer, LocalFrozenMert2
    from noteforge.audio.mert_attention import install_query_chunks
    base = FrozenMert2(device='cpu', local_source=source)  # pinned-identity load (validate_local_source)
    model = base.model.to('mps')
    model.feature_extractor = CpuMelTransfer(model.feature_extractor, 'mps')
    install_query_chunks(model, 512)
    return LocalFrozenMert2(model, base.spec), 'mps_bf16_cpu_fp32_frontend_query_chunk512'


def extractor_identity(extractor, backend, device, source_identity):
    import torch
    from noteforge.audio.mert2 import CELL_SAMPLES, REPO, REVISION, SR, Mert2Spec
    from noteforge.audio.preprocess import PREPROCESS_RECIPE
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    spec = getattr(extractor, 'spec', None) or Mert2Spec()
    return dict(cls=f'{type(extractor).__module__}.{type(extractor).__qualname__}', backend=backend, device=device,
                repo=REPO, revision=REVISION, spec=asdict(spec), sample_rate=SR, cell_samples=CELL_SAMPLES,
                frame_clock='times_s = (k*960+360)/24000, intervals [k*960, min((k+1)*960, samples)]/24000',
                dtype='float16 (bf16 forward)', mel='noteforge.audio.local_mel_cache.complete_song_mel (fp16, '
                '4 frames per 40 ms cell, complete zero-padded tail)', mel_recipe=PREPROCESS_RECIPE,
                torch=torch.__version__, transformers=transformers_version, source=source_identity,
                database=FEATURE_CONTRACT)


def prepare_features(audio_path, audio_sha, features_dir, args, *, extractor=None, decoder=None):
    """Existing verified features for this audio, else decode + extract + write the database layout.

    Returns (receipt, record) where record holds the stage timings and whether features were reused."""
    from noteforge.audio.mert2 import REVISION
    from noteforge.audio.features import existing_feature_receipt
    folder = Path(features_dir)/'features'/audio_sha
    if (folder/'receipt.json').is_file():
        receipt = existing_feature_receipt(folder, SimpleNamespace(revision=REVISION))  # revision + file sha256
        backend = str((receipt.get('draft_song') or {}).get('extractor', {}).get('backend', receipt.get('backend')))
        if receipt.get('complete') is not True or receipt.get('audio_sha256') != audio_sha:
            raise SystemExit(f'{folder}: incomplete feature receipt or a different audio')
        if backend.startswith('injected') and extractor is None:
            raise SystemExit(f'{folder} was written by an injected (test) extractor; refusing to draft from it')
        return receipt, dict(reused=True, folder=str(folder))
    import numpy as np
    from noteforge.ingest.audio import decode
    from noteforge.audio.features import commit_feature, feature_payloads, write_json as db_write_json
    timings = {}
    codec = args.codec or probe(audio_path)
    tick = time.monotonic()
    wave = (decoder or decode)(str(audio_path))
    timings['decode_s'] = time.monotonic()-tick
    if not len(wave) or not np.isfinite(wave).all():
        raise SystemExit(f'{audio_path}: decoded audio is empty or not finite')
    source_identity, assembled = None, None
    tick = time.monotonic()
    try:
        if extractor is None:
            source, source_identity, assembled = resolve_mert_source(args.mert_source)
            mert_device = args.mert_device or args.device or auto_device()
            extractor, backend = make_extractor(mert_device, source)
        else:
            mert_device = getattr(extractor, 'device', None)
            backend = f'injected:{type(extractor).__name__}'
        timings['mert_load_s'] = time.monotonic()-tick
        identity = extractor_identity(extractor, backend, mert_device, source_identity)
        marks = {}
        tick = time.monotonic()
        features, mel, expected = feature_payloads(
            wave, extractor, phase_callback=lambda phase: marks.__setitem__(phase, time.monotonic()))
        done = time.monotonic()
    finally:
        if assembled is not None:
            shutil.rmtree(assembled, ignore_errors=True)
    timings['mert_s'] = marks['gpu_encode_end']-marks['gpu_encode_start']
    timings['mel_s'] = done-marks['gpu_encode_end']
    scratch = Path(tempfile.mkdtemp(prefix='noteforge-draft-scratch-'))  # never contains the user's audio
    tick = time.monotonic()
    try:
        receipt = commit_feature(audio_sha, Path(audio_path), features_dir, scratch, features, mel, expected)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    # commit_feature labels every receipt with the database backend; record the backend that actually ran.
    receipt['backend'] = backend.split(':')[0] if backend.startswith('injected') else backend
    receipt['draft_song'] = dict(schema=SCHEMA, audio=str(audio_path), codec=codec, codec_ext=Path(audio_path).suffix,
                                 decoder=dict(fn='noteforge.ingest.audio.decode',
                                              command='ffmpeg -nostdin -v error -i <audio> -vn -f f32le -ac 1 -ar 24000',
                                              ffmpeg=tool_version('ffmpeg'), injected=decoder is not None),
                                 extractor=identity, timings=dict(timings))
    db_write_json(folder/'receipt.json', receipt)
    timings['features_write_s'] = time.monotonic()-tick
    del extractor, features, mel, wave
    gc.collect()
    _empty_device_cache()
    return receipt, dict(reused=False, folder=str(folder), timings=timings)


def probe(audio_path):
    """ffprobe codec_name of the first audio stream (noteforge.ingest.audio.probe_codec)."""
    from noteforge.ingest.audio import probe_codec
    try:
        return probe_codec(str(audio_path))
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'ffprobe failed on {audio_path} ({type(exc).__name__}: {exc}); pass --codec') from exc


def _empty_device_cache():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def open_features(features_dir, audio_sha, stats, expected_sha256):
    """Open the published feature store and apply the frozen mel normalization."""
    from noteforge.inference.normalization import MelNormalizedFeatures
    from noteforge.inference.features import MertFeatureStore
    store = MertFeatureStore(Path(features_dir), Path(features_dir)/'features')
    store.ensure(audio_sha)
    store._cache(audio_sha)  # shape / clock / coverage checks of MertFeatureStore._open
    if stats is None:
        return store
    # The check guards the published coefficients, not the identity of a new song.
    return MelNormalizedFeatures(store, stats, expected_sha256=expected_sha256)


# ------------------------------------------------------------------ timing
def timing_section_lines(data):
    """Non-empty lines of the [TimingPoints] section, verbatim (line endings removed)."""
    out, inside = [], False
    for index, raw in enumerate(bytes(data).splitlines()):
        line = raw.decode('utf-8')
        if index == 0:
            line = line.lstrip('﻿')
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            inside = stripped == '[TimingPoints]'
        elif inside and stripped:
            out.append(line)
    return out


def _timing_rows(chart):
    side = chart.sidecar or {}
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'line'} for r in rows or ()]  # noqa: E731
    return strip(side.get('timing_rows')), strip(side.get('sv_rows'))


def read_timing(path):
    """Charter timing from a 4K osu!mania .osu: verbatim [TimingPoints] lines, red points [(ms, bpm)] and beat
    segments [(start_us, beat_us, meter)] on the chart's (playback) clock."""
    from noteforge.events.chart_export import timing_points_from_map
    from noteforge.events.osu_mania import parse_osu
    from noteforge.scripts.build_chart_timing import GimmickTiming, timing_segments
    path = Path(path)
    data = path.read_bytes()
    try:
        chart = parse_osu(data, path.name)
    except Exception as exc:  # noqa: BLE001 - surfaced as a CLI error
        raise SystemExit(f'--timing {path}: not a readable 4K osu!mania file ({type(exc).__name__}: {exc})')
    bad = sorted({i.code for i in chart.issues if i.code in TIMING_ISSUES and i.severity == 'quarantine'})
    if chart.timing is None or bad:
        raise SystemExit(f'--timing {path}: unusable timing points ({", ".join(bad) or "no timing"})')
    lines = timing_section_lines(data)
    try:
        segments, meter_source = timing_segments(chart)
        gimmick = None
    except GimmickTiming as exc:
        segments, meter_source, gimmick = None, None, str(exc)
    red, green = _timing_rows(chart)
    return dict(path=str(path), sha256=wb.sha_file(path), lines=lines, red_points=timing_points_from_map(chart.timing),
                segments=segments, meter_source=meter_source, gimmick=gimmick, red_lines=len(red),
                green_lines=len(green), first_bpm=round(60e6/segments[0][1], 4) if segments else None,
                other_quarantine=sorted({i.code for i in chart.quarantine_reasons()} - TIMING_ISSUES))


def splice_timing_points(osu_bytes, lines):
    """Replace the exporter's [TimingPoints] block with ``lines`` (verbatim), CRLF like the exporter."""
    rows = bytes(osu_bytes).decode('utf-8').split('\r\n')
    start = rows.index('[TimingPoints]')
    end = next(i for i in range(start+1, len(rows)) if rows[i].startswith('['))
    return '\r\n'.join(rows[:start+1] + list(lines) + [''] + rows[end:]).encode('utf-8')


def write_verbatim_timing(export, timing, events):
    """--timing on mania: the written .osu carries the charter's [TimingPoints] block unchanged; re-verified with
    the repo parser (notes exact, timing rows identical) and the .osz rebuilt."""
    from noteforge.events.osu_mania import parse_osu
    osu = Path(export['chart']['path'])
    data = splice_timing_points(osu.read_bytes(), timing['lines'])
    parsed = parse_osu(data, osu.name)
    source = parse_osu(Path(timing['path']).read_bytes(), Path(timing['path']).name)
    same_rows = _timing_rows(parsed) == _timing_rows(source)
    same_lines = timing_section_lines(data) == timing['lines']
    if not (same_rows and same_lines) or parsed.timing is None:
        raise SystemExit('Verbatim charter timing did not survive the osu round trip')
    osu.write_bytes(data)
    roundtrip = wb.verify_osu_on_disk(osu, events)
    if not roundtrip['exact']:
        raise SystemExit('Draft notes changed after writing the charter timing')
    files = [osu, osu.parent/f"audio{export['audio']['ext']}"]
    if export.get('osz'):
        Path(export['osz']).unlink()
        export['osz'] = str(wb.write_osz(Path(export['osz']), files))
    export['disk_roundtrip'] = roundtrip
    export['timing_points'] = dict(written='verbatim [TimingPoints] block of --timing (playback clock, unshifted)',
                                   lines=len(timing['lines']), red_lines=timing['red_lines'],
                                   green_lines=timing['green_lines'], rows_identical=same_rows,
                                   lines_identical=same_lines)
    return export


# ------------------------------------------------------------------ conditions
def constant_density_provider(density, *, cell_us, record):
    """condition_provider: every core requests one constant heads/s ``density`` -- the same rendering as
    scripts.runtime.preset_condition_provider (inference.conditions.condition_inputs)."""
    from noteforge.inference.conditions import condition_inputs

    def provider(state, request):
        segments = [dict(start_us=int(request.start_us), end_us=int(request.end_us), density=float(density))]
        update = condition_inputs(state['mode'], request, segments=segments, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(record))
    return provider


STYLE_FIELDS = ('hold_head_ratio', 'chord_ratio', 'hold_occupancy_ratio')
# Phira types are steered at decode time: the condition bands barely move flick and drag, a log-probability bias does.
# Frozen decode bias: .8 x the logit change of the share lands close by.
KIND_BASE = dict(flick=.071, drag=.043)
KIND_GAIN = 1/.8


def kind_bias(args):
    out = {}
    if getattr(args, 'kind_as_band', False):  # shares went to the model as a request; a decode bias on top would overshoot
        return out
    for kind, field in (('flick', 'flick_ratio'), ('drag', 'drag_ratio')):
        value = getattr(args, field, None)
        if value is None:
            continue
        if args.mode != 'phira':
            raise SystemExit(f'{field} is a Phira control')
        if not 0 < value < 1:
            raise SystemExit(f'{field} must be between 0 and 1')
        base = KIND_BASE[kind]
        out[kind] = KIND_GAIN*(math.log(value/(1-value))-math.log(base/(1-base)))/1
    return out


def style_bands(args):
    """The charter's requested shares as table bands. A field this mode has no control for is an error."""
    from noteforge.events import controls as C
    bands = {}
    fields = (*STYLE_FIELDS, 'flick_ratio', 'drag_ratio') if getattr(args, 'kind_as_band', False) else STYLE_FIELDS
    for field in fields:
        value = getattr(args, field, None)
        if value is None:
            continue
        if field not in C.MODE_FIELDS[args.mode]:
            raise SystemExit(f'{field} is not a control of mode {args.mode}')
        if not 0 <= value <= 1:
            raise SystemExit(f'{field} must be within 0-1')
        edges = C.BAND_EDGES[field]
        bands[field] = next([lo, hi] for lo, hi in zip(edges, edges[1:]) if lo <= value < hi or (hi == edges[-1] and value == hi))
    return bands


def curve_style_provider(segments, bands, *, cell_us):
    """condition_provider: the charter's section densities (labelled charter_intent_density) plus style bands."""
    from noteforge.inference.conditions import condition_inputs

    def provider(state, request):
        parts = [dict(start_us=max(int(s['start_s']*1e6), request.start_us), end_us=min(int(s['end_s']*1e6), request.end_us),
                      density=float(s['density'])) for s in segments
                 if s['start_s']*1e6 < request.end_us and s['end_s']*1e6 > request.start_us]
        update = condition_inputs(state['mode'], request, bands=bands, segments=parts, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(label='charter_intent_style', segments=parts, bands=bands))
    return provider


def condition_request(args, model, cell_us):
    """(condition_provider or None, guidance or None, record)."""
    level = getattr(args, 'phira_level', None), getattr(args, 'mania_star', None)
    bands = style_bands(args)
    curve = json.loads(args.density_curve.read_text()) if getattr(args, 'density_curve', None) else None
    if curve is not None or (bands and args.tier is None and args.density is None and level == (None, None)):
        if curve is not None and (args.tier is not None or args.density is not None or level != (None, None)):
            raise SystemExit('--density-curve replaces --tier / --density / level requests')
        guidance = None if args.guidance is None or float(args.guidance) == 1. else float(args.guidance)
        return curve_style_provider(curve or [], bands, cell_us=cell_us), guidance, dict(
            label='charter_intent_style', bands=bands, density_curve=curve, render_cell_us=cell_us, guidance=args.guidance)
    if args.tier is None and args.density is None and level == (None, None):
        return None, None, dict(label='unconditioned', rule='unspecified density curve + unconditional controls')
    if not getattr(model.c, 'intensity_dim', 0):
        raise SystemExit('--tier / --density / --phira-level / --mania-star need a model with a density curve input '
                         '(config.intensity_dim > 0)')
    if level != (None, None):
        from noteforge.events.difficulty_presets import DifficultyPresets
        presets = DifficultyPresets.bundled()
        kind, value = ('phira', level[0]) if level[0] is not None else ('mania_4k', level[1])
        density = presets.heads_per_s(kind, value)
        record = dict(label='difficulty_level_preset', **{('phira_level' if kind == 'phira' else 'mania_star'): value},
                      density=round(density, 3))
        source = presets.source(kind)
    elif args.tier is not None:
        density = DIFFICULTY_PRESETS[(args.mode, args.tier)]
        record = dict(label='charter_difficulty_preset', tier=args.tier, density=density)
        source = PRESET_SOURCE
    else:
        density = float(args.density)
        record = dict(label='charter_intent_density', density=density)
        source = 'user: constant heads/s for every core'
    guidance = None if args.guidance is None or float(args.guidance) == 1. else float(args.guidance)
    provider = constant_density_provider(density, cell_us=cell_us, record=record)
    return provider, guidance, dict(record, unit='heads (notes) per second, constant over every 8 s core',
                                    render_cell_us=cell_us, source=source, guidance=args.guidance,
                                    cfg='off (plain conditioning)' if guidance is None else
                                        'logits_u + w*(logits_c - logits_u); unconditional = unspecified curve + '
                                        'controls')


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--model', type=Path, help='published r5 model weights')
    p.add_argument('--backend', choices=('auto', 'torch', 'mlx'), default=os.environ.get('NOTEFORGE_BACKEND', 'auto'),
                   help='auto prefers MLX on Apple Silicon when its exported weights are configured, else PyTorch')
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--timing-segments', type=Path, help='JSON [[start_us, beat_us, meter], ...] on the playback clock: '
                   'full charter timing as a model input for any mode (exclusive with --timing and --beat-grid)')
    p.add_argument('--kind-as-band', action='store_true', help='flick/drag shares as request bands (how the frozen setup was measured), not as a decode bias')
    p.add_argument('--save-samples', action='store_true', help='also write every sample to draft-samples.json (model clock)')
    p.add_argument('--timing', type=Path, help="the charter's 4K osu!mania .osu: timing points written verbatim, "
                   'fed to beat_features models as charter_timing_input')
    level = p.add_mutually_exclusive_group()
    level.add_argument('--tier', choices=('low', 'medium', 'high'), help='difficulty tier preset (train-median heads/s)')
    level.add_argument('--density', type=float, help='requested constant heads (notes) per second')
    level.add_argument('--phira-level', type=float, metavar='LV',
                       help='Phira level preset, HD 8 .. AT 16 (database median density of charts at that level)')
    level.add_argument('--mania-star', type=float, metavar='STAR',
                       help='osu!mania 4K star preset, about 1 .. 7 (approximate star; database median density)')
    p.add_argument('--guidance', type=float, help='CFG weight for --tier/--density (default: plain conditioning)')
    p.add_argument('--consensus', type=int, default=PRODUCT_CONSENSUS, help='samples per draft (1 = single sample)')
    p.add_argument('--consensus-pattern', choices=('majority', 'anchor'), default='anchor',
                   help='pattern of each kept onset: per-onset majority, or the most central sample')
    p.add_argument('--consensus-votes', type=int, help=f'samples an onset needs (default: {PRODUCT_VOTES} of 8, scaled)')
    for field, name in (('hold_head_ratio', 'hold'), ('chord_ratio', 'chord'), ('drag_ratio', 'drag'), ('flick_ratio', 'flick'),
                        ('hold_occupancy_ratio', 'hold-occupancy')):
        p.add_argument(f'--{name}-ratio', type=float, dest=field, metavar='R', help=f'requested share, 0-1 ({field})')
    p.add_argument('--density-curve', type=Path, help='charter section densities: JSON list of {start_s, end_s, density}')
    p.add_argument('--profile', choices=('relaxed', 'standard', 'expert', 'none'), help="Phira playability limits (default "
                   "standard): fingers, holds down, close onsets. See noteforge/events/phira_postprocess.py")
    p.add_argument('--consensus-min-jack-ms', type=float, default=100.,
                   help='mania: avoid same-key repeats under this gap when a voted pattern allows it (0 = off)')
    p.add_argument('--no-fragment-guard', dest='fragment_guard', action='store_false',
                   help='keep lone squeezed onsets that fewer than half of the samples wrote')
    p.add_argument('--scheduler', choices=('legacy', 'grouped', 'continuous'), default='continuous',
                   help='batch filling; token-identical, continuous is the fastest on CUDA')
    p.add_argument('--temperature', default=str(PRODUCT_TEMPERATURE), help="sampling temperature or 'none' (greedy)")
    p.add_argument('--field-temperature', type=json.loads, default=None,
                   help=f'JSON per-field temperatures (default {json.dumps(PRODUCT_FIELD_TEMPERATURE)})')
    p.add_argument('--top-p', type=float)
    p.add_argument('--hold-release-ms', type=float, default=40.,
                   help='mania: a hold releases >= MS before the next press of the same key, else it is shortened '
                        '(or becomes a tap); 0 disables (default 40: holds ending on the next note are unplayable)')
    p.add_argument('--seed', type=int, default=0, help='samples use seeds seed..seed+N-1')
    p.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default=os.environ.get('NOTEFORGE_DEVICE', 'auto'),
                   help='PyTorch device; auto: cuda, else mps, else cpu')
    p.add_argument('--cpu-threads', type=int, help='PyTorch CPU intra-op threads (or NOTEFORGE_CPU_THREADS)')
    p.add_argument('--cpu-interop-threads', type=int, help='PyTorch CPU inter-op threads (or NOTEFORGE_CPU_INTEROP_THREADS)')
    p.add_argument('--reference-decode', action='store_true', help='disable the optimized static-cache decode loop')
    p.add_argument('--no-cuda-graph', action='store_true', help='disable CUDA graph replay on NVIDIA GPUs')
    p.add_argument('--autocast', choices=('bf16',), help='explicit bf16 decode; this changes the numerical path')
    p.add_argument('--compile-decode', action='store_true', help='request torch.compile for the eager decode step')
    p.add_argument('--mlx-weights', type=Path, default=os.environ.get('NOTEFORGE_FAST_WEIGHTS') or os.environ.get('MLX_FAST'),
                   help='exported MLX fast-model folder')
    p.add_argument('--mlx-mert', type=Path, default=os.environ.get('NOTEFORGE_FAST_MERT'),
                   help='exported MLX fast-MERT folder')
    p.add_argument('--mlx-cache-mb', type=int, default=300, help='MLX cache limit in MiB')
    p.add_argument('--mlx-plan', choices=('balanced', 'fast', 'sequential'), default='balanced',
                   help='MLX rollout plan')
    p.add_argument('--batch-size', type=int, default=64, help='rows (8 s cores) per generation batch')
    p.add_argument('--token-cap', type=int, default=4096, help='token cap per 8 s core')
    p.add_argument('--max-wall-s', type=float, default=14400., help='wall limit of the whole rollout')
    p.add_argument('--beat-grid', metavar='START_US,BEAT_US', help='one constant beat grid on the model clock, as a model '
                   'input only (see scripts/planner/auto_draft.py); the export still carries no charter timing')
    p.add_argument('--history-reference', action='store_true',
                   help="each core sees the model's own previous 8 s (history_context-trained models only)")
    p.add_argument('--plan', type=Path, help='optional inference contract with counterfactual_cell_us')
    p.add_argument('--mel-stats', type=Path, help='mel normalization statistics file (sha256 checked when declared)')
    p.add_argument('--features-dir', type=Path, help='persistent feature dir (reused when it holds this audio); '
                   'default: a temporary dir removed afterwards')
    p.add_argument('--extract-only', action='store_true', help='write the features (needs --features-dir) and stop')
    p.add_argument('--mert-source', help="pinned MERT-v2 directory, or 'hf' to download the pinned revision")
    p.add_argument('--mert-device', choices=('auto', 'cuda', 'mps', 'cpu'), help='default: --device')
    p.add_argument('--codec', help='audio codec name (ffprobe codec_name) when ffprobe is unavailable')
    p.add_argument('--clock-constant-ms', type=int, help='override the model->playback clock constant c')
    p.add_argument('--title')
    p.add_argument('--artist', default='unknown')
    p.add_argument('--version', dest='chart_version', help='difficulty name written into the chart')
    a = p.parse_args(argv)
    if not a.extract_only and (a.model is None or a.mode is None):
        p.error('--model and --mode are required (or --extract-only)')
    if a.extract_only and a.features_dir is None:
        p.error('--extract-only needs --features-dir (a temporary feature dir is deleted)')
    if a.consensus < 1:
        p.error('--consensus must be >= 1')
    temperature = wb.parse_temperature(a.temperature)
    if a.consensus > 1 and temperature is None:
        p.error('--consensus > 1 needs sampling (identical greedy samples)')
    if a.field_temperature is not None and temperature is None:
        p.error('--field-temperature needs sampling')
    if a.field_temperature is None and temperature is not None:
        a.field_temperature = dict(PRODUCT_FIELD_TEMPERATURE)
    if a.phira_level is not None and a.mode not in (None, 'phira'):
        p.error('--phira-level is a Phira preset (use --mania-star for mania_4k)')
    if a.mania_star is not None and a.mode not in (None, 'mania_4k'):
        p.error('--mania-star is a mania_4k preset (use --phira-level for phira)')
    if a.phira_level is not None and not 6 <= a.phira_level <= 18:
        p.error('--phira-level must be within 6..18 (presets are fit on levels 6..18)')
    if a.mania_star is not None and not 1 <= a.mania_star <= 7:
        p.error('--mania-star must be within 1..7 (enough database charts only there)')
    if a.guidance is not None and (a.tier is None and a.density is None and a.phira_level is None
                                   and a.mania_star is None or not a.guidance > 0):
        p.error('--guidance needs --tier / --density / --phira-level / --mania-star and a positive weight')
    if a.density is not None and not a.density > 0:
        p.error('--density must be positive')
    return a


def stem_for(audio, mode):
    base = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(audio).stem).strip('._') or 'draft'
    return f'{base[:80]}-{mode}'


def code_identity():
    return {name: wb.sha_file(ROOT/name) for name in CODE_FILES if (ROOT/name).is_file()}


def run(args, *, extractor=None, decoder=None):
    """Execute one draft (or extraction) and return the receipt; ``extractor`` / ``decoder`` are injectable."""
    started = time.monotonic()
    requested_device = args.device or 'auto'
    device = resolve_torch_device(requested_device)
    changed_threads = configure_torch(getattr(args, 'cpu_threads', None), getattr(args, 'cpu_interop_threads', None))
    args.device = device
    if getattr(args, 'mert_device', None) == 'auto':
        args.mert_device = device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out/'draft-receipt.json'
    keep = Path(args.features_dir).resolve() if args.features_dir else None
    leftovers = [p for p in out.iterdir() if p.resolve() != keep]
    if leftovers:
        raise SystemExit(f'--out-dir {out} is not empty ({leftovers[0].name}); drafts are never overwritten')
    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        raise SystemExit(f'--audio {audio_path} not found')
    receipt = dict(schema=SCHEMA, created_unix=time.time(), command=list(getattr(args, 'argv', None) or sys.argv),
                   code=code_identity(), complete=False)
    active_backend = os.environ.get('NOTEFORGE_ACTIVE_BACKEND', 'torch')
    receipt['runtime'] = dict(
        backend=active_backend,
        requested_backend=getattr(args, 'backend', 'torch'),
        requested_device=requested_device,
        control_device=device,
        thread_overrides=changed_threads,
        torch=torch_identity(device),
    )
    if active_backend == 'mlx':
        receipt['runtime']['mlx'] = mlx_identity()
    timings = receipt['timings'] = {}
    temporary = keep is None
    features_dir = keep or Path(tempfile.mkdtemp(prefix='noteforge-draft-'))
    try:
        tick = time.monotonic()
        audio_sha = wb.sha_file(audio_path)
        feature_receipt, feature_record = prepare_features(audio_path, audio_sha, features_dir, args,
                                                           extractor=extractor, decoder=decoder)
        timings.update(feature_record.get('timings') or {})
        timings['features_total_s'] = time.monotonic()-tick
        info = feature_receipt.get('draft_song') or {}
        codec = args.codec or info.get('codec') or probe(audio_path)
        receipt['audio'] = dict(path=str(audio_path), sha256=audio_sha, bytes=audio_path.stat().st_size,
                                codec=codec, codec_source='--codec' if args.codec else
                                'feature receipt (ffprobe at extraction)' if info.get('codec') else 'ffprobe',
                                decoded_samples=feature_receipt['decoded_samples'],
                                decoder=info.get('decoder'), database_decoder=FEATURE_CONTRACT['decoder'])
        receipt['features'] = dict(dir=str(features_dir), temporary=temporary, reused=feature_record['reused'],
                                   folder=feature_record['folder'],
                                   store='noteforge.inference.features.MertFeatureStore(root, <dir>/features)',
                                   layout='features/<audio_sha256>/{L12,L23,times_s,intervals_s,mel10_complete}.npy + '
                                          'receipt.json, written by noteforge.audio.features.commit_feature',
                                   receipt=feature_receipt)
        if args.extract_only:
            receipt.update(complete=True, extract_only=True)
            return receipt
        tick = time.monotonic()
        model, model_identity, declared = load_model(args.model, device)
        plan = json.loads(Path(args.plan).read_text()) if args.plan else {}
        if plan:
            if declared is not None and plan.get('mel_normalization') not in (None, declared):
                raise SystemExit('--plan mel_normalization disagrees with the published model contract')
            declared = declared if declared is not None else plan.get('mel_normalization')
            model_identity['plan'] = dict(path=str(args.plan), sha256=wb.sha_file(args.plan), id=plan.get('id'))
        stats, normalization = resolve_mel_normalization(declared, mel_stats=args.mel_stats)
        cell_us = int(plan.get('counterfactual_cell_us', 100_000))
        timings['model_load_s'] = time.monotonic()-tick
        receipt['model'] = model_identity
        receipt['features']['mel_normalization'] = normalization
        features = open_features(features_dir, audio_sha, stats,
                                 normalization['coefficient_sha256'] if stats is not None else None)
        duration_us = int(features.records[audio_sha]['duration_us'])
        receipt['audio']['duration_us'] = duration_us
        clock = codec_constant(args.mode, codec, args.clock_constant_ms)
        shift_us = clock['c_ms']*1000
        receipt['clock'] = dict(clock, shift_us=shift_us,
                                model_clock='decoded-audio clock (ffmpeg 24 kHz mono); frozen model targets use the '
                                            'published playback-clock constant',
                                rule='written (playback) time = model time - c; a new song has no offset decision, '
                                     'so the per-chart residual shift is 0')
        timing, timing_provider = None, None
        if args.timing:
            timing = read_timing(args.timing)
            beat_clock = codec_constant('mania_4k', codec, args.clock_constant_ms)  # the .osu's own platform
            beat = dict(used=False, shift_us=beat_clock['c_ms']*1000, clock=beat_clock,
                        meter_source=timing['meter_source'])
            if not getattr(model.c, 'beat_features', False):
                beat['reason'] = 'model has no config.beat_features (timing is written, not a model input)'
            elif timing['segments'] is None:
                beat['reason'] = f"gimmick timing, no usable beat grid: {timing['gimmick']} (absent state)"
            else:
                segments = [[round(s+beat['shift_us'], 3), round(b, 6), int(m)] for s, b, m in timing['segments']]
                beat.update(used=True, label='charter_timing_input', segments_model_clock=segments,
                            rule='per core: noteforge.inference.beat_features.window_beat_features of the charter grid '
                                 'on the model clock (raw grid + c)')
                timing_provider = lambda state, request, segments=segments: segments  # noqa: E731
            timing['beat_features'] = beat
            receipt['timing'] = {k: v for k, v in timing.items() if k != 'segments'}
            receipt['timing']['segments_playback_clock'] = timing['segments']
        if getattr(args, 'timing_segments', None):
            if args.timing or getattr(args, 'beat_grid', None):
                raise SystemExit('--timing-segments, --timing and --beat-grid are exclusive')
            if not getattr(model.c, 'beat_features', False):
                raise SystemExit('--timing-segments needs a model with config.beat_features')
            given = json.loads(Path(args.timing_segments).read_text())
            segments = [[round(float(s)+shift_us, 3), round(float(b), 6), int(m)] for s, b, m in given]
            if not segments or any(b <= 0 or m < 1 for _, b, m in segments):
                raise SystemExit('--timing-segments needs at least one segment with positive beat and meter')
            timing_provider = lambda state, request, segments=segments: segments  # noqa: E731
            receipt['timing_segments'] = dict(segments_model_clock=segments, label='charter_timing_input')
        if getattr(args, 'beat_grid', None):
            if args.timing:
                raise SystemExit('--beat-grid and --timing are exclusive')
            if not getattr(model.c, 'beat_features', False):
                raise SystemExit('--beat-grid needs a model with config.beat_features')
            start_us, beat_us = (float(x) for x in args.beat_grid.split(','))
            if beat_us <= 0:
                raise SystemExit('--beat-grid needs a positive beat')
            grid = [[round(start_us, 3), round(beat_us, 6), 4]]
            timing_provider = lambda state, request, segments=grid: segments  # noqa: E731
            receipt['beat_grid'] = dict(segments_model_clock=grid, bpm=round(60e6/beat_us, 4), label='auto_draft_bpm')
        provider, guidance, condition = condition_request(args, model, cell_us)
        temperature = wb.parse_temperature(args.temperature)
        seeds = [args.seed+k for k in range(args.consensus)]
        song_id = f'draft:{audio_sha[:16]}'
        song = dict(chart_id=song_id, audio_sha256=audio_sha, mode=args.mode, duration_us=duration_us)
        args.kind_bias = kind_bias(args)
        if args.kind_bias and args.scheduler != 'continuous':
            raise SystemExit('--flick-ratio / --drag-ratio need --scheduler continuous')
        generator = wb.make_generator(temperature, args.top_p, seeds[0], args.field_temperature,
                                      wb.decode_options(args), guidance=guidance)
        intensity = None
        if model.c.intensity_dim:
            from noteforge.inference.density import grid_density_curve
            intensity = lambda audio, start, end: grid_density_curve([], start, end, cell_us=cell_us, scale=16.)  # noqa: E731
        from noteforge.model.full_event_generate import CUDA_GRAPH_DEFAULT, FAST_DEFAULT, fast_supported
        from noteforge.model.local_song_rollout import generate_complete_song_samples
        tick = time.monotonic()
        results = generate_complete_song_samples(
            model, features, [song], seeds, device=device, batch_size=args.batch_size, token_cap=args.token_cap,
            max_wall_s=args.max_wall_s, history_reference=args.history_reference, intensity_provider=intensity,
            condition_provider=provider, timing_provider=timing_provider, generator=generator,
            scheduler=args.scheduler)
        timings['generation_s'] = time.monotonic()-tick
        states = [r['songs'][0] for r in results]
        receipt['decoding'] = dict(
            rule='greedy' if temperature is None else 'ancestral', temperature=temperature, top_p=args.top_p,
            field_temperature=args.field_temperature, seeds=seeds, samples=len(seeds),
            rng='per-row torch.Generator seeded from (seed, mode, duration, core start)',
            aggregation='single sample' if len(seeds) == 1 else f'consensus_{len(seeds)} (D6)',
            consensus=None if len(seeds) == 1 else dict(
                samples=len(seeds), window_ms=wb.CONSENSUS_WINDOW_US//1000, min_votes=math.ceil(len(seeds)/2),
                onset='lower median', pattern='majority (ties: earliest sample)',
                hold_tail='lower median of the voting members',
                overlap='mania: holds containing a later same-lane head dropped (head kept as tap)'),
            token_cap_per_core=args.token_cap, core_us=8_000_000, batch_size=args.batch_size,
            max_wall_s=args.max_wall_s, history_reference=bool(args.history_reference), device=device,
            generation_loop=dict(fast=bool(FAST_DEFAULT and fast_supported(model)),
                                 cuda_graph=bool(CUDA_GRAPH_DEFAULT and device == 'cuda')),
            condition=condition, density_curve_render_cell_us=cell_us,
            cell_us_source='plan counterfactual_cell_us' if 'counterfactual_cell_us' in plan else
                           'default 100000 (every recipe-v2 plan)',
            charter_timing_input=None if timing is None else timing['beat_features'],
            product_default=dict(consensus=PRODUCT_CONSENSUS, temperature=PRODUCT_TEMPERATURE,
                                 field_temperature=PRODUCT_FIELD_TEMPERATURE, source=PRODUCT_SOURCE),
            matches_product_default=bool(len(seeds) == PRODUCT_CONSENSUS and temperature == PRODUCT_TEMPERATURE
                                         and args.field_temperature == PRODUCT_FIELD_TEMPERATURE
                                         and args.top_p is None))
        receipt['samples'] = [dict(seed=seed, complete=st['complete'], failure=st['failure'], notes=len(st['events']),
                                   cores=len(st['cores']), tokens=sum(len(c['tokens']) for c in st['cores']),
                                   conditioned_cores=sum(c.get('condition') is not None for c in st['cores']),
                                   charter_timing_cores=sum(bool((c.get('charter_timing') or {}).get('present'))
                                                            for c in st['cores']),
                                   legality_issues=len(st.get('legality') or ()))
                              for seed, st in zip(seeds, states)]
        if not all(st['complete'] for st in states):
            raise SystemExit('Incomplete samples: ' + json.dumps(receipt['samples']))
        if getattr(args, 'save_samples', False):
            write_json(out/'draft-samples.json', dict(clock='model (decoded audio)', seeds=seeds, samples=[
                [[int(x[0]), int(x[1]), x[2], int(x[3])] for x in wb.draft_tuples(song, st)] for st in states]))
        tick = time.monotonic()
        if len(seeds) == 1:
            notes = wb.draft_tuples(song, states[0])
            aggregation = dict(samples=1, notes=len(notes))
        else:
            notes, aggregation = wb.consensus_draft(
                [wb.draft_tuples(song, st) for st in states], args.mode, duration_us,
                min_jack_us=int(round(args.consensus_min_jack_ms*1000)) if args.consensus_min_jack_ms else None,
                pattern_source=args.consensus_pattern,
                min_votes=args.consensus_votes or max(1, round(len(seeds)*PRODUCT_VOTES/PRODUCT_CONSENSUS)),
                fragment_guard=args.fragment_guard)
        if args.hold_release_ms:
            from noteforge.inference.hold_release import repair_hold_release
            notes, aggregation['hold_release_repair'] = repair_hold_release(
                notes, args.mode, min_release_us=int(round(args.hold_release_ms * 1000)))
        aggregation['legality'] = wb.consensus_legality(notes, args.mode, duration_us)
        timings['consensus_s'] = time.monotonic()-tick
        if aggregation['legality']:
            raise SystemExit('Draft fails the independent legality check: ' + json.dumps(aggregation['legality'][:5]))
        events = [dict(onset_us=int(a), end_us=int(b), type=t, lane=int(lane)) for a, b, t, lane in notes]
        profile = args.profile or ('standard' if args.mode == 'phira' else 'none')
        if args.mode == 'phira' and profile != 'none':
            from noteforge.events.phira_postprocess import playable
            events, aggregation['playability'] = playable(events, profile)
            aggregation['playability_profile'] = profile
        types = {}
        for e in events:
            types[e['type']] = types.get(e['type'], 0)+1
        receipt['draft'] = dict(aggregation, events_file='draft-events.json', clock='model',
                                notes_per_second=round(len(events)/max(duration_us/1e6, 1e-6), 4), types=types)
        write_json(out/'draft-events.json', dict(schema=SCHEMA, clock='model (decoded audio)', mode=args.mode,
                                                 audio_sha256=audio_sha, duration_us=duration_us, events=events))
        tick = time.monotonic()
        from noteforge.ingest.audio import codec_ext
        stem = stem_for(audio_path, args.mode)
        audio = dict(ok=True, source=f'--audio {audio_path}', ext=audio_path.suffix.lower() or '.'+codec_ext(codec),
                     path=audio_path)
        label = args.tier or (f'{args.density:g} nps' if args.density is not None else
                              f'Phira Lv.{args.phira_level:g}' if getattr(args, 'phira_level', None) is not None else
                              f'{args.mania_star:g}* preset' if getattr(args, 'mania_star', None) is not None else
                              'unconditioned')
        version = args.chart_version or (f'NoteForge draft ({label}, '
                                         + ('single sample' if len(seeds) == 1 else f'consensus {len(seeds)}') + ')')
        export = wb.export_song(out/stem, stem, dict(mode=args.mode, audio_sha256=audio_sha), events, audio,
                                title=args.title or audio_path.stem, artist=args.artist, version=version,
                                timing_points=timing['red_points'] if timing and args.mode == 'mania_4k' else None,
                                clock_shift_us=shift_us,
                                clock_source=f"database constant c({clock['source']}, {codec}) = {clock['c_ms']} ms "
                                             f"[{clock['basis']}]")
        if export.get('failed'):
            raise SystemExit('Export failed: ' + export['failed'])
        if args.mode == 'mania_4k':
            if not export['disk_roundtrip']['exact']:
                raise SystemExit('Exported .osu does not round-trip the draft')
            if timing is not None:
                written = [dict(e, onset_us=e['onset_us']-shift_us, end_us=e['end_us']-shift_us) for e in events
                           if e['onset_us']-shift_us >= 0]
                export = write_verbatim_timing(export, timing, written)
            else:
                export['timing_points'] = dict(written='placeholder: one 120 BPM point at 0 ms (no --timing); the '
                                                       'charter is expected to time the song')
        elif timing is not None:
            export['timing_points'] = dict(written='none: the Phira preview has no charter timing (beat features only)')
        timings['export_s'] = time.monotonic()-tick
        receipt['export'] = export
        receipt['complete'] = True
        return receipt
    finally:
        timings['total_s'] = time.monotonic()-started
        if receipt.get('complete') or receipt.get('audio'):
            write_json(receipt_path, receipt)
        if temporary:
            shutil.rmtree(features_dir, ignore_errors=True)


def _print_receipt(receipt, args):
    if receipt.get('extract_only'):
        print(json.dumps(dict(features=receipt['features']['folder'], audio_sha256=receipt['audio']['sha256'],
                              codec=receipt['audio']['codec'], timings=receipt['timings']), indent=1))
    else:
        export = receipt['export']
        print(json.dumps(dict(chart=export.get('osz') or export['chart'].get('path'), notes=receipt['draft']['notes'],
                              clock_shift_ms=receipt['clock']['c_ms'], receipt=str(Path(args.out_dir)/'draft-receipt.json'),
                              timings={k: round(v, 3) for k, v in receipt['timings'].items()}), indent=1))


def main(argv=None, *, extractor=None, decoder=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw)
    args.argv = ['noteforge.scripts.draft_song', *raw]
    if args.backend in ('auto', 'mlx'):
        from noteforge.backends import resolve_backend
        target = resolve_backend(args.backend, args.device, args.mlx_weights, args.mlx_mert)
        if target.backend == 'mlx':
            if not target.ready:
                raise SystemExit(target.error)
            stride = dict(fast='2', balanced='-2').get(args.mlx_plan)
            env = dict(MLX_FAST=str(args.mlx_weights), MLX_CACHE_MB=str(args.mlx_cache_mb),
                       NOTEFORGE_FAST_WEIGHTS=str(args.mlx_weights),
                       NOTEFORGE_FAST_MERT=str(args.mlx_mert),
                       NOTEFORGE_ACTIVE_BACKEND='mlx')
            if stride:
                env['MLX_STRIDE'] = stride
            from noteforge.mlx import draft_mlx
            receipt = draft_mlx.run(raw, env)
            _print_receipt(receipt, args)
            return receipt
    receipt = run(args, extractor=extractor, decoder=decoder)
    _print_receipt(receipt, args)
    return receipt


if __name__ == '__main__':
    main()
