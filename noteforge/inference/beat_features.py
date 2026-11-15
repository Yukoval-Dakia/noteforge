"""Beat-grid features for an optional charter timing input.

A charter supplies BPM / offset / meter before placing notes. These features are
computed from the caller's timing only; no hidden chart or evaluation panel is read.
"""
from pathlib import Path

import numpy as np

from noteforge.model.full_event_model import BEAT_FEATURE_DIM

SCHEMA = 'chart-timing-table-v1'
BEAT_FEATURE_NAMES = ('beat_sin', 'beat_cos', 'bar_sin', 'bar_cos', 'beat2_sin', 'beat2_cos',
                      'beat4_sin', 'beat4_cos', 'offset_quarter', 'offset_third', 'log_beat', 'present')
OFFSET_CLIP_MS = 25.
LOG_BEAT_CLIP = 3.
LABEL = 'charter_timing_input'
if len(BEAT_FEATURE_NAMES) != BEAT_FEATURE_DIM:
    raise ImportError('Beat feature names and model input width disagree')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def segment_array(segments):
    """[(start_us, beat_us, meter), ...] -> float64 [S, 3] sorted by start (stable), or None."""
    if segments is None:
        return None
    seg = np.asarray(segments, np.float64).reshape(-1, 3)
    if not len(seg):
        return None
    if not np.all(np.isfinite(seg)) or np.any(seg[:, 1] <= 0) or np.any(seg[:, 2] < 1):
        raise ValueError('Charter timing segments need finite start, beat_us > 0 and meter >= 1')
    return seg[np.argsort(seg[:, 0], kind='stable')]


def beat_features(times_us, segments):
    """float32 [T, BEAT_FEATURE_DIM] for absolute song times (us); all zeros when segments is None."""
    t = np.asarray(times_us, np.float64).reshape(-1)
    out = np.zeros((len(t), BEAT_FEATURE_DIM), np.float32)
    seg = segment_array(segments)
    if seg is None or not len(t):
        return out
    starts, beat, meter = seg[:, 0], seg[:, 1], seg[:, 2]
    i = np.clip(np.searchsorted(starts, t, 'right')-1, 0, len(seg)-1)
    b = beat[i]
    x = (t-starts[i])/b
    two_pi = 2*np.pi
    columns = []
    for angle in (two_pi*x, two_pi*x/meter[i], 2*two_pi*x, 4*two_pi*x):
        columns += [np.sin(angle), np.cos(angle)]
    for k in (4, 3):
        offset_ms = (x*k-np.round(x*k))*b/k/1000.
        columns.append(np.clip(offset_ms, -OFFSET_CLIP_MS, OFFSET_CLIP_MS)/OFFSET_CLIP_MS)
    columns.append(np.clip(np.log(b/500_000.), -LOG_BEAT_CLIP, LOG_BEAT_CLIP))
    columns.append(np.ones(len(t)))
    out[:] = np.stack(columns, 1)
    return out


def window_beat_features(inputs, start_us, segments):
    """``beat`` [T_mel, F] for one request's inputs, aligned to ``mel_times`` (s, relative to start_us)."""
    times = np.asarray(inputs['mel_times'], np.float64)*1e6+float(start_us)
    beat = beat_features(times, segments)
    if 'mel_valid' in inputs:
        beat[~np.asarray(inputs['mel_valid'], bool)] = 0
    return beat
