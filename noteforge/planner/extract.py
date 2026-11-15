"""Per-song 2 s audio features for the density planner (inference path).

``song_cells(audio_sha)`` reads a song's frozen feature cache
(``<FEATURES>/<audio_sha256>/{times_s,L23,mel10_complete}.npy``) and returns the
pooled MERT and mel/flux cells the planner consumes. ``FEATURES`` is the feature
root; ``plan_song.py`` sets it from ``--features-dir`` before calling.

The corpus-level extraction entry points (``main`` / ``targets_only``) that read
the corpus pipeline are not included here.
"""
from pathlib import Path

import numpy as np

FEATURES = Path('features')  # overridden by plan_song --features-dir
CELL_S = 2.0


def pool(values, times, n):
    index = np.minimum((times / CELL_S).astype(int), n - 1)
    total = np.zeros((n, values.shape[1]), np.float32)
    count = np.zeros(n, np.float32)
    np.add.at(total, index, values.astype(np.float32))
    np.add.at(count, index, 1)
    return total / np.maximum(count, 1)[:, None]


def song_cells(audio):
    d = FEATURES / audio
    times = np.load(d / 'times_s.npy')
    n = max(int(np.ceil(float(times[-1]) / CELL_S)), 1)
    mert = pool(np.load(d / 'L23.npy'), times, n)
    mel = np.load(d / 'mel10_complete.npy').astype(np.float32)
    bands = mel[:, :128 // 16 * 16].reshape(len(mel), 16, -1).mean(-1)
    mel_t = (np.arange(len(mel)) + .5) * .01
    flux = np.maximum(np.diff(bands, axis=0, prepend=bands[:1]), 0)
    energy = flux.sum(-1)
    peaks = (energy > energy.mean() + energy.std()) & (energy >= np.roll(energy, 1)) & (energy >= np.roll(energy, -1))
    extra = pool(np.stack([energy, peaks.astype(np.float32), bands.mean(-1)], 1), mel_t, n)
    return mert, np.concatenate([pool(bands, mel_t, n), pool(flux, mel_t, n), extra], 1)
