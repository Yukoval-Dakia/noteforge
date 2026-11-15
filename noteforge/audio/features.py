"""Frozen audio feature extraction and its resumable on-disk receipt.

The model reads MERT-v2 layers 12/23 and the complete-tail 10 ms mel on their
native physical clocks. Corpus preparation and training pipelines are private.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from noteforge.audio.local_mel_cache import complete_song_mel

SCHEMA = 'noteforge.audio-features.v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def existing_feature_receipt(folder, encoder):
    """Validate a previously committed feature folder without rewriting it."""
    receipt = folder / 'receipt.json'
    if not receipt.exists():
        return None
    value = json.loads(receipt.read_text())
    if value['metadata']['revision'] != encoder.revision:
        raise ValueError('Committed MERT revision differs')
    for name, info in value['files'].items():
        if digest(folder / name) != info['sha256']:
            raise ValueError('Committed feature changed')
    return value


def feature_payloads(wave, encoder, *, mel=None, phase_callback=None):
    """Frozen encoder and mel equations; both retain their native physical clocks."""
    if phase_callback is not None:
        phase_callback('gpu_encode_start')
    features = encoder.encode(wave)
    if phase_callback is not None:
        phase_callback('gpu_encode_end')
    if mel is None:
        mel = complete_song_mel(wave)
    expected = (len(wave) + 959) // 960
    if features['layers'].shape != (2, expected, 1024) or mel['mel'].shape != (expected * 4, 128):
        raise ValueError('MERT/mel physical clocks disagree')
    return features, mel, expected


def commit_feature(audio, original, output_root, temporary_root, features, mel, expected):
    folder = Path(output_root) / 'features' / audio
    folder.mkdir(parents=True, exist_ok=True)
    values = {'L12.npy': features['layers'][0], 'L23.npy': features['layers'][1],
              'times_s.npy': features['times_s'], 'intervals_s.npy': features['intervals_s'],
              'mel10_complete.npy': mel['mel']}
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite output ' + name)
        temporary = folder / (name + '.partial')
        with temporary.open('wb') as stream:
            np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        temporary.replace(folder / name)
    result = dict(schema=SCHEMA, audio_sha256=audio, decoded_samples=features['metadata']['real_samples'],
        frames=expected, metadata=features['metadata'], backend='cuda_bf16',
        files={name: dict(bytes=(folder / name).stat().st_size, sha256=digest(folder / name)) for name in values},
        wall_s=None, complete=True)
    write_json(folder / 'receipt.json', result)
    if original.is_relative_to(Path(temporary_root)):
        original.unlink()
    return result
