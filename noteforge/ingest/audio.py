"""Audio decode helpers for inference.

``decode`` / ``probe_codec`` / ``codec_ext`` turn an audio file into the 24 kHz
mono float32 waveform the feature extractor consumes, and report its codec and
file extension. These are the only ingestion helpers the drafting path needs;
the corpus batch pipeline is part of the closed training code and is not here.
"""
from __future__ import annotations

import re
import subprocess

import numpy as np

SR = 24000
CODEC_EXT = {'mp3': 'mp3', 'vorbis': 'ogg', 'opus': 'opus', 'flac': 'flac', 'aac': 'm4a', 'alac': 'm4a',
             'wavpack': 'wv'}


def probe_codec(path: str) -> str:
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name',
                          '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=120)
    c = out.stdout.strip().split('\n')[0].split(',')[0].strip()
    return c or 'unknown'


def codec_ext(codec: str) -> str:
    if codec.startswith('pcm_'):
        return 'wav'
    return CODEC_EXT.get(codec, codec if re.fullmatch(r'[a-z0-9]{1,8}', codec) else 'bin')


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-vn', '-f', 'f32le', '-ac', '1',
                          '-ar', str(SR), 'pipe:1'], capture_output=True, check=True, timeout=1800).stdout
    return np.frombuffer(raw, dtype='<f4').copy()
