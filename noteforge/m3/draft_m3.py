"""The frozen draft path on Apple silicon: scripts.draft_song with the model swapped for the fp16 MLX mirror.

Environment: M3_FAST weights folder (required), M3_CACHE_MB, M3_STRIDE (absent: core by core; 2: two parallel rounds;
-2: a third round rewrites the cores written without history), M3_CONTINUE (JSON with events and keep_before_us).
Importable: engine_server.py keeps one loaded model for many calls."""
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
import mlx.core as mx
import torch
from noteforge.model import local_song_rollout
from noteforge.model.mlx_full_event import MlxFullEventModel
from noteforge.scripts import draft_song as ds
from noteforge.m3 import fast_model
from noteforge.m3 import rounds

_encode_all = MlxFullEventModel.encode
_sequential = local_song_rollout.generate_complete_song_samples


def encode_rows(self, inputs):
    """The encoder's self attention is M x M per row: encode rows one at a time, decode them together."""
    rows = len(inputs['mode'])
    if rows == 1:
        return _encode_all(self, inputs)
    parts = []
    for i in range(rows):
        one = {k: (v[i:i+1] if torch.is_tensor(v) and v.shape[:1] == (rows,) else v) for k, v in inputs.items()}
        parts.append(_encode_all(self, one))
        mx.clear_cache()
    out = {}
    for k in parts[0]:
        if k == '_mlx':
            out[k] = {n: mx.concatenate([p[k][n] for p in parts], 0) for n in parts[0][k]}
            mx.eval(*out[k].values())
        else:
            out[k] = torch.cat([p[k] for p in parts], 0)
    return out


MlxFullEventModel.encode = encode_rows
_loaded = {}


def load(path, device, weights):
    folder = os.environ['M3_FAST']
    if folder not in _loaded:
        _loaded[folder] = fast_model.load(folder)
    return _loaded[folder]


ds.load_model = load


def configure(env):
    """Choose the rollout for one call from its environment."""
    mx.set_cache_limit(int(env.get('M3_CACHE_MB', 300))*2**20)
    stride = env.get('M3_STRIDE')
    extra = {}
    if env.get('M3_CONTINUE'):
        given = json.loads(Path(env['M3_CONTINUE']).read_text())
        extra = dict(initial_events=given['events'], keep_before_us=given['keep_before_us'])
    if stride:
        local_song_rollout.generate_complete_song_samples = partial(rounds.generate_rounds, stride=abs(int(stride)), refine=int(stride) < 0, **extra)
    else:
        local_song_rollout.generate_complete_song_samples = _sequential


def run(argv, env=None):
    os.environ.update({k: str(v) for k, v in (env or {}).items()})
    configure(os.environ)
    return ds.main(list(argv))


if __name__ == '__main__':
    run(sys.argv[1:])
