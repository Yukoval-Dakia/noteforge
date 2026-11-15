"""One-time export of the frozen model to fp16 arrays, and a loader that never builds fp32 weights.

    fast_model.py export CHECKPOINT OUT_DIR
"""
import json, sys
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import torch
from noteforge.model.mlx_full_event import MlxFullEventModel, _rope_scale


def export(checkpoint, out):
    from noteforge.scripts import draft_song as ds
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model, identity, declared = ds.load_model(checkpoint, 'cpu', 'auto')
    model.eval()
    keep = ('final_norm.', 'output.', 'onset_norm.', 'onset_projection.')
    torch.save({k: v for k, v in model.state_dict().items() if k.startswith(keep) and v.numel()}, out/'head.pt')
    torch.save(model.vocab, out/'vocab.pt')  # its grammar options come from the checkpoint, not the defaults
    mirror = MlxFullEventModel(model, 'fp16', release_torch=True)
    flat = dict(tree_flatten(mirror.w))
    mx.save_safetensors(str(out/'weights.safetensors'), flat)
    (out/'meta.json').write_text(json.dumps(dict(config=asdict(model.c), identity=identity, declared=declared), default=str))
    print('arrays', len(flat), 'dtypes', sorted({str(v.dtype) for v in flat.values()}), 'bytes', (out/'weights.safetensors').stat().st_size)


def load(folder):
    from noteforge.model.full_event_model import FullEventConfig, FullEventModel
    folder = Path(folder)
    meta = json.loads((folder/'meta.json').read_text())
    config = FullEventConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta['config'].items()})
    with torch.device('meta'):
        model = FullEventModel(config, torch.load(folder/'vocab.pt', weights_only=False))
    model.eval()
    state = torch.load(folder/'head.pt', map_location='cpu')
    missing = model.load_state_dict(state, strict=False, assign=True)
    if missing.unexpected_keys:
        raise ValueError(f'unexpected keys {missing.unexpected_keys[:3]}')
    self = MlxFullEventModel.__new__(MlxFullEventModel)
    self._m = mx
    self.torch_model = model
    self.c, self.vocab = model.c, model.vocab
    self.training = False
    self.dtype = mx.float16
    arrays = mx.load(str(folder/'weights.safetensors'))
    self.w = tree_unflatten(list(arrays.items()))
    self._audio_cache = {}
    self._compiled = self._step  # measured: mx.compile is no faster here and took 5.5 GB
    hd = self.c.d//self.c.heads
    self._tables = dict(time_half=(.02, 2400, hd//4), pos_half=(2, 65536, hd//4), time_full=(.02, 2400, hd//2))
    self._rope_scales = {k: _rope_scale(*v) for k, v in self._tables.items()}
    self._released = True
    self.fp16_inputs = True
    return self, meta['identity'], meta['declared']


if __name__ == '__main__':
    export(sys.argv[2], sys.argv[3])
