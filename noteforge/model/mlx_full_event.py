"""MLX (Apple-silicon GPU) mirror of ``FullEventModel`` eval-time encode/decode.

The torch model stays the authority for weights, config and grammar; this adapter
re-implements ``encode`` / ``start_static_cache`` / ``decode_static`` with
``mlx.core`` ops so one decode step is a single compiled GPU graph instead of
~500 tiny MPS dispatches (Torch 2.6 MPS ~80 ms/step on a 250M model at batch 8).

Semantic deltas vs ``FullEventModel.decode_static`` (same math, different layout):

- Self-attention K/V is a paged cache: pages of ``PAGE`` positions are appended
  to a flat ``hist[B,kv,H,hd]`` every ``PAGE`` steps; the current page lives in
  ``cur[B,kv,PAGE,hd]`` and is updated functionally inside the compiled step
  (``mx.where`` on the slot index). The attended set equals
  ``slots[:position+1]`` of the torch static cache: unwritten slots are masked
  by ``pos`` and get exactly zero weight.
- ``expand_memory`` marks ``memory['_dup']`` when ``index`` is exactly
  ``repeat_interleave(arange(U), dup)`` (seed samples sharing one core memory).
  ``decode_static`` then keeps cross-attention K/V at the U unique rows and folds
  the dup axis into the GQA group axis -- identical attention sets, ~dup x less
  memory-K/V traffic. Rows must be ordered (unique, dup)-major.
- The step is one ``mx.compile``d function; a new page count / batch shape /
  memory length retraces automatically (one-time ~0.1-0.3 s per new geometry,
  reused for every later call in the process).

``fast_supported(model)`` holds: ``.c``, ``.vocab``, ``decode_static``,
``start_static_cache`` are exposed; ``decode``/``start_cache`` (reference path)
delegate to the wrapped torch model against the torch-side memory.
"""
import math

import numpy as np
import torch

from .full_event_model import BEAT_FEATURE_DIM

PAGE = 64

DTYPES = {'fp16': 'float16', 'bf16': 'bfloat16', 'fp32': 'float32'}


def available():
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


def _mx():
    import mlx.core as mx
    return mx


def _rope_scale(lo, hi, half):
    """``2*pi/periods`` computed BY torch, bit-identical to ``rotary_tables``.

    Rope angles reach ~1e9 rad at microsecond times, where a 1-ULP difference
    in the fp32 scale amplifies into ~100 rad of phase error. Torch's
    scalar/tensor division rounds differently from numpy fp32 (and from fp64),
    so the scale is produced by torch itself -- same ``torch.logspace`` and the
    same ``2 * math.pi / periods`` expression ``rotary_tables`` evaluates.
    """
    periods = torch.logspace(math.log10(lo), math.log10(hi), half)
    return (2 * math.pi / periods).numpy().astype(np.float32)


def _rope_tables_np(positions, scale, timed=None):
    """fp32 (cos, sin) [B,1,N,half] numpy tables matching ``rotary_tables``.

    ``positions*scale`` and the optional ``*timed`` are IEEE fp32 multiplies --
    bit-identical to torch. ``np.cos`` then reduces the (~1e9 rad) argument in
    fp64, where MLX GPU's fast trig loses up to ~1 rad; result cast back to
    fp32 agrees with torch's libm cos to ~1 ULP.
    """
    angle = positions.astype(np.float32)[:, None, :, None] * np.asarray(scale, np.float32)
    if timed is not None:
        angle = angle * timed.astype(np.float32)[:, None, :, None]
    cos = np.cos(angle.astype(np.float64)).astype(np.float32)
    sin = np.sin(angle.astype(np.float64)).astype(np.float32)
    return cos, sin


def _layer_norm_last(x, eps=1e-5):
    """F.layer_norm(x.float(), (x.shape[-1],)) with no affine (biased variance)."""
    mx = _mx()
    x = x.astype(mx.float32)
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps)


def _rms_norm(x, weight):
    """FullEventModel.RMSNorm: fp32 variance, cast back, affine weight.

    ``mx.fast.rms_norm`` is one fused Metal kernel (fp32 accumulation) and
    verified within fp16 tolerance of the torch expression."""
    return _mx().fast.rms_norm(x, weight, 1e-6)


def _rope_tables(positions, scale, timed=None):
    """MLX arrays of rope (cos, sin); ``positions``/``timed`` numpy or mx."""
    cos, sin = _rope_tables_np(np.asarray(positions, np.float32), scale,
                               None if timed is None else np.asarray(timed, np.float32))
    mx = _mx()
    return mx.array(cos), mx.array(sin)


def _apply_rope(x, cos, sin, inverse=False):
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    cos, sin = cos.astype(x.dtype), -sin.astype(x.dtype) if inverse else sin.astype(x.dtype)
    mx = _mx()
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin], -1)


def _rotate(x, positions, *, lo, hi, timed=None):
    """rotate(x, positions, lo, hi, timed): full-width rope (half = dim//2)."""
    cos, sin = _rope_tables(positions, _rope_scale(lo, hi, x.shape[-1] // 2), timed)
    return _apply_rope(x, cos, sin)


def _silu(x):
    mx = _mx()
    return x * mx.sigmoid(x)


def _sdpa(q, k, v, mask, width):
    """aligned_attention's zero-pad + SDPA + slice in MLX."""
    mx = _mx()
    common = max(width, v.shape[-1])
    common += (-common) % 8
    if common != q.shape[-1]:
        q = mx.concatenate([q, mx.zeros((*q.shape[:-1], common - q.shape[-1]), q.dtype)], -1)
        k = mx.concatenate([k, mx.zeros((*k.shape[:-1], common - k.shape[-1]), k.dtype)], -1)
    if common != v.shape[-1]:
        v = mx.concatenate([v, mx.zeros((*v.shape[:-1], common - v.shape[-1]), v.dtype)], -1)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=width ** -.5, mask=mask)
    return y[..., :width]


def _expand(memory, index):
    """Torch-side memory row expansion used when the model has no expand_memory."""
    rows = len(memory['mode'])
    out = {}
    for key, value in memory.items():
        out[key] = value[index] if torch.is_tensor(value) and value.shape[:1] == (rows,) else value
    return out


class MlxFullEventModel:
    """Eval-time MLX mirror of a FullEventModel (attention history mixer)."""

    def __init__(self, model, dtype='fp16', release_torch=False):
        if model.training:
            raise ValueError('MLX decoding requires eval mode')
        if model.c.history_mixer != 'attention':
            raise ValueError('MLX mirror implements the attention history mixer only')
        self._m = _mx()
        self.torch_model = model
        self.c, self.vocab = model.c, model.vocab
        self.training = False
        self.dtype = getattr(self._m, DTYPES[dtype])
        self.w = self._convert(model)
        self._compiled = None
        hd = self.c.d // self.c.heads
        self._tables = dict(time_half=(.02, 2400, hd // 4), pos_half=(2, 65536, hd // 4),
                            time_full=(.02, 2400, hd // 2))
        self._rope_scales = {k: _rope_scale(*v) for k, v in self._tables.items()}
        self._released = bool(release_torch)
        # Rollouts may downcast large float inputs to fp16 under this flag:
        # the mirror is dtype-fp16 anyway and the torch reference path is gone.
        self.fp16_inputs = self._released
        if release_torch:
            self._release_torch(model)

    @staticmethod
    def _release_torch(model):
        """Drop the fp32 torch weights the MLX path never reads (~1 GB RSS).

        Everything MLX decode needs already lives in ``self.w``. The torch head
        (``final_norm`` + ``output`` + onset head) is kept so hooked-output
        replay in ``decode_static`` still works; the reference ``decode`` /
        ``start_cache`` path is not usable afterwards (the MLX mirror only
        serves the static-cache generation path anyway).
        """
        keep = ('final_norm.', 'output.', 'onset_norm.', 'onset_projection.')
        for name, p in model.named_parameters():
            if not name.startswith(keep):
                p.data = torch.empty(0)
        for name, b in model.named_buffers():
            b.data = torch.empty(0)
        import gc
        gc.collect()

    # ------------------------------------------------------------------ weights
    def _convert(self, model):
        mx = self._m
        sd = model.state_dict()

        def tensor(name, dtype=None):
            return mx.array(sd[name].detach().to(torch.float32).cpu().numpy()).astype(dtype or self.dtype)

        def affine(name):
            weight = tensor(name + '.weight')
            bias = tensor(name + '.bias') if name + '.bias' in sd else None
            return dict(w=weight, b=bias)

        def attention(prefix):
            qkv = mx.concatenate([tensor(prefix + 'q.weight'), tensor(prefix + 'k.weight'),
                                  tensor(prefix + 'v.weight')], 0)
            out = dict(qkv=qkv, out=tensor(prefix + 'out.weight'))
            if self.c.qk_norm:
                out.update(q_norm=tensor(prefix + 'q_norm.weight'),
                           k_norm=tensor(prefix + 'k_norm.weight'))
            return out

        def block(prefix):
            return dict(norm1=tensor(prefix + 'norm1.weight'), norm2=tensor(prefix + 'norm2.weight'),
                        attn=attention(prefix + 'attn.'), ffn_up=tensor(prefix + 'ffn.up.weight'),
                        ffn_down=tensor(prefix + 'ffn.down.weight'))

        c = self.c
        w = dict(layer_mix=tensor('layer_mix', self._m.float32),
                 local_projection=affine('local_projection'), band_embedding=tensor('band_embedding'),
                 band_queries=tensor('band_queries'), song_projection=affine('song_projection'),
                 reference_projection=affine('reference_projection'),
                 control_projection=affine('control_projection'),
                 control_ids=tensor('control_ids.weight'), empty_reference=tensor('empty_reference'),
                 source_ids=tensor('source_ids.weight'), mode_ids=tensor('mode_ids.weight'),
                 token_embedding=tensor('token_embedding.weight'),
                 field_embedding=tensor('field_embedding.weight'),
                 encoders=[[block(f'encoders.{s}.{i}.') for i in range(c.encoder_layers)] for s in range(3)],
                 final_norm=tensor('final_norm.weight'), output=tensor('output.weight'))
        if c.learnable_audio_scale:
            w.update(local_input_scale=tensor('local_input_scale'),
                     song_input_scale=tensor('song_input_scale'))
        if c.mel_bins:
            w.update(mel_projection=affine('mel_projection'), mel_source=tensor('mel_source'))
        if c.intensity_dim:
            w.update(intensity_projection=affine('intensity_projection'),
                     intensity_source=tensor('intensity_source'))
            if c.timed_intensity:
                w['intensity_unspecified'] = tensor('intensity_unspecified')
        if c.onset_head:
            w.update(onset_norm=tensor('onset_norm.weight'), onset_projection=affine('onset_projection'))
        if c.beat_features:
            w['beat_projection'] = affine('beat_projection')
        decoder = []
        for i in range(c.decoder_layers):
            p = f'decoder.{i}.'
            layer = dict(norm1=tensor(p + 'norm1.weight'), norm2=tensor(p + 'norm2.weight'),
                         norm3=tensor(p + 'norm3.weight'), ffn_up=tensor(p + 'ffn.up.weight'),
                         ffn_down=tensor(p + 'ffn.down.weight'),
                         self_attn=attention(p + 'self_attn.'),
                         cross_q=tensor(p + 'cross_attn.q.weight'),
                         cross_kv=mx.concatenate([tensor(p + 'cross_attn.k.weight'),
                                                  tensor(p + 'cross_attn.v.weight')], 0),
                         cross_out=tensor(p + 'cross_attn.out.weight'))
            if c.qk_norm:
                layer.update(cross_q_norm=tensor(p + 'cross_attn.q_norm.weight'),
                             cross_k_norm=tensor(p + 'cross_attn.k_norm.weight'))
            decoder.append(layer)
        w['decoder'] = decoder
        return w

    def __getattr__(self, name):
        # Anything not mirrored (decode/start_cache reference path, parameters(),
        # output.weight, mode_ids, ...) delegates to the wrapped torch model.
        if name in ('torch_model', '_m', 'w', '_compiled', '_rope_scales', '_tables'):
            raise AttributeError(name)
        return getattr(self.torch_model, name)

    # ------------------------------------------------------------------ helpers
    def _linear(self, wb, x):
        y = x @ wb['w'].T
        return y + wb['b'] if wb['b'] is not None else y

    def _enc_block(self, w, x, time_cos, time_sin, valid):
        mx = self._m
        b, n, d = x.shape
        z = _rms_norm(x, w['norm1'])
        qkv = z @ w['attn']['qkv'].T
        hd, kv, heads = d // self.c.heads, self.c.kv_heads, self.c.heads
        kvd = kv * hd
        q = qkv[..., :d].reshape(b, n, heads, hd).transpose(0, 2, 1, 3)
        k = qkv[..., d:d + kvd].reshape(b, n, kv, hd).transpose(0, 2, 1, 3)
        v = qkv[..., d + kvd:].reshape(b, n, kv, hd).transpose(0, 2, 1, 3)
        if 'q_norm' in w['attn']:
            q = _rms_norm(q, w['attn']['q_norm'])
            k = _rms_norm(k, w['attn']['k_norm'])
        q = _apply_rope(q, time_cos, time_sin)
        k = _apply_rope(k, time_cos, time_sin)
        y = _sdpa(q, k, v, valid[:, None, None, :], hd)
        y = y.transpose(0, 2, 1, 3).reshape(b, n, d)
        x = x + y @ w['attn']['out'].T
        z = _rms_norm(x, w['norm2'])
        up = z @ w['ffn_up'].T
        half = up.shape[-1] // 2
        x = x + (_silu(up[..., :half]) * up[..., half:]) @ w['ffn_down'].T
        return x * valid[..., None].astype(x.dtype)

    def _cross_kv(self, w, values, time_cos, time_sin, timed):
        """decoder.cross_attn.project_kv(x, t, positions, timed) for one layer."""
        mx = self._m
        b, m, d = values.shape
        hd, kv = d // self.c.heads, self.c.kv_heads
        kvd = kv * hd
        kv_pair = values @ w['cross_kv'].T
        k = kv_pair[..., :kvd].reshape(b, m, kv, hd).transpose(0, 2, 1, 3)
        v = kv_pair[..., kvd:].reshape(b, m, kv, hd).transpose(0, 2, 1, 3)
        if 'cross_k_norm' in w:
            k = _rms_norm(k, w['cross_k_norm'])
        if self.c.timed_memory_values:
            weight = timed.astype(v.dtype)[:, None, :, None]
            v = mx.concatenate(
                [_apply_rope(v, time_cos, time_sin) * weight, v * (1 - weight)], -1)
        k = _apply_rope(k, time_cos, time_sin)
        return k, v

    def _project_controls(self, controls):
        return (self._linear(self.w['control_projection'], controls)
                + self.w['control_ids'][None] + self.w['source_ids'][3])

    # ------------------------------------------------------------------ encode
    def encode(self, inputs):
        """Same contract as FullEventModel.encode; torch-side memory + ``_mlx`` payload."""
        mx = self._m
        c, w = self.c, self.w

        def arr(key):
            return mx.array(inputs[key].cpu().numpy())

        def npf(key):
            return inputs[key].cpu().numpy().astype(np.float32)

        local = arr('local')
        if local.ndim != 5 or local.shape[1] != c.taps or tuple(local.shape[3:]) != (c.bands, c.tap_dim):
            raise ValueError('Local frontend tensor contract mismatch')
        b, _, n = local.shape[:3]
        normalized = _layer_norm_last(local).astype(self.dtype)
        if c.learnable_audio_scale:
            normalized = normalized * w['local_input_scale'][None, :, None, None, :]
        mixed = (normalized * mx.softmax(w['layer_mix'])[None, :, None, None, None]).sum(1)
        band = self._linear(w['local_projection'], mixed) + w['band_embedding']
        weights = mx.softmax(mx.einsum('btfd,qd->btqf', band, w['band_queries']).astype(mx.float32)
                             / math.sqrt(c.d), -1).astype(band.dtype)
        local_tokens = mx.einsum('btqf,btfd->btqd', weights, band).reshape(b, n * c.band_slots, c.d)
        local_times = mx.repeat(arr('local_times'), c.band_slots, axis=1)
        local_times_np = np.repeat(npf('local_times'), c.band_slots, 1)
        local_valid = mx.repeat(arr('local_valid'), c.band_slots, axis=1)
        if c.mel_bins:
            mel = arr('mel')
            if mel.ndim != 3 or mel.shape[-1] != c.mel_bins:
                raise ValueError('10ms mel tensor contract mismatch')
            detail = self._linear(w['mel_projection'], mel) + w['mel_source']
            if c.beat_features:
                detail = detail + self._linear(w['beat_projection'], self._beat_inputs(inputs, mel))
            local_tokens = mx.concatenate([local_tokens, detail], 1)
            local_times = mx.concatenate([local_times, arr('mel_times')], 1)
            local_times_np = np.concatenate([local_times_np, npf('mel_times')], 1)
            local_valid = mx.concatenate([local_valid, arr('mel_valid')], 1)
        curve_tokens = curve_specified = curve_attend = None
        if c.intensity_dim:
            intensity = arr('intensity')
            if intensity.ndim != 3 or intensity.shape[-1] != c.intensity_dim:
                raise ValueError('Intensity curve tensor contract mismatch')
            curve_tokens = (self._linear(w['intensity_projection'], intensity)
                            + w['intensity_source'] + w['source_ids'][3])
            curve_specified = arr('intensity_valid')
            if c.timed_intensity:
                present = inputs.get('intensity_present')
                curve_attend = (curve_specified if present is None else
                                mx.array(present.cpu().numpy()).astype(mx.bool_) | curve_specified)
                curve_tokens = curve_tokens + (~curve_specified)[..., None].astype(
                    curve_tokens.dtype) * w['intensity_unspecified']
        song = _layer_norm_last(arr('song')).astype(self.dtype)
        if c.learnable_audio_scale:
            song = song * w['song_input_scale']
        song = self._linear(w['song_projection'], song)
        reference = self._linear(w['reference_projection'], arr('reference'))
        reference = mx.concatenate([mx.broadcast_to(w['empty_reference'], (b, 1, c.d)), reference], 1)
        rt = mx.concatenate([mx.zeros((b, 1), mx.float32), arr('reference_times')], 1)
        rv = mx.concatenate([mx.ones((b, 1), mx.bool_), arr('reference_valid')], 1)
        streams = [(local_tokens, local_times, local_times_np, local_valid),
                   (song, arr('song_times'), npf('song_times'), arr('song_valid')),
                   (reference, rt,
                    np.concatenate([np.zeros((b, 1), np.float32), npf('reference_times')], 1), rv)]
        memory, times, valid, timed = [], [], [], []
        onset = {}
        for i, (x, t, t_np, mask) in enumerate(streams):
            x = x + w['source_ids'][i]
            time_cos, time_sin = _rope_tables(t_np, self._rope_scales['time_full'])
            for enc in w['encoders'][i]:
                x = self._enc_block(enc, x, time_cos, time_sin, mask)
            memory.append(x); times.append(t); valid.append(mask)
            if i == 0 and c.onset_head:
                onset = self._onset_outputs(x, inputs, n)
            if i == 2:
                mask = mx.concatenate([mx.zeros((b, 1), mx.bool_), mask[:, 1:]], 1)
            timed.append(mask)
        controls = self._project_controls(arr('controls'))
        if curve_tokens is not None and c.timed_intensity:
            memory.append(curve_tokens)
            times.append(arr('intensity_times').astype(mx.float32) * curve_specified)
            valid.append(curve_attend)
            timed.append(curve_specified & curve_attend)
            curve_tokens = None
        condition_valid = mx.ones((b, c.controls), mx.bool_)
        if curve_tokens is not None:
            controls = mx.concatenate([curve_tokens, controls], 1)
            condition_valid = mx.concatenate([curve_specified, condition_valid], 1)
        memory.append(controls)
        times.append(mx.zeros((b, controls.shape[1]), mx.float32))
        valid.append(condition_valid)
        timed.append(mx.zeros((b, controls.shape[1]), mx.bool_))
        payload = dict(values=mx.concatenate(memory, 1), times=mx.concatenate(times, 1),
                       valid=mx.concatenate(valid, 1), timed=mx.concatenate(timed, 1),
                       mode=arr('mode').astype(mx.int32))
        mx.eval(*payload.values())
        result = {k: torch.from_numpy(np.asarray(v)) for k, v in payload.items()}
        # The torch side feeds the reference decode fallback and _onset_tables:
        # keep the original long mode ids. With the torch body released the
        # fp32 values copy is dead weight (~0.5 GB RSS at large row batches) --
        # fp16 halves it and nothing on the surviving path reads it back.
        result['values'] = result['values'].half() if self._released else result['values'].float()
        result['mode'] = inputs['mode']
        result['_mlx'] = payload
        result.update(onset)
        return result

    def _beat_inputs(self, inputs, mel):
        beat = inputs.get('beat')
        b, t = mel.shape[:2]
        if beat is None:
            return self._m.zeros((b, t, BEAT_FEATURE_DIM), mel.dtype)
        beat = self._m.array(beat.cpu().numpy()).astype(mel.dtype)
        if beat.ndim != 3 or tuple(beat.shape[:2]) != (b, t) or beat.shape[-1] != BEAT_FEATURE_DIM:
            raise ValueError('Beat feature tensor contract mismatch')
        if inputs.get('beat_valid') is not None:
            beat = beat * self._m.array(inputs['beat_valid'].cpu().numpy()).astype(beat.dtype)[..., None]
        return beat

    def _onset_outputs(self, encoded_local, inputs, frames):
        mx = self._m
        c = self.c

        def arr(key):
            return mx.array(inputs[key].cpu().numpy())

        if c.mel_bins:
            x = encoded_local[:, frames * c.band_slots:]
            times, valid = arr('mel_times'), arr('mel_valid')
        else:
            x = encoded_local[:, :frames * c.band_slots].reshape(
                encoded_local.shape[0], frames, c.band_slots, c.d).mean(2)
            times, valid = arr('local_times'), arr('local_valid')
        logits = self._linear(self.w['onset_projection'], _rms_norm(x, self.w['onset_norm']))
        return dict(onset_logits=torch.from_numpy(np.asarray(logits.astype(mx.float32))),
                    onset_times=torch.from_numpy(np.asarray(times)),
                    onset_valid=torch.from_numpy(np.asarray(valid)))

    # -------------------------------------------------------------- memory rows
    def expand_memory(self, memory, index):
        """Row-expand ``memory`` by ``index`` (torch long [B] into U rows).

        When ``index`` is exactly ``repeat_interleave(arange(U), dup)`` the mlx
        payload stays at U rows and ``_dup=dup`` lets ``decode_static`` fold the
        dup axis into the GQA group axis; otherwise the payload is expanded too.
        """
        mx = self._m
        rows = len(memory['mode'])
        index = torch.as_tensor(index, dtype=torch.long)
        out = _expand(memory, index)
        count = len(index)
        dup = None
        if count % rows == 0:
            d = count // rows
            if index.tolist() == [u for u in range(rows) for _ in range(d)]:
                dup = d
        payload = memory['_mlx']
        if dup is None:
            dup = 1
            mi = mx.array(index.cpu().numpy(), mx.int32)
            payload = {k: v[mi] for k, v in payload.items()}
        out['_mlx'] = payload
        out['_dup'] = dup
        return out

    # ------------------------------------------------------------- static cache
    def start_static_cache(self, memory, capacity):
        """Paged KV cache for ``decode_static`` (eval, attention mixer)."""
        mx = self._m
        c = self.c
        if capacity < 1:
            raise ValueError('Static decoding requires eval mode and the attention history mixer')
        payload = memory['_mlx']
        dup = int(memory.get('_dup', 1))
        u = len(memory['mode']) // dup
        values, times, timed, valid = payload['values'], payload['times'], payload['timed'], payload['valid']
        if values.shape[0] != u:
            raise ValueError('Expanded memory/_dup mismatch')
        times_np = memory['times'][:u].cpu().numpy().astype(np.float32)
        timed_np = memory['timed'][:u].cpu().numpy()
        time_cos, time_sin = _rope_tables(times_np, self._rope_scales['time_full'], timed_np)
        mem_k, mem_v = [], []
        for layer in self.w['decoder']:
            k, v = self._cross_kv(layer, values, time_cos, time_sin, timed)
            mem_k.append(k); mem_v.append(v)
        hd, kv = c.d // c.heads, c.kv_heads
        b = u * dup
        zeros = lambda *shape: mx.zeros(shape, self.dtype)
        mode_emb = self.w['mode_ids'][payload['mode'].astype(mx.int32)][:, None]
        if dup > 1:
            mode_emb = mx.repeat(mode_emb, dup, axis=0)
        return dict(batch_size=b, capacity=int(capacity), dup=dup, committed=0,
                    hist_k=[zeros(b, kv, 0, hd) for _ in self.w['decoder']],
                    hist_v=[zeros(b, kv, 0, hd) for _ in self.w['decoder']],
                    cur_k=[zeros(b, kv, PAGE, hd) for _ in self.w['decoder']],
                    cur_v=[zeros(b, kv, PAGE, hd) for _ in self.w['decoder']],
                    mem_k=mem_k, mem_v=mem_v,
                    mem_mask=valid[:, None, None, :], mode_embedding=mode_emb)

    # --------------------------------------------------------------- decode step
    def _seq_rope(self, x, time_cos, time_sin, pos_cos, pos_sin):
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        return self._m.concatenate([_apply_rope(a, time_cos, time_sin),
                                    _apply_rope(b, pos_cos, pos_sin)], -1)

    def _step(self, tokens, fields, pos, th, ps, tf, hist_k, hist_v, cur_k, cur_v,
              mem_k, mem_v, mem_mask, mode_emb):
        """One decode step; the only caller-compiled function (shape-traced).

        ``th``/``ps``/``tf`` are precomputed (cos, sin) rope table pairs -- see
        ``_rope_tables_np`` for why they are built on the CPU side per step.
        Self-attention scores hist and the current page separately instead of
        concatenating the whole K/V history each step (one O(history) copy
        less per layer per step). q/k share one rope call by head-concat.
        """
        mx = self._m
        c, w = self.c, self.w
        b = tokens.shape[0]
        u, dup = mem_mask.shape[0], b // mem_mask.shape[0]
        hd, kv, heads = c.d // c.heads, c.kv_heads, c.heads
        group = heads // kv
        x = w['token_embedding'][tokens] + w['field_embedding'][fields] + mode_emb
        slot = pos % PAGE
        slot_match = mx.arange(PAGE)[None, None, :, None] == slot
        h = hist_k[0].shape[2]
        keep = mx.arange(h + PAGE)[None, None, None, :] <= pos
        new_cur_k, new_cur_v = [], []
        for i, lw in enumerate(w['decoder']):
            z = _rms_norm(x, lw['norm1'])
            qkv = z @ lw['self_attn']['qkv'].T
            kvd = kv * hd
            q = qkv[..., :c.d].reshape(b, heads, 1, hd)
            k = qkv[..., c.d:c.d + kvd].reshape(b, kv, 1, hd)
            v = qkv[..., c.d + kvd:].reshape(b, kv, 1, hd)
            if 'q_norm' in lw['self_attn']:
                q = _rms_norm(q, lw['self_attn']['q_norm'])
                k = _rms_norm(k, lw['self_attn']['k_norm'])
            qk = self._seq_rope(mx.concatenate([q, k], 1), *th, *ps)
            q, k = qk[:, :heads], qk[:, heads:]
            cur_k_i = mx.where(slot_match, k, cur_k[i])
            cur_v_i = mx.where(slot_match, v, cur_v[i])
            if b * (h + PAGE) <= 16384:
                # Latency-bound regime: one fused attention kernel.
                y = _sdpa(q, mx.concatenate([hist_k[i], cur_k_i], 2),
                          mx.concatenate([hist_v[i], cur_v_i], 2), keep, hd)
                y = y.reshape(b, 1, c.d)
            else:
                # Large rows*history: score hist/cur separately so the full K/V
                # history is not re-materialized every step.
                qg = q.reshape(b, kv, group, hd)
                scores = (mx.concatenate([qg @ hist_k[i].transpose(0, 1, 3, 2),
                                          qg @ cur_k_i.transpose(0, 1, 3, 2)], -1)
                          .astype(mx.float32) * (hd ** -.5))
                p = mx.softmax(mx.where(keep, scores, -mx.inf), -1).astype(self.dtype)
                y = (p[..., :h] @ hist_v[i] + p[..., h:] @ cur_v_i
                     if h else p @ cur_v_i)
                y = y.reshape(b, 1, c.d)
            x = x + y @ lw['self_attn']['out'].T
            z = _rms_norm(x, lw['norm2'])
            qc = (z @ lw['cross_q'].T).reshape(b, heads, 1, hd)
            if 'cross_q_norm' in lw:
                qc = _rms_norm(qc, lw['cross_q_norm'])
            qc = _apply_rope(qc, *tf)
            qc = qc.reshape(u, dup, kv, group, hd).transpose(0, 2, 1, 3, 4).reshape(u, kv, dup * group, hd)
            yc = _sdpa(qc, mem_k[i], mem_v[i], mem_mask, hd)
            yc = yc.reshape(u, kv, dup, group, -1).transpose(0, 2, 1, 3, 4).reshape(b, heads, 1, -1)
            if c.timed_memory_values:
                timed_part, untimed_part = yc[..., :hd], yc[..., hd:]
                yc = _apply_rope(timed_part, *tf, inverse=True) + untimed_part
            x = x + yc.transpose(0, 2, 1, 3).reshape(b, 1, c.d) @ lw['cross_out'].T
            z = _rms_norm(x, lw['norm3'])
            up = z @ lw['ffn_up'].T
            half = up.shape[-1] // 2
            x = x + (_silu(up[..., :half]) * up[..., half:]) @ lw['ffn_down'].T
            new_cur_k.append(cur_k_i)
            new_cur_v.append(cur_v_i)
        logits = _rms_norm(x, w['final_norm']) @ w['output'].T
        return logits, x, new_cur_k, new_cur_v

    def decode_static(self, tokens, fields, times, cache, position, length):
        """decode_static drop-in: torch tensors in, torch logits [B,1,V] out."""
        mx = self._m
        if self.training or tokens.shape != (cache['batch_size'], 1) or not 0 < length <= cache['capacity']:
            raise ValueError('Static decode needs eval mode, one query per row and length <= capacity')
        pos = int(position.reshape(-1)[0].item())
        if pos >= cache['capacity']:
            raise ValueError('Static decode position exceeds cache capacity')
        while (cache['committed'] + 1) * PAGE <= pos:
            for i in range(len(cache['hist_k'])):
                cache['hist_k'][i] = mx.concatenate([cache['hist_k'][i], cache['cur_k'][i]], 2)
                cache['hist_v'][i] = mx.concatenate([cache['hist_v'][i], cache['cur_v'][i]], 2)
                cache['cur_k'][i] = mx.zeros_like(cache['cur_k'][i])
                cache['cur_v'][i] = mx.zeros_like(cache['cur_v'][i])
            cache['committed'] += 1
        if self._compiled is None:
            self._compiled = mx.compile(self._step)
        b = cache['batch_size']
        times_np = times.cpu().numpy().astype(np.float32)
        pos_np = np.broadcast_to(np.array([[pos]], np.float32), (b, 1))
        scales = self._rope_scales
        rope = [_rope_tables_np(times_np, scales['time_half']),
                _rope_tables_np(pos_np, scales['pos_half']),
                _rope_tables_np(times_np, scales['time_full'])]
        rope = [mx.array(t).astype(self.dtype) for pair in rope for t in pair]
        logits, x, new_cur_k, new_cur_v = self._compiled(
            mx.array(tokens.cpu().numpy(), mx.int32), mx.array(fields.cpu().numpy(), mx.int32),
            mx.array([pos], mx.int32),
            (rope[0], rope[1]), (rope[2], rope[3]), (rope[4], rope[5]),
            cache['hist_k'], cache['hist_v'], cache['cur_k'], cache['cur_v'],
            cache['mem_k'], cache['mem_v'], cache['mem_mask'], cache['mode_embedding'])
        mx.eval(logits, *new_cur_k, *new_cur_v)
        cache['cur_k'], cache['cur_v'] = new_cur_k, new_cur_v
        head = self.torch_model.output
        patched = (head._forward_hooks or head._forward_pre_hooks
                   or getattr(head.forward, '__func__', head.forward) is not torch.nn.Linear.forward)
        if patched:
            # Tests hook ``model.output`` (e.g. logit biases); replay the head in
            # torch so those hooks observe the real (input, output) pair.
            normed = self.torch_model.final_norm(
                torch.from_numpy(np.asarray(x.astype(mx.float32))))
            return head(normed)
        return torch.from_numpy(np.asarray(logits.astype(mx.float32)))
