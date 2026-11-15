"""Assemble frozen inference inputs without any target sequence."""
import numpy as np
import torch

STREAM_AXES = {'local': 2, 'mel': 1, 'song': 1, 'reference': 1, 'intensity': 1}


def collate(rows, *, input_shapes=None):
    """Stack one or more inference rows; keep every valid value unchanged."""
    if not rows:
        raise ValueError('Cannot collate an empty batch')
    input_shapes = input_shapes or {}
    if set(input_shapes) - set(STREAM_AXES):
        raise ValueError('Unknown fixed input stream')
    result = {}
    if any(set(row) != set(rows[0]) for row in rows):
        raise ValueError('Batch rows have different fields')
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in ('mode', 'controls'):
            joined = np.stack(values)
        elif key == 'local':
            actual = max(v.shape[1] for v in values)
            length = input_shapes.get('local', actual)
            if length < actual:
                raise ValueError('local exceeds the frozen input shape')
            joined = np.zeros((len(rows), values[0].shape[0], length,
                               *values[0].shape[2:]), values[0].dtype)
            for i, v in enumerate(values):
                joined[i, :, :len(v[0])] = v
        else:
            stream = key.removesuffix('_times').removesuffix('_valid')
            stream = 'mel' if stream == 'beat' else stream
            length = max(map(len, values))
            fixed = input_shapes.get(stream, length)
            if fixed < length or fixed < 1:
                raise ValueError(f'{key} exceeds the frozen input shape')
            length = fixed
            joined = np.zeros((len(rows), length, *values[0].shape[1:]), values[0].dtype)
            for i, v in enumerate(values):
                joined[i, :len(v)] = v
        result[key] = torch.from_numpy(joined)
    return result


def pad_input_shapes(inputs, lengths):
    """Pad dictionary tensors to the frozen stream lengths."""
    result = dict(inputs)
    for stream, length in lengths.items():
        if stream not in STREAM_AXES or length < 1:
            raise ValueError('Unknown stream or invalid fixed length')
        axis = STREAM_AXES[stream]
        keys = (stream, stream+'_times', stream+'_valid')
        if stream == 'mel':
            keys += tuple(k for k in ('beat', 'beat_valid') if k in inputs)
        for key in keys:
            value = inputs[key]
            key_axis = axis if key == stream else 1
            n = value.shape[key_axis]
            if n > length:
                raise ValueError(f'{key} has {n} entries, exceeding the frozen shape {length}')
            if n == length:
                continue
            shape = list(value.shape)
            shape[key_axis] = length
            padded = value.new_zeros(shape)
            index = [slice(None)]*value.ndim
            index[key_axis] = slice(0, n)
            padded[tuple(index)] = value
            result[key] = padded
    return result


def request_inputs(features, audio, request, controls, *, sources=()):
    """Audio, visible caller references, conditions and mode; no hidden target."""
    from noteforge.inference.reference import encode_reference
    if request.duration_us != features.records[audio]['duration_us']:
        raise ValueError('Request duration differs from the admitted sample clock')
    reference, provenance = encode_reference(
        mode=request.mode, audio_sha256=audio, duration_us=request.duration_us,
        origin_us=request.start_us, unknown_regions=((request.start_us, request.end_us),),
        sources=sources)
    inputs = dict(features.window(audio, request.start_us, request.end_us), **reference,
                  controls=np.asarray(controls, np.float32),
                  mode=np.asarray(0 if request.mode == 'mania_4k' else 1, np.int64))
    return inputs, provenance
