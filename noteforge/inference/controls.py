"""Encode caller-supplied condition bands into the model's fixed control table."""
from noteforge.events import controls as C

CONTROL_FIELDS = tuple(C.BAND_EDGES)
CONTROL_COLUMNS = ('lower', 'upper', 'specified', 'applicable', 'start', 'end')
TIME_SCALE_US = 1_200_000_000


def _scope(scope):
    if (len(scope) != 2 or any(type(v) is not int for v in scope)
            or not 0 <= scope[0] < scope[1] <= TIME_SCALE_US):
        raise ValueError('Use one nonempty request interval in integer microseconds')
    return tuple(scope)


def encode_controls(mode, bands, scope, *, origin_us):
    """Encode only caller-supplied conditions, including true zero and unknown."""
    scope = _scope(scope)
    if mode not in C.MODE_FIELDS or type(origin_us) is not int or not 0 <= origin_us <= TIME_SCALE_US:
        raise ValueError('Invalid mode or request clock origin')
    if not set(bands) <= set(C.MODE_FIELDS[mode]):
        raise ValueError('Unknown or inapplicable condition')
    import numpy as np
    out = np.zeros((len(CONTROL_FIELDS), len(CONTROL_COLUMNS)), np.float32)
    for i, field in enumerate(CONTROL_FIELDS):
        if field not in C.MODE_FIELDS[mode]:
            continue
        out[i, 3] = 1
        out[i, 4:] = [(t - origin_us) / TIME_SCALE_US for t in scope]
        if field in bands:
            band = bands[field]
            if not C.is_table_band(field, band):
                raise ValueError(f'Invalid frozen band for {field}')
            out[i, :2] = np.asarray(band) / C.BAND_EDGES[field][-1]
            out[i, 2] = 1
    return out
