"""Apply the frozen mel channel statistics published with the r5 model."""
import numpy as np


class MelNormalizedFeatures:
    def __init__(self, base, statistics, *, expected_sha256):
        if (statistics['schema'] != 'noteforge.inference-coefficients.v1'
                or statistics.get('coefficient_sha256') != expected_sha256):
            raise ValueError('Normalization coefficients do not match the published model contract')
        self.base = base
        self.mean = np.asarray(statistics['mean'], np.float32)
        self.std = np.asarray(statistics['std'], np.float32)
        if (self.mean.shape != (128,) or self.std.shape != (128,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError('Invalid mel normalization coefficients')

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def window(self, audio, start_us, end_us):
        row = dict(self.base.window(audio, start_us, end_us))
        values = (row['mel']-self.mean)/self.std
        values[~row['mel_valid']] = 0
        row['mel'] = values
        return row
