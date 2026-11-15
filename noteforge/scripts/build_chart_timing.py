"""Parse a charter's timing points into model beat segments.

The inference path uses these functions for an uploaded chart.
"""
from fractions import Fraction

from noteforge.inference.beat_features import SCHEMA
from noteforge.scripts.runtime import sha_file


class GimmickTiming(ValueError):
    """Timing map with a stop/teleport BPM: no usable beat grid."""


MIN_BEAT_US, MAX_BEAT_US = 1_000.0, 60_000_000.0  # 60000 BPM .. 1 BPM


def timing_segments(raw):
    """Raw chart timing -> ([(start_us, beat_us, meter), ...], source) on the audio clock."""
    timing = raw.timing
    rows = (raw.sidecar or {}).get('timing_rows') if raw.source_format.startswith('osu') else None
    meters, meter_source = [4]*len(timing.segments), 'default_4'
    if rows and len(rows) == len(timing.segments):
        meters, meter_source = [max(1, int(r['meter'])) for r in rows], 'osu_timing_point'
    segments = []
    for seg, meter in zip(timing.segments, meters):
        start = float((Fraction(seg.start_sec)+Fraction(timing.offset_sec))*1_000_000)
        try:
            beat = float(Fraction(60_000_000)/Fraction(seg.bpm))
        except (OverflowError, ZeroDivisionError) as exc:
            raise GimmickTiming(f'unusable BPM {seg.bpm!r} at {seg.start_sec}s ({type(exc).__name__})') from exc
        if not (MIN_BEAT_US <= beat <= MAX_BEAT_US):
            raise GimmickTiming(f'beat {beat:.6g} us outside [{MIN_BEAT_US}, {MAX_BEAT_US}] at {seg.start_sec}s')
        if segments and abs(segments[-1][0]-start) < 1e-6:
            segments[-1] = [start, beat, meter]
        else:
            segments.append([start, beat, meter])
    return segments, meter_source
