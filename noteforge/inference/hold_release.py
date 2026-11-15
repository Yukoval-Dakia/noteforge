"""Mania hold-release repair used when assembling an editable revision."""
from collections import defaultdict


def repair_hold_release(notes, mode, *, min_release_us=40_000, min_hold_us=60_000):
    """Shorten a hold before the next press of the same key; make very short holds taps."""
    if mode != 'mania_4k':
        return list(notes), dict(shortened=0, to_tap=0)
    notes = sorted(tuple(x) for x in notes)
    out, counts = list(notes), dict(shortened=0, to_tap=0)
    by_lane = defaultdict(list)
    for i, x in enumerate(notes):
        by_lane[x[3]].append(i)
    for idx in by_lane.values():
        for a, b in zip(idx, idx[1:]):
            on, end, typ, lane = out[a]
            nxt = out[b][0]
            if typ == 'hold' and nxt - end < min_release_us:
                new_end = nxt - min_release_us
                if new_end - on < min_hold_us:
                    out[a] = (on, on, 'tap', lane)
                    counts['to_tap'] += 1
                else:
                    out[a] = (on, new_end, typ, lane)
                    counts['shortened'] += 1
    return out, counts


def stable_rank(seed, *parts):
    import hashlib
    return hashlib.sha256(':'.join(map(str, (seed,)+parts)).encode()).hexdigest()
