"""Write a deterministic synthetic feature fixture for backend benchmarks.

This exists so a backend can be measured without the 2.5 GB pinned MERT encoder or
a licensed song. The audio clock, tensor shapes and frame rates match the published
feature contract; the values are synthetic, so a benchmark run here measures the
runtime, never chart quality.

    python -m noteforge.scripts.make_bench_fixture --out ./bench-features --seconds 8
    python -m noteforge.scripts.bench_backends --backend torch --device cpu \
        --audio ./bench-features/silence.wav --features-dir ./bench-features \
        --mode phira --level 10

The fixture writes ``silence.wav`` (24 kHz mono) plus ``features/<sha>/`` with the
same files ``noteforge.audio.features.commit_feature`` produces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from noteforge.audio.mert2 import CELL_SAMPLES, REPO, REVISION, SR

SCHEMA = 'noteforge.audio-features.v1'


def write_wave(path, seconds):
    samples = int(round(seconds*SR))
    frames = np.zeros(samples, np.int16)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(frames.tobytes())
    return samples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    out = args.out
    features_root = out/'features'
    features_root.mkdir(parents=True, exist_ok=True)
    audio = out/'silence.wav'
    samples = write_wave(audio, args.seconds)
    audio_sha = hashlib.sha256(audio.read_bytes()).hexdigest()
    folder = features_root/audio_sha
    folder.mkdir(parents=True, exist_ok=True)
    frames = (samples+CELL_SAMPLES-1)//CELL_SAMPLES
    rng = np.random.default_rng(args.seed)
    times = (np.arange(frames)*CELL_SAMPLES+CELL_SAMPLES//2)/SR
    intervals = (CELL_SAMPLES+np.minimum((np.arange(frames)+1)*CELL_SAMPLES, samples)
                 - np.arange(frames)*CELL_SAMPLES)/SR
    layers = rng.standard_normal((2, frames, 1024), dtype=np.float32)
    mel = rng.standard_normal((frames*4, 128), dtype=np.float32)
    for name, value in (('L12.npy', layers[0]), ('L23.npy', layers[1]), ('times_s.npy', times.astype(np.float32)),
                        ('intervals_s.npy', intervals.astype(np.float32)), ('mel10_complete.npy', mel)):
        np.save(folder/name, np.ascontiguousarray(value), allow_pickle=False)
    files = {name: dict(bytes=(folder/name).stat().st_size,
                        sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest())
             for name in ('L12.npy', 'L23.npy', 'times_s.npy', 'intervals_s.npy', 'mel10_complete.npy')}
    receipt = dict(schema=SCHEMA, audio_sha256=audio_sha, decoded_samples=samples, frames=frames,
                   metadata=dict(repo=REPO, revision=REVISION, sample_rate=SR, real_samples=samples,
                                 nominal_first_center_ms=15, full_song_context=False,
                                 synthetic_benchmark_fixture=True, seed=args.seed),
                   backend='synthetic_fixture', files=files, wall_s=None, complete=True,
                   draft_song=dict(codec='pcm_s16le', decoder=dict(
                       fn='noteforge.scripts.make_bench_fixture.write_wave',
                       comment='24 kHz mono silence; feature values are synthetic')))
    (folder/'receipt.json').write_text(json.dumps(receipt, indent=1)+'\n')
    print(json.dumps(dict(audio=str(audio), features_dir=str(out), audio_sha256=audio_sha,
                          seconds=args.seconds, frames=frames), indent=1))


if __name__ == '__main__':
    main()
