"""One command for the frozen setup: one sample, the model's own
history, a planned density curve, and optionally a typed BPM whose phase comes from a first pass of the model's own notes.

  auto_draft.py --audio song.ogg --mode phira --level 5.5 --bpm 174 --features-dir F --model M --plan P --out-dir OUT
Style ratios default to the planner's suggestion; give --opening-chart (the charter's own opening, G45/G46), or --hold-ratio etc.
to override. Without --level the planner also guesses the level, which passed its check only on mania, so Phira then needs --auto-level.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIELD_TEMPERATURE = '{"kind": 1.0, "head_block": 0.7, "head_cell": 0.7, "head_ms": 0.7}'
PHASE_BAND_BINS = 6  # bins of 2 ms: a +-6 ms band
OPENING_US = 30_000_000


def phase_us(onsets_us, bpm):
    """Position within a 1/4 beat of the strongest onset cluster."""
    onsets = np.asarray(onsets_us, float)
    if not len(onsets) or bpm <= 0:
        raise ValueError('phase_us needs onsets and a positive BPM')
    step = 60e6/bpm/4
    hist, edges = np.histogram(onsets % step, bins=int(step/2000), range=(0, step))
    band = np.convolve(np.r_[hist, hist[:PHASE_BAND_BINS]], np.ones(PHASE_BAND_BINS), 'valid')[:len(hist)]
    return float(edges[int(np.argmax(band))]+PHASE_BAND_BINS*1000)


def flatten_style(style, given):
    """Planner shares as draft_song arguments; shares the charter gave win."""
    out = []
    for field, value in style.items():
        name = field.split('_')[0]
        chosen = given.get(name)
        out += [f'--{name}-ratio', str(value if chosen is None else chosen)]
    return out


def opening_shares(notes, mode, opening_us=OPENING_US):
    """Style shares of a chart's first 30 s from its first note. notes: (onset_us, type). Same definitions as the benchmark's style bands."""
    notes = sorted(notes)
    if not notes:
        raise ValueError('the opening chart has no notes')
    head = [n for n in notes if n[0] < notes[0][0]+opening_us]
    onsets = {}
    for on, _ in head:
        onsets[on] = onsets.get(on, 0)+1
    share = dict(hold=sum(t == 'hold' for _, t in head)/len(head), chord=sum(c >= 2 for c in onsets.values())/len(onsets))
    if mode == 'phira':
        share.update(flick=sum(t == 'flick' for _, t in head)/len(head), drag=sum(t == 'drag' for _, t in head)/len(head))
    return share


def read_opening(path, mode):
    data = Path(path).read_bytes()
    if mode == 'mania_4k':
        from noteforge.events.osu_mania import parse_osu
        chart = parse_osu(data, Path(path).name)
    else:
        from noteforge.events.phira import parse_phira
        chart = parse_phira(data, Path(path).name)
    return [(e.onset_us, e.type) for e in chart.events]


def run(cmd):
    print('+', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--audio', type=Path, required=True)
    a.add_argument('--mode', choices=('phira', 'mania_4k'), required=True)
    a.add_argument('--level', type=float, help='overall notes per second')
    a.add_argument('--auto-level', action='store_true', help='Phira only: let the planner guess the level (did not pass its check)')
    a.add_argument('--bpm', type=float, help='the song\'s BPM (tempo changes are not supported)')
    for name in ('hold', 'chord', 'flick', 'drag'):
        a.add_argument(f'--{name}-ratio', type=float)
    a.add_argument('--opening-chart', type=Path, help='the charter\'s own chart of at least the opening (.osu for mania, RPE or official .json for Phira): style from its first 30 s')
    a.add_argument('--features-dir', type=Path, required=True)
    a.add_argument('--planner-dir', type=Path, default=Path('planner'))
    a.add_argument('--model', type=Path, required=True)
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--out-dir', type=Path, required=True)
    a.add_argument('--device', default='cuda')
    a.add_argument('--batch-size', type=int, default=64)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--max-wall-s', type=float, default=1200.)
    a.add_argument('--mert-source', type=Path)
    a.add_argument('--title')
    args, passthrough = a.parse_known_args(argv)
    if args.level is None and args.mode == 'phira' and not args.auto_level:
        a.error('Phira needs --level (or --auto-level: the planner\'s guess did not pass its check on Phira)')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(args.audio.read_bytes()).hexdigest()
    nested = args.features_dir/'features'
    py = sys.executable
    plan = args.out_dir/'curve.json'
    command = [py, HERE/'plan_song.py', '--features-dir', nested if nested.is_dir() else args.features_dir, '--audio-sha', sha,
               '--mode', args.mode, '--planner-dir', args.planner_dir, '--device', args.device, '--out', plan]
    if args.level is not None:
        command += ['--density', args.level]
    print('+', ' '.join(str(x) for x in command), flush=True)
    planned = subprocess.run([str(x) for x in command], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    planned = json.loads(planned)
    print('plan', planned, flush=True)
    given = opening_shares(read_opening(args.opening_chart, args.mode), args.mode) if args.opening_chart else {}
    if given:
        print('opening style', given, flush=True)
    for name, value in dict(hold=args.hold_ratio, chord=args.chord_ratio, flick=args.flick_ratio, drag=args.drag_ratio).items():
        if value is not None:
            given[name] = value
    style = planned['suggested_style'] if args.mode == 'phira' else {k: v for k, v in planned['suggested_style'].items() if k.startswith(('hold', 'chord'))}
    base = [py, '-m', 'noteforge.scripts.draft_song', '--audio', args.audio, '--mode', args.mode, '--features-dir', args.features_dir,
            '--model', args.model, '--plan', args.plan, '--device', args.device, '--batch-size', args.batch_size, '--scheduler', 'continuous',
            '--seed', args.seed, '--max-wall-s', args.max_wall_s, '--field-temperature', FIELD_TEMPERATURE, '--history-reference',
            '--density-curve', plan, *flatten_style(style, given), *passthrough]
    if args.mert_source:
        base += ['--mert-source', args.mert_source]
    if args.title:
        base += ['--title', args.title]
    final = [*base, '--out-dir', args.out_dir/'draft', '--consensus', '1']
    if args.bpm:
        first = args.out_dir/'first_pass'
        run([*base, '--out-dir', first, '--consensus', '4', '--consensus-pattern', 'anchor', '--consensus-votes', '2', '--profile', 'none'])
        events = json.loads((first/'draft-events.json').read_text())['events']
        phase = phase_us([e['onset_us'] for e in events], args.bpm)
        beat = 60e6/args.bpm
        print(f'beat grid: BPM {args.bpm}, phase {phase/1000:.1f} ms on the model clock', flush=True)
        final += ['--beat-grid', f'{phase % beat:.3f},{beat:.6f}']
    run(final)
    print('done', args.out_dir/'draft', flush=True)


if __name__ == '__main__':
    main()
