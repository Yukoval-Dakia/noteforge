"""Measure a NoteForge draft on one backend with a fixed audio, seed and decoding contract.

This is a measurement tool, not a quality gate: it reports model load, generation and
wall time plus the note multiset hash, so two backends can be compared on identical
inputs. It never rewrites the chart or trims notes to make a backend look faster.

    python -m noteforge.scripts.bench_backends --backend torch --device cpu \
        --cpu-threads 4 --audio song.mp3 --features-dir ./features --mode phira --level 10

The backend arguments are passed through to ``noteforge.scripts.draft_song``, so the same
backend flags the product uses (``--mlx-plan``, ``--autocast``, ``--reference-decode``,
``--batch-size``) are measured.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from noteforge.backends import resolve_backend


def note_hash(events):
    canonical = sorted(json.dumps(e, sort_keys=True, separators=(',', ':')) for e in events)
    return hashlib.sha256('\n'.join(canonical).encode()).hexdigest()


def one_run(args, out):
    command = [sys.executable, '-m', 'noteforge.scripts.draft_song',
               '--backend', args.backend, '--device', args.device,
               '--model', args.model, '--audio', args.audio, '--out-dir', str(out),
               '--features-dir', args.features_dir, '--mode', args.mode,
               '--consensus', '1', '--scheduler', args.scheduler, '--batch-size', str(args.batch_size),
               '--token-cap', str(args.token_cap), '--max-wall-s', str(args.max_wall_s),
               '--seed', str(args.seed), '--temperature', args.temperature, '--save-samples']
    command += ['--plan', args.plan] if args.plan else []
    command += ['--profile', args.profile] if args.profile else []
    command += ['--cpu-threads', str(args.cpu_threads)] if args.cpu_threads else []
    command += ['--cpu-interop-threads', str(args.cpu_interop_threads)] if args.cpu_interop_threads else []
    command += ['--reference-decode'] if args.reference_decode else []
    command += ['--autocast', args.autocast] if args.autocast else []
    if args.mlx_weights:
        command += ['--mlx-weights', args.mlx_weights]
    if args.mlx_mert:
        command += ['--mlx-mert', args.mlx_mert]
    if args.mlx_plan:
        command += ['--mlx-plan', args.mlx_plan]
    if args.level is not None:
        command += ['--phira-level' if args.mode == 'phira' else '--mania-star', str(args.level)]
    tick = time.monotonic()
    done = subprocess.run(command, capture_output=True, text=True)
    wall = time.monotonic()-tick
    if done.returncode:
        raise SystemExit(f'run failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}')
    receipt = json.loads((out/'draft-receipt.json').read_text())
    events = json.loads((out/'draft-events.json').read_text())['events']
    return dict(wall_s=round(wall, 4), generation_s=round(receipt['timings']['generation_s'], 4),
                model_load_s=round(receipt['timings']['model_load_s'], 4),
                total_s=round(receipt['timings']['total_s'], 4),
                tokens=sum(s['tokens'] for s in receipt['samples']), notes=len(events),
                cores=sum(s['cores'] for s in receipt['samples']),
                output_sha256=note_hash(events), runtime=receipt['runtime'], command=command)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=('auto', 'torch', 'mlx'), default='auto')
    p.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    p.add_argument('--model', default='models/noteforge-r5.pt')
    p.add_argument('--plan', default='models/config/r5-inference.json')
    p.add_argument('--audio', required=True)
    p.add_argument('--features-dir', required=True)
    p.add_argument('--mode', choices=('phira', 'mania_4k'), default='phira')
    p.add_argument('--level', type=float, help='Phira level or mania star preset')
    p.add_argument('--profile', default='none')
    p.add_argument('--scheduler', choices=('legacy', 'grouped', 'continuous'), default='grouped')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--token-cap', type=int, default=4096)
    p.add_argument('--max-wall-s', type=float, default=1800.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--temperature', default='none')
    p.add_argument('--cpu-threads', type=int)
    p.add_argument('--cpu-interop-threads', type=int)
    p.add_argument('--reference-decode', action='store_true')
    p.add_argument('--autocast', choices=('bf16',))
    p.add_argument('--mlx-weights')
    p.add_argument('--mlx-mert')
    p.add_argument('--mlx-plan', choices=('balanced', 'fast', 'sequential'))
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--out-dir')
    args = p.parse_args(argv)
    args.features_dir = os.path.abspath(args.features_dir)
    args.audio = os.path.abspath(args.audio)
    args.model = os.path.abspath(args.model)
    if args.plan:
        args.plan = os.path.abspath(args.plan)
    target = resolve_backend(args.backend, args.device, args.mlx_weights, args.mlx_mert)
    if not target.ready:
        raise SystemExit(target.error)
    root = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix='noteforge-bench-'))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in range(args.runs):
        out = root/f'run-{run}'
        if out.exists():
            shutil.rmtree(out)
        row = one_run(args, out)
        row['run'] = run
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = dict(target=target.as_dict(), args=vars(args), rows=rows,
                  generation_median_s=sorted(r['generation_s'] for r in rows)[len(rows)//2],
                  generation_range_s=[min(r['generation_s'] for r in rows), max(r['generation_s'] for r in rows)],
                  outputs=sorted({r['output_sha256'] for r in rows}),
                  identical=len({r['output_sha256'] for r in rows}) == 1)
    (root/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('generation_median_s', 'generation_range_s', 'outputs', 'identical')}, indent=2))
    return report


if __name__ == '__main__':
    main()
