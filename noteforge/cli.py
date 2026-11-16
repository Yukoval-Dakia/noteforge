"""One entry point for NoteForge: `noteforge <command>`.

    noteforge doctor     check this host and say what is missing (add --fix)
    noteforge draft      generate an editable chart draft
    noteforge extract    write the reusable features for one song and stop
    noteforge plan       plan a per-section density curve before drafting
    noteforge auto       plan, draft, and (optionally) align to a BPM in one run
    noteforge inspect    read a chart file: notes, timing segments, opening style
    noteforge bench      measure one backend on a fixed input
    noteforge fixture    write a synthetic feature fixture for benchmarks
    noteforge mert       fetch or check the shared MERT source; export its MLX cache
    noteforge export     export the published r5 model to the MLX fp16 runtime
    noteforge serve      start the HTTP API
    noteforge version    print the version

Every command forwards its own arguments unchanged to the module that
implements it, so `noteforge draft --help` is exactly `draft_song --help` and no
option is lost to a second, thinner parser. Run `noteforge <command> --help` for
that command's full option list.
"""
import argparse
import os
from pathlib import Path
import sys

COMMANDS = ('doctor', 'draft', 'extract', 'plan', 'auto', 'inspect', 'bench', 'fixture', 'mert', 'export', 'serve')


def version():
    """The source version, which is what `noteforge` ships with.

    Installed metadata is deliberately not consulted: a stale editable install
    would then report a different number than the code being run.
    """
    from noteforge import __version__
    return __version__


def top_parser():
    parser = argparse.ArgumentParser(
        prog='noteforge', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=COMMANDS,
                        help='the command to run; run `noteforge <command> --help` for its options')
    return parser


def run_doctor(argv):
    from noteforge.scripts import doctor
    return doctor.main(argv)


def run_draft(argv):
    from noteforge.scripts import draft_song
    receipt = draft_song.main(argv)
    return 0 if receipt else 1


def run_extract(argv):
    """Feature extraction is `draft_song --extract-only`; keep the same options."""
    if '--extract-only' not in argv:
        argv = ['--extract-only', *argv]
    from noteforge.scripts import draft_song
    receipt = draft_song.main(argv)
    return 0 if receipt else 1


def run_plan(argv):
    from noteforge.planner import plan_song
    plan_song.main(argv)
    return 0


def run_auto(argv):
    from noteforge.planner import auto_draft
    auto_draft.main(argv)
    return 0


def inspect_parser():
    parser = argparse.ArgumentParser(
        prog='noteforge inspect',
        description='Read a chart file (.osu for mania, RPE or official JSON for Phira) and print notes, '
                    'timing segments and opening style as JSON.')
    parser.add_argument('chart', help='the chart file')
    parser.add_argument('--mode', choices=('mania_4k', 'phira'), required=True)
    return parser


def run_inspect(argv):
    args = inspect_parser().parse_args(argv)
    from noteforge_api import ops
    return ops.main(['inspect-chart', args.chart, args.mode])


def run_bench(argv):
    from noteforge.scripts import bench_backends
    bench_backends.main(argv)
    return 0


def run_fixture(argv):
    from noteforge.scripts import make_bench_fixture
    make_bench_fixture.main(argv)
    return 0


def mert_parser():
    parser = argparse.ArgumentParser(
        prog='noteforge mert',
        description='Manage the shared pinned MERT-v2 source used by torch CPU/CUDA/MPS and MLX. '
                    '`fetch --source hf` downloads about 2.4 GB; `export-mlx` is an optional '
                    'Apple-only bf16 cache derived from that source.')
    parser.add_argument('action', choices=('status', 'fetch', 'export-mlx'))
    parser.add_argument('--source', default=os.environ.get('NOTEFORGE_MERT_SOURCE'),
                        help="a pinned local directory or 'hf' (default: NOTEFORGE_MERT_SOURCE)")
    parser.add_argument('--out', help='source directory for fetch, or the derived MLX cache for export-mlx')
    parser.add_argument('--json', action='store_true', help='print status as JSON')
    return parser


def run_mert(argv):
    args = mert_parser().parse_args(argv)
    from noteforge.audio import mert_source
    if args.action == 'status':
        if not args.source:
            print('noteforge mert status needs --source DIR (or NOTEFORGE_MERT_SOURCE)', file=sys.stderr)
            return 2
        source, identity = mert_source.resolve(args.source)
        report = dict(
            action='status',
            source=str(source) if source is not None else 'hf',
            downloaded=source is not None,
            identity=identity,
            note='The same MERT source is shared by torch CPU/CUDA/MPS and MLX. '
                 'The mlx-mert path is only an optional derived bf16 export.',
        )
        if args.json:
            import json
            print(json.dumps(report, indent=2))
        else:
            print(f"MERT source: {report['source']}")
            print('Shared by: torch CPU, torch CUDA, torch MPS, MLX')
            print('MLX bf16 cache: optional derived export, not a download restriction')
        return 0
    if args.action == 'export-mlx':
        if not args.source:
            print('noteforge mert export-mlx needs --source DIR|hf (or NOTEFORGE_MERT_SOURCE)', file=sys.stderr)
            return 2
        from noteforge.backends import MLX_MERT_DIR
        from noteforge.mlx import mert_fast
        root = Path(__file__).resolve().parents[1]
        out = Path(args.out) if args.out else root/MLX_MERT_DIR
        mert_fast.export(args.source, out)
        return 0
    if not args.source:
        print('noteforge mert fetch needs --source DIR|hf (or NOTEFORGE_MERT_SOURCE)', file=sys.stderr)
        return 2
    if args.source == 'hf' and not args.out:
        print('noteforge mert fetch --source hf needs --out DIR so the 2.4 GB download has an explicit target',
              file=sys.stderr)
        return 2
    if not args.out:
        print('noteforge mert fetch needs --out DIR', file=sys.stderr)
        return 2
    print(mert_source.fetch(args.source, args.out))
    return 0


def export_parser():
    parser = argparse.ArgumentParser(
        prog='noteforge export',
        description='Export the model to the MLX fp16 runtime. `mlx-mert` is a compatibility alias '
                    'for the optional bf16 MERT cache; it is not the shared MERT download path.')
    parser.add_argument('target', choices=('mlx', 'mlx-mert'), help='what to export')
    parser.add_argument('--checkpoint', default=os.environ.get('NOTEFORGE_MODEL', 'models/noteforge-r5.pt'),
                        help='r5 checkpoint for `mlx`')
    parser.add_argument('--mert-source', default=os.environ.get('NOTEFORGE_MERT_SOURCE'),
                        help="shared pinned MERT-v2 directory or 'hf' for the optional `mlx-mert` export")
    parser.add_argument('--out', help='output folder (default: the conventional models/ folder)')
    return parser


def run_export(argv):
    args = export_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    from noteforge.backends import MLX_FAST_DIR, MLX_MERT_DIR
    if args.target == 'mlx':
        from noteforge.mlx import fast_model
        out = Path(args.out) if args.out else root/MLX_FAST_DIR
        fast_model.export(args.checkpoint, out)
    else:
        if not args.mert_source:
            print('noteforge export mlx-mert needs --mert-source DIR (or NOTEFORGE_MERT_SOURCE)', file=sys.stderr)
            return 2
        from noteforge.mlx import mert_fast
        out = Path(args.out) if args.out else root/MLX_MERT_DIR
        mert_fast.export(args.mert_source, out)
    return 0


def serve_parser():
    parser = argparse.ArgumentParser(
        prog='noteforge serve', description='Start the NoteForge HTTP API (uvicorn noteforge_api.asgi:app).')
    parser.add_argument('--host', default=os.environ.get('NOTEFORGE_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('NOTEFORGE_PORT', 8787)))
    parser.add_argument('--reload', action='store_true', help='reload on source changes (development)')
    parser.add_argument('--log-level', default='info')
    return parser


def run_serve(argv):
    args = serve_parser().parse_args(argv)
    if not os.environ.get('NOTEFORGE_API_TOKEN'):
        print('noteforge serve needs NOTEFORGE_API_TOKEN (any bearer token you choose); set it first', file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError:
        print('the HTTP API needs uvicorn: pip install -e ".[api]"', file=sys.stderr)
        return 2
    print(f'serving on http://{args.host}:{args.port}  (docs at /docs, OpenAPI in openapi.json)', flush=True)
    uvicorn.run('noteforge_api.asgi:app', host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
    return 0


RUNNERS = dict(doctor=run_doctor, draft=run_draft, extract=run_extract, plan=run_plan, auto=run_auto,
               inspect=run_inspect, bench=run_bench, fixture=run_fixture, mert=run_mert,
               export=run_export, serve=run_serve)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = top_parser()
    if not raw:
        parser.print_help()
        print(f"\nstart here: noteforge doctor   (then `noteforge doctor --fix` on Apple Silicon)")
        return 0
    if raw[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    if raw[0] == '--version' or raw[0] == 'version':
        print(version())
        return 0
    if raw[0] == 'help':
        if len(raw) > 1 and raw[1] in RUNNERS:
            return RUNNERS[raw[1]](['--help'])
        parser.print_help()
        return 0
    command, rest = raw[0], raw[1:]
    if command not in RUNNERS:
        parser.error(f'unknown command {command!r}; choose from {", ".join(COMMANDS)}')
    # The submodule parsers take their prog from sys.argv[0]; name it after the
    # subcommand so `noteforge draft --help` reads `noteforge draft`, not `cli.py`.
    previous = sys.argv[0]
    sys.argv[0] = f'noteforge {command}'
    try:
        return RUNNERS[command](rest)
    finally:
        sys.argv[0] = previous


if __name__ == '__main__':
    sys.exit(main())
