"""One entry point for NoteForge: `noteforge <command>`.

    noteforge doctor     check this host and say what is missing (add --fix)
    noteforge draft      generate an editable chart draft
    noteforge extract    write the reusable features for one song and stop
    noteforge plan       plan a per-section density curve before drafting
    noteforge auto       plan, draft, and (optionally) align to a BPM in one run
    noteforge bench      measure one backend on a fixed input
    noteforge fixture    write a synthetic feature fixture for benchmarks
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

COMMANDS = ('doctor', 'draft', 'extract', 'plan', 'auto', 'bench', 'fixture', 'export', 'serve')


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


def run_bench(argv):
    from noteforge.scripts import bench_backends
    bench_backends.main(argv)
    return 0


def run_fixture(argv):
    from noteforge.scripts import make_bench_fixture
    make_bench_fixture.main(argv)
    return 0


def export_parser():
    parser = argparse.ArgumentParser(
        prog='noteforge export', description='Export the model to the MLX fp16 runtime, or the MERT front end to bf16.')
    parser.add_argument('target', choices=('mlx', 'mlx-mert'), help='what to export')
    parser.add_argument('--checkpoint', default=os.environ.get('NOTEFORGE_MODEL', 'models/noteforge-r5.pt'),
                        help='r5 checkpoint for `mlx`')
    parser.add_argument('--mert-source', default=os.environ.get('NOTEFORGE_MERT_SOURCE'),
                        help='pinned MERT-v2 directory for `mlx-mert`')
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
               bench=run_bench, fixture=run_fixture, export=run_export, serve=run_serve)


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
