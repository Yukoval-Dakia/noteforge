"""Check whether this host can draft a chart, and say what is missing when it cannot.

    python -m noteforge.scripts.doctor            # human-readable report
    python -m noteforge.scripts.doctor --json     # the same report as JSON
    python -m noteforge.scripts.doctor --fix      # build the MLX exports it can build

The report reuses the same resolvers the runtime uses (``noteforge.backends``,
``resolve_mert_source``, the manifest) instead of a second opinion, so a green
doctor means the normal path is ready, not just that some file exists.

Exit status is 0 when the host can draft, 1 when a required piece is missing.
Warnings (slow paths, absent optional extras) do not change the status.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from importlib.util import find_spec

import noteforge.backends as backends

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'models/MANIFEST.json'
STATUS_ORDER = ('ok', 'info', 'warn', 'fail')


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, status, message, **detail):
        self.checks.append(dict(check=name, status=status, message=message, **detail))

    @property
    def status(self):
        return max((c['status'] for c in self.checks), key=STATUS_ORDER.index, default='ok')

    @property
    def ready(self):
        return self.status != 'fail'


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def tool(name):
    """(path, version) for an external tool, or (None, None) when it is not on PATH."""
    path = shutil.which(name)
    if path is None:
        return None, None
    try:
        done = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return path, None
    first = (done.stdout or done.stderr or '').strip().splitlines()
    return path, (first[0].strip() if first else None)


def check_platform(report):
    report.add('platform', 'info', f'{platform.platform()} / {platform.machine()}',
               apple_silicon=backends.apple_silicon(), python=sys.version.split()[0])


def check_tools(report):
    for name in ('ffmpeg', 'ffprobe'):
        path, version = tool(name)
        if path:
            report.add(f'tool.{name}', 'ok', version or path, path=path)
        else:
            report.add(f'tool.{name}', 'fail', f'{name} is not on PATH; audio decode and codec probing need it')


def check_manifest(report, verify_hashes):
    if not MANIFEST.is_file():
        report.add('manifest', 'fail', f'{MANIFEST} is missing; reinstall or re-clone the repository')
        return
    data = json.loads(MANIFEST.read_text())
    entries = {entry['path']: entry for entry in data['files']}
    if not verify_hashes:
        report.add('manifest', 'info', f'{len(entries)} entries',
                   verify='skipped (--no-hash); run without --no-hash to check SHA-256')
        return
    missing, mismatched, checked = [], [], 0
    for path, entry in sorted(entries.items()):
        full = ROOT/path
        if not full.is_file():
            # A file that is not on disk is missing, never a hash mismatch.
            missing.append(path)
            continue
        actual = sha256_file(full)
        checked += 1
        if actual != entry['sha256']:
            mismatched.append(f'{path} (expected {entry["sha256"][:12]}..., got {actual[:12]}...)')
    if mismatched:
        report.add('manifest', 'fail', 'weight files do not match the published SHA-256: ' + '; '.join(mismatched),
                   checked=checked)
    elif missing:
        report.add('manifest', 'fail', 'missing weights: ' + ', '.join(missing), checked=checked,
                   hint=f'download it with the URL in {MANIFEST.relative_to(ROOT)}')
    else:
        report.add('manifest', 'ok', f'{checked} weight files match the published SHA-256', checked=checked)


def check_dependencies(report):
    runtime = backends.torch_runtime()
    if runtime['available']:
        report.add('torch', 'ok', f"torch {runtime['version']}", cuda=runtime['cuda'], mps=runtime['mps'])
    else:
        report.add('torch', 'fail', 'PyTorch is required: ' + str(runtime['error']))
    for name, status, note in (
            ('torchaudio', 'fail', 'the pinned MERT modeling file imports torchaudio'),
            ('transformers', 'fail', 'feature extraction needs the pinned transformers version'),
            ('numpy', 'fail', 'the runtime imports numpy'),
            ('safetensors', 'warn', 'only the MLX weight export uses it'),
            ('fastapi', 'warn', 'the HTTP API needs it; the CLI does not'),
            ('uvicorn', 'warn', 'the HTTP API needs it; the CLI does not'),
            ('mlx', 'info', 'the Apple Silicon backend needs it; torch still works without it')):
        found = find_spec(name) is not None
        report.add(f'dependency.{name}', 'ok' if found else status,
                   'installed' if found else f'{name} is not installed: {note}')


def check_mert_source(report, source_arg):
    from noteforge.audio.mert2 import LOCAL_SOURCE_SHA256, REPO, REVISION
    from noteforge.scripts.draft_song import LOCAL_MERT_SPLIT
    if source_arg == 'hf':
        report.add('mert_source', 'info', f'pinned Hugging Face revision {REVISION} downloads on first use', repo=REPO)
        return
    candidates = [Path(source_arg)] if source_arg else [ROOT/'models/mert2-source']
    for candidate in candidates:
        if all((candidate/name).is_file() for name in LOCAL_SOURCE_SHA256):
            report.add('mert_source', 'ok', f'pinned MERT-v2 files under {candidate}', path=str(candidate))
            return
        if all((candidate/rel).is_file() for rel in LOCAL_MERT_SPLIT.values()):
            report.add('mert_source', 'ok', f'split pinned MERT-v2 layout under {candidate}', path=str(candidate))
            return
    report.add('mert_source', 'fail',
               f'no pinned MERT-v2 source under {", ".join(map(str, candidates))}; the encoder can download it with '
               '--mert-source hf', hint='--mert-source hf')


def check_backend(report, backend, device, fast_weights, fast_mert):
    target = backends.resolve_backend(backend, device, fast_weights, fast_mert, root=ROOT)
    if target.ready:
        report.add('backend', 'ok', f'{target.backend} on {target.device}',
                   requested_backend=target.requested_backend, requested_device=target.requested_device,
                   backend=target.backend, device=target.device)
    else:
        report.add('backend', 'fail', target.error or 'no backend resolved',
                   requested_backend=target.requested_backend, requested_device=target.requested_device,
                   backend=target.backend)
    if backend in ('auto', 'mlx') and backends.apple_silicon() and not backends.mlx_assets_ready(fast_weights, fast_mert):
        weights, mert = backends.discover_mlx_assets(ROOT)
        if weights is None or mert is None:
            report.add('backend.mlx_assets', 'info',
                       f'no MLX export under {backends.MLX_FAST_DIR} and {backends.MLX_MERT_DIR}; '
                       'run --fix to build them, or the torch backend is used', fixable=True)
    return target


def check_optional_api(report):
    if find_spec('fastapi') and find_spec('uvicorn'):
        report.add('api', 'ok', 'fastapi and uvicorn are installed; `uvicorn noteforge_api.asgi:app` can start')
    else:
        report.add('api', 'warn', 'the HTTP API is not installed; `pip install -e ".[api]"` to serve it')


def run_fix(report, args):
    from noteforge.scripts import draft_song as ds
    weights, mert = backends.discover_mlx_assets(ROOT)
    if weights is not None and mert is not None:
        report.add('fix', 'info', 'MLX exports already exist; nothing to build')
        return
    if not backends.mlx_available():
        report.add('fix', 'warn', 'cannot build MLX exports without the mlx extra; `pip install -e ".[mlx]"`')
        return
    checkpoint = Path(args.model or os.environ.get('NOTEFORGE_MODEL') or ROOT/'models/noteforge-r5.pt')
    if not checkpoint.is_file():
        report.add('fix', 'warn', f'cannot export MLX weights: {checkpoint} is missing')
        return
    from noteforge.mlx import fast_model, mert_fast
    if weights is None:
        fast_model.export(checkpoint, ROOT/backends.MLX_FAST_DIR)
        report.add('fix', 'ok', f'wrote {backends.MLX_FAST_DIR}')
    if mert is None:
        try:
            source = ds.resolve_mert_source(args.mert_source)[0]
        except SystemExit as exc:
            report.add('fix', 'warn', f'MLX MERT export skipped: {exc}')
            return
        mert_fast.export(str(source), ROOT/backends.MLX_MERT_DIR)
        report.add('fix', 'ok', f'wrote {backends.MLX_MERT_DIR}')


def print_text(report):
    marks = dict(ok='OK  ', info='INFO', warn='WARN', fail='FAIL')
    for check in report.checks:
        print(f"[{marks[check['status']]}] {check['check']}: {check['message']}")
        if check.get('hint'):
            print(f"         -> {check['hint']}")
    print()
    print(f"summary: {report.status.upper()} "
          + ('(this host can draft a chart)' if report.ready else '(this host cannot draft yet)'))


def collect(args):
    report = Report()
    check_platform(report)
    check_tools(report)
    check_manifest(report, verify_hashes=not args.no_hash)
    check_dependencies(report)
    check_mert_source(report, args.mert_source)
    check_backend(report, args.backend, args.device, args.mlx_weights, args.mlx_mert)
    check_optional_api(report)
    if args.fix:
        run_fix(report, args)
        check_backend(report, args.backend, args.device, args.mlx_weights, args.mlx_mert)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='store_true', help='print the report as JSON')
    parser.add_argument('--fix', action='store_true', help='build anything that can be built here (the MLX exports)')
    parser.add_argument('--no-hash', action='store_true', help='skip the SHA-256 check of the committed weights')
    parser.add_argument('--backend', choices=backends.BACKENDS, default=os.environ.get('NOTEFORGE_BACKEND', 'auto'))
    parser.add_argument('--device', choices=backends.DEVICES, default=os.environ.get('NOTEFORGE_DEVICE', 'auto'))
    parser.add_argument('--mlx-weights', type=Path, default=os.environ.get('NOTEFORGE_FAST_WEIGHTS') or None)
    parser.add_argument('--mlx-mert', type=Path, default=os.environ.get('NOTEFORGE_FAST_MERT') or None)
    parser.add_argument('--mert-source', default=os.environ.get('NOTEFORGE_MERT_SOURCE'),
                        help="pinned MERT-v2 directory, or 'hf' (default: NOTEFORGE_MERT_SOURCE)")
    parser.add_argument('--model', type=Path, help='checkpoint to export for --fix (default: the published r5 weights)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = collect(args)
    if args.json:
        print(json.dumps(dict(ready=report.ready, status=report.status, checks=report.checks), indent=1))
    else:
        print_text(report)
    return 0 if report.ready else 1


if __name__ == '__main__':
    sys.exit(main())
