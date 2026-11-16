"""Backend-neutral access to the pinned MERT-v2 upstream source.

MERT is shared by the PyTorch CPU/CUDA/MPS paths and the Apple MLX path. The
``mlx-mert`` directory is only an optional derived bf16 export; it is not a
separate download source or a backend restriction.
"""
from pathlib import Path
import shutil


def resolve(source_arg):
    """Resolve a local source without downloading anything.

    Returns ``(path_or_none, identity)``. ``None`` means the caller should use
    the pinned Hugging Face revision.
    """
    from noteforge.audio.mert2 import REPO, REVISION
    from noteforge.scripts.draft_song import resolve_mert_source

    directory, identity, _ = resolve_mert_source(source_arg)
    if directory is None:
        return None, dict(
            kind='huggingface',
            repo=REPO,
            revision=REVISION,
            shared_by=('torch-cpu', 'torch-cuda', 'torch-mps', 'mlx'),
        )
    identity = dict(identity)
    identity['shared_by'] = ('torch-cpu', 'torch-cuda', 'torch-mps', 'mlx')
    return Path(directory), identity


def source_dir(source_arg):
    """Return the pinned MERT source on disk, downloading ``hf`` when requested."""
    from noteforge.audio.mert2 import LOCAL_SOURCE_SHA256, REPO, REVISION, validate_local_source

    directory, _ = resolve(source_arg)
    if directory is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                '--mert-source hf needs huggingface_hub; pip install huggingface_hub'
            ) from exc
        directory = Path(snapshot_download(
            REPO,
            revision=REVISION,
            allow_patterns=sorted(LOCAL_SOURCE_SHA256),
        ))
    validate_local_source(directory)
    return directory


def fetch(source_arg, out):
    """Materialize the validated pinned source in ``out``.

    ``source_arg='hf'`` is the only form that downloads the 2.4 GB upstream
    checkpoint. A local source is copied to ``out`` when the paths differ.
    """
    from noteforge.audio.mert2 import LOCAL_SOURCE_SHA256, REPO, REVISION, validate_local_source

    out = Path(out)
    if source_arg == 'hf':
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                'noteforge mert fetch --source hf needs huggingface_hub; '
                'pip install huggingface_hub'
            ) from exc
        out.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            REPO,
            revision=REVISION,
            allow_patterns=sorted(LOCAL_SOURCE_SHA256),
            local_dir=str(out),
        )
        validate_local_source(out)
        return out.resolve()

    source, _ = resolve(source_arg)
    source = Path(validate_local_source(source))
    if source.resolve() == out.resolve():
        return out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in LOCAL_SOURCE_SHA256:
        shutil.copy2(source/name, out/name)
    validate_local_source(out)
    return out.resolve()
