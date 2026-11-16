"""Hardware backend discovery and runtime configuration for NoteForge inference."""
from dataclasses import asdict, dataclass
from importlib.util import find_spec
import os
from pathlib import Path
import platform
import sys

BACKENDS = ('auto', 'torch', 'mlx')
DEVICES = ('auto', 'cuda', 'mps', 'cpu')
MLX_WEIGHT_FILES = ('meta.json', 'weights.safetensors', 'head.pt', 'vocab.pt')
MLX_MERT_FILES = ('model.safetensors', 'config.json', 'configuration_mert2.py', 'modeling_mert2.py')


@dataclass(frozen=True)
class BackendTarget:
    requested_backend: str
    backend: str
    requested_device: str
    device: str
    error: str | None = None

    @property
    def ready(self):
        return self.error is None

    def as_dict(self):
        return asdict(self)


def _choice(value, choices, field):
    value = str(value or choices[0]).lower()
    if value not in choices:
        raise ValueError(f'{field} must be one of {", ".join(choices)}')
    return value


def apple_silicon():
    return sys.platform == 'darwin' and platform.machine() == 'arm64'


def mlx_available():
    return find_spec('mlx') is not None


def folder_has(folder, names):
    if not folder:
        return False
    root = Path(folder)
    return root.is_dir() and all((root / name).is_file() for name in names)


def mlx_assets_ready(fast_weights=None, fast_mert=None):
    return folder_has(fast_weights, MLX_WEIGHT_FILES) and folder_has(fast_mert, MLX_MERT_FILES)


def torch_runtime():
    """Availability of the PyTorch devices without constructing or loading a model."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - optional probe
        return dict(available=False, error=f'{type(exc).__name__}: {exc}', cuda=False, mps=False)
    mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    return dict(available=True, error=None, version=torch.__version__, cuda=torch.cuda.is_available(), mps=mps)


def resolve_torch_device(requested='auto'):
    requested = _choice(requested, DEVICES, 'device')
    runtime = torch_runtime()
    if not runtime['available']:
        raise RuntimeError('PyTorch is required for the torch backend: ' + runtime['error'])
    if requested == 'auto':
        return 'cuda' if runtime['cuda'] else 'mps' if runtime['mps'] else 'cpu'
    if requested == 'cuda' and not runtime['cuda']:
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is false')
    if requested == 'mps' and not runtime['mps']:
        raise RuntimeError('MPS was requested, but torch.backends.mps.is_available() is false')
    return requested


def resolve_backend(requested_backend='auto', requested_device='auto', fast_weights=None, fast_mert=None):
    requested_backend = _choice(requested_backend, BACKENDS, 'backend')
    requested_device = _choice(requested_device, DEVICES, 'device')
    if requested_backend == 'mlx' or (requested_backend == 'auto' and requested_device == 'auto'
                                      and apple_silicon() and mlx_available()
                                      and mlx_assets_ready(fast_weights, fast_mert)):
        if not apple_silicon():
            error = 'MLX inference requires Apple Silicon (darwin arm64)'
        elif not mlx_available():
            error = 'MLX is not installed; install the mlx extra'
        elif not folder_has(fast_weights, MLX_WEIGHT_FILES):
            error = 'MLX fast weights are missing; export them with noteforge.mlx.fast_model'
        elif not folder_has(fast_mert, MLX_MERT_FILES):
            error = 'MLX fast MERT weights are missing; export them with noteforge.mlx.mert_fast'
        else:
            error = None
        return BackendTarget(requested_backend, 'mlx', requested_device, 'mlx', error)
    if requested_backend == 'mlx':
        return BackendTarget(requested_backend, 'mlx', requested_device, None, 'MLX inference is not available on this host')
    try:
        device = resolve_torch_device(requested_device)
    except Exception as exc:  # noqa: BLE001 - reported through health/capabilities
        # ``device`` stays None: nothing resolved, so reporting the requested name
        # as if it were live would misstate the host.
        return BackendTarget(requested_backend, 'torch', requested_device, None, str(exc))
    return BackendTarget(requested_backend, 'torch', requested_device, device)


def configure_torch(cpu_threads=None, interop_threads=None):
    import torch
    cpu_threads = int(cpu_threads or os.environ.get('NOTEFORGE_CPU_THREADS', 0) or 0)
    interop_threads = int(interop_threads or os.environ.get('NOTEFORGE_CPU_INTEROP_THREADS', 0) or 0)
    changed = {}
    if cpu_threads:
        if cpu_threads < 1:
            raise ValueError('cpu_threads must be positive')
        if torch.get_num_threads() != cpu_threads:
            torch.set_num_threads(cpu_threads)
        changed['cpu_threads'] = cpu_threads
    if interop_threads:
        if interop_threads < 1:
            raise ValueError('interop_threads must be positive')
        if torch.get_num_interop_threads() != interop_threads:
            torch.set_num_interop_threads(interop_threads)
        changed['interop_threads'] = interop_threads
    return changed


def torch_identity(device=None):
    runtime = torch_runtime()
    if not runtime['available']:
        return runtime
    import torch
    selected = resolve_torch_device(device or 'auto')
    identity = dict(runtime, selected_device=selected, cpu_threads=torch.get_num_threads(),
                    interop_threads=torch.get_num_interop_threads(),
                    mkldnn=bool(getattr(torch.backends, 'mkldnn', None) and torch.backends.mkldnn.is_available()),
                    machine=platform.machine(), os=platform.platform())
    if selected == 'cuda':
        identity['cuda_device'] = torch.cuda.get_device_name(0)
        identity['cuda_capability'] = list(torch.cuda.get_device_capability(0))
        bf16 = getattr(torch.cuda, 'is_bf16_supported', None)
        identity['cuda_bf16'] = bool(bf16()) if bf16 is not None else False
    return identity


def mlx_identity():
    if not mlx_available():
        return dict(available=False)
    import mlx.core as mx
    metal = getattr(getattr(mx, 'metal', None), 'is_available', None)
    return dict(available=True, version=getattr(mx, '__version__', None),
                default_device=str(mx.default_device()),
                metal=bool(metal()) if metal is not None else None)


def availability(fast_weights=None, fast_mert=None):
    torch = torch_runtime()
    return dict(torch=torch, mlx=dict(available=mlx_available(), apple_silicon=apple_silicon(),
                                      assets=mlx_assets_ready(fast_weights, fast_mert)),
                cpu=True)
