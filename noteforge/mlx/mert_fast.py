"""MERT at bf16 from a one-time export: no fp32 load and no 2.5 GB hash on every run.

    mert_fast.py export SOURCE_DIR OUT_DIR     checks the pinned source once, writes bf16 weights + config + code
    mert_fast.py extract FAST_DIR ...          draft_song arguments
The audio frontend stays fp32, as the pinned model requires. Measured: features bit-identical to the pinned MPS path."""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
import torch
from noteforge.scripts import draft_song as ds


def source_dir(source_arg):
    """The pinned MERT-v2 snapshot as a local directory.

    ``--mert-source hf`` means "use the pinned Hugging Face revision"; for the
    export we need its files on disk, so download that exact revision first.
    The four exported files are checked against their pinned SHA-256 either way.
    """
    from noteforge.audio.mert2 import LOCAL_SOURCE_SHA256, REPO, REVISION, validate_local_source
    directory = ds.resolve_mert_source(source_arg)[0]
    if directory is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit('--mert-source hf needs huggingface_hub; pip install huggingface_hub') from exc
        directory = snapshot_download(REPO, revision=REVISION, allow_patterns=sorted(LOCAL_SOURCE_SHA256))
    validate_local_source(directory)
    return Path(directory)


def export(source_arg, out):
    from safetensors.torch import load_file, save_file
    source = source_dir(source_arg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = load_file(str(Path(source)/'model.safetensors'))
    narrowed = {k: v.to(torch.bfloat16) if v.is_floating_point() and not k.startswith('feature_extractor.') else v for k, v in state.items()}
    save_file(narrowed, str(out/'model.safetensors'), metadata=dict(format='pt'))
    for name in ('config.json', 'configuration_mert2.py', 'modeling_mert2.py'):
        shutil.copy(Path(source)/name, out/name)
    print('bytes', (out/'model.safetensors').stat().st_size)


def light_model(folder):
    """The pinned model code, run without importing transformers (1.3 s). It uses only these three small base classes."""
    import importlib.util
    import json
    import types

    class PretrainedConfig:
        output_hidden_states, use_return_dict, _attn_implementation = False, True, 'sdpa'

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def register_for_auto_class(cls, *args, **kwargs):
            pass

    class PreTrainedModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

        @classmethod
        def register_for_auto_class(cls, *args, **kwargs):
            pass

    stub, utils = types.ModuleType('transformers'), types.ModuleType('transformers.utils')
    stub.PretrainedConfig, stub.PreTrainedModel, stub.__version__ = PretrainedConfig, PreTrainedModel, 'light (pinned model code only)'
    utils.ModelOutput = type('ModelOutput', (), {})
    stub.utils = utils
    sys.modules.update(transformers=stub, **{'transformers.utils': utils})
    package = types.ModuleType('mert2_light')
    package.__path__ = [str(folder)]
    sys.modules['mert2_light'] = package
    modules = {}
    for name in ('configuration_mert2', 'modeling_mert2'):
        spec = importlib.util.spec_from_file_location(f'mert2_light.{name}', folder/f'{name}.py')
        modules[name] = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = modules[name]
        spec.loader.exec_module(modules[name])
    values = {k: v for k, v in json.loads((folder/'config.json').read_text()).items() if k not in ('architectures', 'auto_map', 'model_type', 'torch_dtype', 'transformers_version')}
    config = modules['configuration_mert2'].MERT2Config(**values)
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        return modules['modeling_mert2'].MERT2Model(config)


def fast_extractor(folder):
    def make_extractor(device, source):
        from safetensors.torch import load_file
        from noteforge.audio.local_mert import CpuMelTransfer, LocalFrozenMert2
        from noteforge.audio.mert2 import Mert2Spec
        from noteforge.audio.mert_attention import install_query_chunks
        if device != 'mps':
            raise SystemExit('the MLX fast feature path uses PyTorch MPS for MERT; pass --mert-device mps')
        tick = time.monotonic()
        model = light_model(folder)
        result = model.load_state_dict(load_file(str(folder/'model.safetensors')), strict=False, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f'MERT weights do not match: {result.missing_keys[:3]} {result.unexpected_keys[:3]}')
        model.feature_extractor = model.feature_extractor.float()
        model = model.to(device).eval().requires_grad_(False)
        model.feature_extractor = CpuMelTransfer(model.feature_extractor, device)
        install_query_chunks(model, 512)
        print('fast mert load s', round(time.monotonic()-tick, 2), flush=True)
        return LocalFrozenMert2(model, Mert2Spec()), 'mps_bf16_fast'
    return make_extractor


if __name__ == '__main__':
    if sys.argv[1] == 'export':
        export(sys.argv[2], sys.argv[3])
    else:
        folder = Path(sys.argv[2])
        ds.make_extractor = fast_extractor(folder)
        ds.resolve_mert_source = lambda arg, root=None: (None, dict(kind='bf16_export', path=str(folder)), None)
        ds.main(sys.argv[3:])
