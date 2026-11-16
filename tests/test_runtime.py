import numpy as np

from noteforge.backends import resolve_backend, resolve_torch_device
from noteforge.model.local_song_rollout import _device_rows


def test_backend_auto_never_names_a_hardware_vendor():
    target = resolve_backend('auto', 'cpu')
    assert target.backend == 'torch'
    assert target.device == 'cpu'
    assert target.error is None


def test_torch_auto_resolves_to_a_real_device():
    assert resolve_torch_device('auto') in ('cuda', 'mps', 'cpu')


def test_rollout_device_rows_unwrap_the_collator_adapter():
    row = dict(inputs=dict(mode=np.asarray(1, np.int64), controls=np.zeros((2, 3), np.float32)),
               sequence={"adapter": True})
    inputs = _device_rows([row], None, 'cpu')
    assert tuple(inputs['mode'].shape) == (1,)
    assert tuple(inputs['controls'].shape) == (1, 2, 3)


def test_explicit_unavailable_cuda_fails_closed():
    try:
        device = resolve_torch_device('cuda')
    except RuntimeError as exc:
        assert 'CUDA was requested' in str(exc)
    else:
        assert device == 'cuda'


def test_mlx_backend_use_is_reported_without_claiming_metal_as_a_torch_device():
    try:
        from noteforge.mlx import draft_mlx
    except ImportError:
        return
    args = draft_mlx.parse_mlx_args(['--model', 'm', '--audio', 'a', '--out-dir', 'o', '--mode', 'phira',
                                     '--device', 'auto'])
    assert args.device == 'cpu'
    assert args.mert_device == 'mps'


def test_phira_layout_handles_an_empty_draft():
    import torch
    from noteforge.placement.apply import layout
    from noteforge.placement.model import M
    net = M()
    assert layout([], net, seed=0) == []


def test_placement_import_does_not_override_cpu_threads():
    import importlib
    import torch
    before = torch.get_num_threads()
    import noteforge.placement.model as placement_model
    importlib.reload(placement_model)
    assert torch.get_num_threads() == before


def test_meta_model_construction_matches_the_plain_path():
    """load_model builds on meta and assigns weights; the result must equal plain construct + load."""
    import torch
    from noteforge.model.full_event_codec import EventVocabulary
    from noteforge.model.full_event_model import FullEventConfig, FullEventModel
    from noteforge.scripts.draft_song import build_model_on_meta
    torch.manual_seed(0)
    config = FullEventConfig(
        d=64, heads=4, kv_heads=2, ffn=128, encoder_layers=1, decoder_layers=2,
        tap_dim=32, taps=1, bands=2, band_slots=1, reference_dim=4, controls=2, dropout=0.)
    vocabulary = EventVocabulary()
    state = FullEventModel(config, vocabulary).state_dict()
    plain = FullEventModel(config, vocabulary)
    plain.load_state_dict(state, strict=True)
    meta = build_model_on_meta(config, vocabulary, state)
    assert not any(p.is_meta for p in meta.parameters())
    left, right = dict(plain.named_parameters()), dict(meta.named_parameters())
    assert set(left) == set(right)
    assert all(torch.equal(left[k], right[k]) for k in left)
